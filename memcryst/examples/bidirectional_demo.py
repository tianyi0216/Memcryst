"""Bidirectional feedback closed-loop demo.

The MEMCRYST paper's punch line: skill failure → axis-level M3 feedback →
memory reweight → re-crystallize → v2 skill avoids the failure mode. This
script demonstrates the loop end to end on the ALFWorld setup.

Workflow:

1. Load v1 memory + v1 skills from --input-dir (assumed already built via
   seed_alfworld_skills).
2. Apply curated failure signals derived from a round 1 eval:
     - For each family, record axis failures on the matching skill via
       SkillStore.record_axis_failure().
     - Add new memory entries capturing the lessons learned (these would
       come from automated M3 axis attribution + agent trajectory writeback
       in a fully automated pipeline; for the PoC demo they are curated).
3. Run M3 FeedbackLoop to flag affected ancestry memory.
4. Run SleepPipeline → produces v2 skill bank at --output-dir.
5. Print v1 vs v2 skill body diffs for the affected families.

User then re-runs alfworld_eval against the v2 skill bank to measure round
2 success rate. The headline result of the paper is round 2 > round 1 on
the failure-prone families.

Usage:

    export OPENAI_API_KEY="sk-..."
    python -m memcryst.examples.bidirectional_demo \\
        --input-dir /tmp/memcryst-alfworld \\
        --output-dir /tmp/memcryst-alfworld-v2

The eval comparison is then:

    python -m memcryst.examples.alfworld_eval \\
        --skills-dir /tmp/memcryst-alfworld-v2/skills \\
        --modes capability \\
        --num-tasks 20
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.core import LLMClient
from memcryst.memory_bank import MemoryEntry, MemoryRetriever, MemoryStore
from memcryst.skill_bank import SkillStore
from memcryst.sleep import (
    FeedbackLoop,
    PatternMiner,
    SkillAuditor,
    SkillSynthesizer,
    SleepPipeline,
)


console = Console()


# Curated round 1 failure attribution. In a fully automated pipeline these
# would come from agent trajectory writeback + LLM-based axis attribution.
# Here we hand-curate them to demonstrate the loop. Each entry captures one
# observed failure mode from the v1 ALFWorld eval (batch 20, June 2026).
CURATED_FEEDBACK = [
    {
        "skill_family": "cool",
        "skill_match_keyword": "cool-and-place",
        "axis_failures": {"affordance": 3, "target_type": 1},
        "new_memory_entry": MemoryEntry(
            question="Put a cool tomato in microwave when the microwave already contains another object.",
            improvement=(
                "Locate the tomato (often on a countertop), take it, go to the fridge, cool the tomato "
                "with the fridge, then go to the microwave. CRITICAL: if the microwave is closed, open it "
                "first; if it already contains another object such as an egg, the put action may still "
                "succeed in ALFWorld, but the agent should not retry put repeatedly on failure — instead "
                "use the env primitive `move tomato 1 to microwave 1` not `put tomato 1 in microwave 1`."
            ),
            failure_reason=(
                "Round 1 cool tasks failed because the agent emitted `put tomato 1 in/on microwave 1` "
                "but ALFWorld primitive is `move <obj> to <recep>`. The agent looped on close/open of "
                "microwave instead of attempting move. Destination clearing was not the real blocker — "
                "the verb mismatch was."
            ),
            category="general",
            weight=1.5,
        ),
    },
    {
        "skill_family": "heat",
        "skill_match_keyword": "heat-and-place",
        "axis_failures": {"affordance": 2, "verb_family": 1},
        "new_memory_entry": MemoryEntry(
            question="Heat some potato and put it in the garbagecan when the potato is hard to locate.",
            improvement=(
                "Take potato from its source location (often a countertop), go to the microwave, heat "
                "the potato with the microwave in one atomic step. Use the env primitive `move potato 1 "
                "to garbagecan 1` for placement, not `put`. Verify the agent is at the destination "
                "before issuing the placement primitive."
            ),
            failure_reason=(
                "Round 1 heat tasks failed when the agent emitted take from a stale obs reference. The "
                "agent must re-verify the object is actually at the receptacle before issuing take. "
                "Also the placement verb mismatch from `put` to `move` blocked completion."
            ),
            category="general",
            weight=1.5,
        ),
    },
    {
        "skill_family": "clean",
        "skill_match_keyword": "clean-and-place",
        "axis_failures": {"affordance": 1, "preconditions": 2},
        "new_memory_entry": MemoryEntry(
            question="Put a clean mug in coffeemachine when the mug is in a cabinet.",
            improvement=(
                "Locate mug by searching cabinets in order. Once found, take mug. Go to sink, clean mug "
                "with sink. Then go to coffeemachine. If coffeemachine is closed, open it. Use env "
                "primitive `move mug 1 to coffeemachine 1` for final placement. Do not skip the sink "
                "step even if the mug appears clean."
            ),
            failure_reason=(
                "Round 1 clean tasks failed when the agent skipped the sink step or emitted the wrong "
                "placement verb. The clean modifier always requires the sink primitive, and placement "
                "uses move not put."
            ),
            category="general",
            weight=1.4,
        ),
    },
]


def apply_curated_feedback(
    memory_store: MemoryStore,
    skill_store: SkillStore,
) -> dict:
    """Apply the curated round 1 feedback to memory + skill bank.

    Returns a report dict for logging.
    """
    report = {
        "axis_failures_recorded": 0,
        "new_memory_entries": 0,
        "skills_touched": [],
    }

    for item in CURATED_FEEDBACK:
        keyword = item["skill_match_keyword"]
        matching_skills = [s for s in skill_store if keyword in s.name]
        for skill in matching_skills:
            for axis, count in item["axis_failures"].items():
                for _ in range(count):
                    skill_store.record_axis_failure(skill.id, axis)
                    report["axis_failures_recorded"] += 1
            report["skills_touched"].append(skill.name)

        memory_store.add(item["new_memory_entry"])
        report["new_memory_entries"] += 1

    return report


def diff_skills_v1_v2(v1_store: SkillStore, v2_store: SkillStore) -> Table:
    """Show v1 skill body vs v2 skill body for each family that changed."""
    table = Table(title="v1 vs v2 skill body diff")
    table.add_column("v1 name")
    table.add_column("v2 name")
    table.add_column("v1 procedure", overflow="fold")
    table.add_column("v2 procedure", overflow="fold")

    v1_by_name = {s.name: s for s in v1_store}
    v2_by_name = {s.name: s for s in v2_store}

    def _short_proc(body: str) -> str:
        lines = body.splitlines()
        in_proc = False
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("## procedure"):
                in_proc = True
                continue
            if in_proc and stripped.startswith("## "):
                break
            if in_proc and stripped:
                out.append(stripped)
        return "\n".join(out[:8])

    matched = set()
    for n in sorted(set(v1_by_name) | set(v2_by_name)):
        v1 = v1_by_name.get(n)
        v2 = v2_by_name.get(n)
        if v1 and v2:
            table.add_row(v1.name, v2.name, _short_proc(v1.body), _short_proc(v2.body))
            matched.add(n)
        elif v1:
            table.add_row(v1.name, "(removed)", _short_proc(v1.body), "")
        elif v2:
            table.add_row("(new)", v2.name, "", _short_proc(v2.body))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="MEMCRYST bidirectional feedback demo")
    parser.add_argument("--input-dir", required=True, help="v1 memory+skills dir")
    parser.add_argument("--output-dir", required=True, help="v2 output dir")
    parser.add_argument("--synth-model", default="gpt-4o-mini")
    parser.add_argument("--audit-model", default="gpt-4o-mini")
    parser.add_argument(
        "--dump-feedback-json",
        default="",
        help="dump the curated feedback signals as JSON for reproducibility",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy v1 memory + skills into output dir so v2 builds on top.
    shutil.copytree(input_dir / "memory", output_dir / "memory")
    shutil.copytree(input_dir / "skills", output_dir / "skills")

    console.print(Panel.fit("MEMCRYST bidirectional feedback closed loop", style="bold cyan"))
    console.print(f"v1: {input_dir}")
    console.print(f"v2: {output_dir}\n")

    memory_store = MemoryStore(store_dir=output_dir / "memory")
    skill_store_v1_view = SkillStore(store_dir=output_dir / "skills")
    console.print(
        f"loaded v1: {len(memory_store)} memory entries, {len(skill_store_v1_view)} skills\n"
    )

    # Step 1: Apply curated round-1 feedback.
    console.rule("step 1: apply curated round-1 feedback")
    report = apply_curated_feedback(memory_store, skill_store_v1_view)
    console.print(
        f"axis failures recorded: {report['axis_failures_recorded']}\n"
        f"new memory entries: {report['new_memory_entries']}\n"
        f"skills touched: {report['skills_touched']}\n"
    )

    if args.dump_feedback_json:
        Path(args.dump_feedback_json).write_text(
            json.dumps(
                [
                    {
                        "skill_family": x["skill_family"],
                        "skill_match_keyword": x["skill_match_keyword"],
                        "axis_failures": x["axis_failures"],
                        "new_memory_question": x["new_memory_entry"].question,
                    }
                    for x in CURATED_FEEDBACK
                ],
                indent=2,
            )
        )

    # Step 2: Run M3 FeedbackLoop to flag affected ancestry memory.
    console.rule("step 2: M3 axis-level feedback over ancestry")
    feedback = FeedbackLoop(
        skill_store_v1_view,
        memory_store,
        failure_threshold=0.1,
        min_uses=1,
    )
    flag_events = feedback.run()
    console.print(
        f"M3 produced {len(flag_events)} flag events. "
        f"Dominant axes: {[e.dominant_axis for e in flag_events]}"
    )
    for e in flag_events:
        console.print(
            f"  - skill {e.skill_name}: severity {e.severity:.2f}, "
            f"{len(e.flagged_entry_ids)} entries flagged "
            f"({len(e.axis_amplified_entry_ids)} amplified on axis {e.dominant_axis})"
        )
    console.print(
        f"\nmemory bank after M3: {memory_store.stats()['flagged']} flagged, "
        f"{memory_store.stats()['active']} active\n"
    )

    # Step 3: Re-crystallize. Old v1 skills stay in place; new skills are
    # added or replace via the standard sleep pipeline.
    console.rule("step 3: re-crystallize (sleep) over updated memory")
    retriever = MemoryRetriever(memory_store)
    synth_llm = LLMClient(model=args.synth_model, temperature=0.3)
    audit_llm = LLMClient(model=args.audit_model, temperature=0.0)
    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=retriever,
        skill_store=skill_store_v1_view,  # writes v2 skills into the same dir
        miner=PatternMiner(retriever, min_cluster_size=3, min_mean_reward=0.4),
        synthesizer=SkillSynthesizer(synth_llm),
        auditor=SkillAuditor(audit_llm, min_pass=10),
        feedback=FeedbackLoop(skill_store_v1_view, memory_store),
        max_candidates_per_run=10,
        archive_on_accept=True,
    )
    report_v2 = pipeline.run()
    console.print(
        f"sleep run took {report_v2.finished_at - report_v2.started_at:.1f}s, "
        f"{len(report_v2.candidates)} candidates, "
        f"{len(report_v2.accepted_skills)} accepted into v2.\n"
    )

    # Step 4: Reload skills from disk to compare v1 vs v2 bodies.
    console.rule("step 4: v1 vs v2 skill diff")
    skill_store_v2 = SkillStore(store_dir=output_dir / "skills")
    console.print(diff_skills_v1_v2(skill_store_v1_view, skill_store_v2))

    console.rule("done")
    console.print(
        f"[green]v2 skills written to {output_dir / 'skills'}[/green]\n"
        f"Run eval with:\n"
        f"  python -m memcryst.examples.alfworld_eval \\\n"
        f"    --skills-dir {output_dir / 'skills'} \\\n"
        f"    --memory-dir {output_dir / 'memory'} \\\n"
        f"    --modes capability \\\n"
        f"    --num-tasks 20"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
