"""Day 3. Seed the memory bank with hand-crafted ALFWorld entries and
run one sleep cycle to crystallize them into skills.

Usage:

    export OPENAI_API_KEY="sk-..."
    python -m memcryst.examples.seed_alfworld_skills \\
        --store-dir /tmp/memcryst-alfworld

Output is a populated skill_store at <store-dir>/skills with 5 skills
(one per task family). Used by Day 4 eval as the frozen skill library.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.core import LLMClient
from memcryst.memory_bank import MemoryRetriever, MemoryStore
from memcryst.skill_bank import SkillStore
from memcryst.sleep import (
    FeedbackLoop,
    PatternMiner,
    SkillAuditor,
    SkillSynthesizer,
    SleepPipeline,
)
from memcryst.examples.alfworld_seed_data import alfworld_seed_entries


console = Console()


def run(store_dir: Path, synth_model: str, audit_model: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    console.print(Panel.fit("MEMCRYST Day 3, ALFWorld seed + crystallize", style="bold cyan"))
    console.print(f"Store dir: {store_dir}")
    console.print(f"Synth model: {synth_model}")
    console.print(f"Audit model: {audit_model}\n")

    memory_store = MemoryStore(store_dir=store_dir / "memory")
    skill_store = SkillStore(store_dir=store_dir / "skills")

    entries = alfworld_seed_entries()
    memory_store.add_many(entries)
    console.print(f"Seeded {len(entries)} memory entries across 5 ALFWorld families\n")

    # Sanity: show distribution of entries by first verb.
    from collections import Counter
    from memcryst.skill_bank.constants import canonicalize_verb

    verb_counts: Counter = Counter()
    for e in entries:
        first = (e.question or "").strip().lower().split()[0].rstrip(".,!?;:")
        verb_counts[canonicalize_verb(first) or first] += 1
    dist_table = Table(title="seed entries by first-verb")
    dist_table.add_column("verb")
    dist_table.add_column("count")
    for v, c in verb_counts.most_common():
        dist_table.add_row(str(v), str(c))
    console.print(dist_table)
    console.print()

    memory_retriever = MemoryRetriever(memory_store)
    synth_llm = LLMClient(model=synth_model, temperature=0.3)
    audit_llm = LLMClient(model=audit_model, temperature=0.0)

    miner = PatternMiner(memory_retriever, min_cluster_size=3, min_mean_reward=0.4)
    synthesizer = SkillSynthesizer(synth_llm)
    auditor = SkillAuditor(audit_llm, min_pass=10)
    feedback = FeedbackLoop(skill_store, memory_store)

    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=memory_retriever,
        skill_store=skill_store,
        miner=miner,
        synthesizer=synthesizer,
        auditor=auditor,
        feedback=feedback,
        max_candidates_per_run=10,
        archive_on_accept=True,
    )

    console.rule("running sleep pipeline")
    report = pipeline.run()

    console.print(
        f"Took {report.finished_at - report.started_at:.2f}s. "
        f"Candidates considered: {len(report.candidates)}. "
        f"Accepted skills: {len(report.accepted_skills)}.\n"
    )

    # Per-cluster summary.
    for outcome in report.candidates:
        verb = (outcome.cluster.members[0].question or "").strip().lower().split()[0].rstrip(".,!?;:")
        canonical_verb = canonicalize_verb(verb) or verb
        console.rule(
            f"cluster {outcome.cluster.cluster_id} verb={canonical_verb} size={outcome.cluster.size}"
        )
        for entry in outcome.cluster.members:
            console.print(f"  - {entry.question}")
        if outcome.synthesis.error:
            console.print(
                Panel(outcome.synthesis.error, title="synthesis error", style="red")
            )
            continue
        skill = outcome.synthesis.skill
        if skill is not None:
            console.print(
                Panel.fit(
                    f"name: {skill.name}\n"
                    f"description: {skill.description}\n"
                    f"capability: verb={skill.capability.verb_family} "
                    f"affordance={skill.capability.affordance} "
                    f"output={skill.capability.output}",
                    title="synthesized skill",
                    style="green",
                )
            )
            console.print(Panel(skill.body.strip(), title="skill body", style="dim"))
        if outcome.audit is not None:
            verdict_style = "green" if outcome.accepted else "red"
            console.print(
                Panel(
                    f"decision: {outcome.audit.decision}, "
                    f"pass_count: {outcome.audit.pass_count} of 13",
                    title="audit",
                    style=verdict_style,
                )
            )

    # Final state.
    console.rule("final bank state")
    console.print(memory_store.stats())
    console.print(skill_store.stats())

    # Verify capability index coverage.
    console.rule("capability index coverage")
    cap_table = Table()
    cap_table.add_column("skill")
    cap_table.add_column("verb")
    cap_table.add_column("affordance")
    cap_table.add_column("output")
    cap_table.add_column("indexed")
    for s in skill_store:
        cap_table.add_row(
            s.name,
            s.capability.verb_family or "MISSING",
            s.capability.affordance or "MISSING",
            s.capability.output or "MISSING",
            "yes" if s.capability.is_complete() else "no",
        )
    console.print(cap_table)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ALFWorld memory seed and crystallize")
    parser.add_argument(
        "--store-dir",
        default="",
        help="store directory, default uses a temp dir that is kept after run",
    )
    parser.add_argument("--synth-model", default="gpt-4o-mini")
    parser.add_argument("--audit-model", default="gpt-4o-mini")
    args = parser.parse_args()

    if args.store_dir:
        store_dir = Path(args.store_dir)
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
    else:
        store_dir = Path(tempfile.mkdtemp(prefix="memcryst-alfworld-"))

    return run(store_dir, args.synth_model, args.audit_model)


if __name__ == "__main__":
    raise SystemExit(main())
