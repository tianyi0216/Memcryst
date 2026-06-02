"""Synthetic pilot. End to end sanity check without any environment.

We seed the memory bank with a small set of hand crafted ALFWorld style
episodes. We then run the sleep pipeline once. The expected outcome is at
least one cluster gets discovered, the synthesizer produces a SKILL.md, the
audit returns a decision, and the script prints everything for inspection.

Run with

    export OPENAI_API_KEY=...
    python -m memcryst.examples.synthetic_pilot

Set MEMCRYST_STORE_DIR to use a custom directory. By default the script
writes to a temporary directory so repeated runs do not pollute the home
directory database.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.core import LLMClient
from memcryst.memory_bank import (
    EntryStatus,
    MemoryEntry,
    MemoryRetriever,
    MemoryStore,
)
from memcryst.skill_bank import SkillStore, SkillRetriever
from memcryst.sleep import (
    FeedbackLoop,
    PatternMiner,
    SkillAuditor,
    SkillSynthesizer,
    SleepPipeline,
)


console = Console()


def seed_entries() -> list[MemoryEntry]:
    """Five clean clusters of ALFWorld style episodes plus light noise."""
    entries = []

    # Cluster A. Pick up then clean object at sink.
    entries += [
        MemoryEntry(
            question="Clean the apple on the countertop and place it on the dining table.",
            failure_reason="First attempt skipped washing at the sink, agent moved apple directly.",
            improvement="Always carry the object to the sink and use the sink before moving it to the final location.",
            missed_information="Sink position. The agent must navigate to the sink before any clean task.",
            category="general",
            weight=1.3,
        ),
        MemoryEntry(
            question="Clean a dirty plate and put it on the dining table.",
            failure_reason="Agent attempted to clean using a cloth instead of the sink.",
            improvement="Bring the plate to the sink and use the sink action before placing the plate.",
            category="general",
            weight=1.4,
        ),
        MemoryEntry(
            question="Wash the bowl and place it in the cabinet.",
            failure_reason="No failure noted, agent succeeded by going to sink first.",
            improvement="The reliable pattern is move to sink, then sink the object, then move to destination.",
            category="general",
            weight=1.5,
        ),
        MemoryEntry(
            question="Clean the egg and put it in the fridge.",
            failure_reason="Initial run forgot to put the egg down before opening the fridge.",
            improvement="Open the fridge first then place the egg, never put the egg down on counter.",
            category="general",
            weight=1.2,
        ),
        MemoryEntry(
            question="Wash the spoon and place it in the drawer.",
            failure_reason="Trivial run, no failure.",
            improvement="Go to sink, sink the spoon, navigate to drawer, place the spoon.",
            category="general",
            weight=1.3,
        ),
    ]

    # Cluster B. Heat object in microwave.
    entries += [
        MemoryEntry(
            question="Heat a mug for coffee and bring it to the sofa.",
            failure_reason="Agent forgot to close the microwave door before starting.",
            improvement="Always close the microwave door before activating the appliance.",
            category="general",
            weight=1.4,
        ),
        MemoryEntry(
            question="Microwave a bowl of soup and put it on the dining table.",
            failure_reason="No failure, succeeded directly.",
            improvement="Open microwave, place bowl, close microwave, activate microwave, retrieve bowl.",
            category="general",
            weight=1.5,
        ),
        MemoryEntry(
            question="Heat the bread roll and place it on the side board.",
            failure_reason="Agent left the microwave door open after retrieving the bread.",
            improvement="Close the microwave door after retrieving the food, even though the task does not require it.",
            category="general",
            weight=1.1,
        ),
        MemoryEntry(
            question="Warm a cup of tea and bring it to the bedside table.",
            failure_reason="Trivial success after open close protocol.",
            improvement="Follow the open place close activate retrieve sequence reliably.",
            category="general",
            weight=1.3,
        ),
    ]

    # Cluster C. Examine in light, three episodes only, less reliable.
    entries += [
        MemoryEntry(
            question="Examine the book under the desk lamp in the office.",
            failure_reason="Agent forgot to turn on the lamp before examining.",
            improvement="Turn on the lamp first, then examine the object held in hand.",
            category="general",
            weight=1.0,
        ),
        MemoryEntry(
            question="Examine the pen using the floor lamp.",
            failure_reason="Trivial success.",
            improvement="Bring object near the lamp first, then turn on the lamp, then examine.",
            category="general",
            weight=1.0,
        ),
        MemoryEntry(
            question="Examine the photo album using the desk light.",
            failure_reason="No failure noted, examined under lamp.",
            improvement="Same lamp on then examine pattern.",
            category="general",
            weight=1.1,
        ),
    ]

    # Noise. Singleton entries that should not cluster.
    entries += [
        MemoryEntry(
            question="Find the cat in the garden and report its location.",
            failure_reason="Out of distribution task.",
            improvement="Not actionable under this agent.",
            category="general",
            weight=0.3,
        ),
        MemoryEntry(
            question="Translate a poem from German to Mandarin.",
            failure_reason="Not in the ALFWorld action space.",
            improvement="None.",
            category="general",
            weight=0.2,
        ),
    ]

    return entries


def run_pilot(
    store_dir: Optional[Path],
    synth_model: str,
    audit_model: str,
    base_url: Optional[str],
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    console.print(Panel.fit("MEMCRYST synthetic pilot", style="bold cyan"))
    console.print(f"Store dir: {store_dir}")
    console.print(f"Synth model: {synth_model}")
    console.print(f"Audit model: {audit_model}\n")

    memory_store = MemoryStore(store_dir=store_dir / "memory")
    skill_store = SkillStore(store_dir=store_dir / "skills")
    memory_retriever = MemoryRetriever(memory_store)
    skill_retriever = SkillRetriever(skill_store)

    # Seed memory bank.
    entries = seed_entries()
    memory_store.add_many(entries)
    console.print(f"Seeded {len(entries)} memory entries\n")

    # Show retrieval sanity check.
    retrieved = memory_retriever.retrieve("Clean the carrot at the sink and put it on the table", top_k=3)
    table = Table(title="Top 3 retrieval for a fresh clean query")
    table.add_column("score")
    table.add_column("question")
    table.add_column("weight")
    for entry, score in retrieved:
        table.add_row(f"{score:.3f}", entry.question, f"{entry.weight:.2f}")
    console.print(table)

    # Build clients.
    synth_llm = LLMClient(model=synth_model, base_url=base_url, temperature=0.3)
    audit_llm = LLMClient(model=audit_model, base_url=base_url, temperature=0.0)

    # Sleep pipeline.
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
        max_candidates_per_run=5,
        archive_on_accept=True,
    )

    report = pipeline.run()

    console.rule("Sleep report")
    console.print(f"Took {report.finished_at - report.started_at:.2f}s")
    console.print(f"Candidates considered: {len(report.candidates)}")
    console.print(f"Accepted skills: {len(report.accepted_skills)}")
    console.print(f"Flag events: {len(report.flag_events)}\n")

    for outcome in report.candidates:
        console.rule(f"Cluster {outcome.cluster.cluster_id} size {outcome.cluster.size}")
        for entry in outcome.cluster.members:
            console.print(f"  - {entry.question}")

        if outcome.synthesis.error:
            console.print(Panel(outcome.synthesis.error, title="synthesis error", style="red"))
            continue

        skill = outcome.synthesis.skill
        if skill is not None:
            console.print(
                Panel.fit(
                    f"name: {skill.name}\ndescription: {skill.description}\ncategory: {skill.category}",
                    title="synthesized skill frontmatter",
                    style="green",
                )
            )
            console.print(Panel(skill.body.strip(), title="skill body", style="dim"))

        if skill is not None and skill.capability.is_complete():
            cap = skill.capability
            console.print(
                Panel.fit(
                    f"verb_family: {cap.verb_family}\n"
                    f"target_type:  {cap.target_type}\n"
                    f"affordance:  {cap.affordance}\n"
                    f"output:      {cap.output}\n"
                    f"preconditions: {cap.preconditions}\n"
                    f"task_pattern: {cap.task_pattern}",
                    title="capability signature",
                    style="cyan",
                )
            )
        elif skill is not None:
            console.print(
                Panel.fit(
                    f"incomplete: {skill.capability.to_dict()}",
                    title="capability signature (incomplete)",
                    style="yellow",
                )
            )

        if outcome.audit is not None:
            verdict_style = "green" if outcome.accepted else "red"
            console.print(
                Panel(
                    f"decision: {outcome.audit.decision}\n"
                    f"pass_count: {outcome.audit.pass_count} of 13\n"
                    f"overall_reason: {outcome.audit.overall_reason}",
                    title="audit result",
                    style=verdict_style,
                )
            )
            checks_table = Table(title="check by check")
            checks_table.add_column("id")
            checks_table.add_column("verdict")
            checks_table.add_column("reason", overflow="fold")
            for check in outcome.audit.checks:
                checks_table.add_row(str(check.get("id")), check["verdict"], check["reason"])
            console.print(checks_table)

    # ---- Capability-indexed retrieval demo ----
    console.rule("Capability-indexed retrieval demo (B-path O(K) lookup)")
    skill_retriever_with_llm = SkillRetriever(skill_store, signature_llm=audit_llm)
    demo_queries = [
        "Clean the carrot and place it on the dining table",
        "Heat the pizza slice and put it on the counter",
        "Examine the journal under the desk lamp",
        "Search the cupboard for the salt shaker",  # no skill should match
    ]
    for q in demo_queries:
        # Extract signature first to show what the index sees.
        cap = skill_retriever_with_llm.signature_extractor.extract(q)
        console.print(
            f"\n[bold cyan]Task[/bold cyan]: {q}\n"
            f"  extracted sig: verb={cap.verb_family} affordance={cap.affordance} output={cap.output}"
        )
        results = skill_retriever_with_llm.retrieve_by_capability(q, top_k=3)
        if not results:
            console.print("  [yellow]no matching skill, capability bucket empty[/yellow]")
            continue
        for skill, score, meta in results:
            console.print(
                f"  [green]match[/green] skill={skill.name}  score={score:.3f}  path={meta['path']}"
            )

    console.rule("Final bank state")
    console.print(memory_store.stats())
    console.print(skill_store.stats())


def main() -> int:
    parser = argparse.ArgumentParser(description="MEMCRYST synthetic pilot")
    parser.add_argument(
        "--store-dir",
        type=str,
        default="",
        help="custom MEMCRYST store directory, default uses a fresh temp directory",
    )
    parser.add_argument("--synth-model", default="gpt-4o-mini")
    parser.add_argument("--audit-model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--keep", action="store_true", help="keep the store directory after the run")
    args = parser.parse_args()

    if args.store_dir:
        store_dir = Path(args.store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
    else:
        store_dir = Path(tempfile.mkdtemp(prefix="memcryst-pilot-"))

    cleanup = not args.keep and not args.store_dir
    try:
        run_pilot(
            store_dir=store_dir,
            synth_model=args.synth_model,
            audit_model=args.audit_model,
            base_url=args.base_url,
        )
    finally:
        if cleanup:
            shutil.rmtree(store_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
