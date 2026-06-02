"""Offline smoke test. No API key required.

Verifies that the memory bank, skill bank, retriever, and pattern miner all
import and run end to end. Skips synthesis and audit because those need
network access. Useful as the first sanity check after installation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from memcryst.memory_bank import (
    EntryStatus,
    MemoryEntry,
    MemoryRetriever,
    MemoryStore,
)
from memcryst.skill_bank import Skill, SkillStore, SkillRetriever
from memcryst.sleep import PatternMiner


console = Console()


def main() -> int:
    console.rule("MEMCRYST offline smoke test")
    tmp = Path(tempfile.mkdtemp(prefix="memcryst-smoke-"))
    console.print(f"using temp dir {tmp}")

    memory_store = MemoryStore(store_dir=tmp / "memory")
    skill_store = SkillStore(store_dir=tmp / "skills")

    # Seed a few entries.
    entries = [
        MemoryEntry(
            question="Clean the apple on the countertop.",
            failure_reason="Skipped sink.",
            improvement="Use the sink before placing.",
            category="general",
            weight=1.3,
        ),
        MemoryEntry(
            question="Wash the plate then place on the table.",
            failure_reason="None.",
            improvement="Go to sink first.",
            category="general",
            weight=1.4,
        ),
        MemoryEntry(
            question="Clean the cup and put it in the cabinet.",
            failure_reason="Forgot to navigate to sink.",
            improvement="Sink is always the first stop.",
            category="general",
            weight=1.5,
        ),
        MemoryEntry(
            question="Heat the mug in the microwave.",
            failure_reason="Open door not closed.",
            improvement="Close the microwave before activation.",
            category="general",
            weight=1.3,
        ),
        MemoryEntry(
            question="Microwave the bowl of soup.",
            failure_reason="None.",
            improvement="Open close activate retrieve.",
            category="general",
            weight=1.4,
        ),
    ]
    memory_store.add_many(entries)
    console.print(f"added {len(entries)} entries")

    retriever = MemoryRetriever(memory_store)
    retriever.ensure_embeddings()

    results = retriever.retrieve("Clean a fruit at the sink and put it on the dining table", top_k=3)
    table = Table(title="retrieval check")
    table.add_column("score")
    table.add_column("question")
    for entry, score in results:
        table.add_row(f"{score:.3f}", entry.question)
    console.print(table)

    miner = PatternMiner(retriever, min_cluster_size=2, min_mean_reward=0.3)
    candidates = miner.mine()
    console.print(f"miner returned {len(candidates)} candidate clusters")
    for c in candidates:
        console.print(
            f"  cluster {c.cluster_id} size {c.size} mean_reward {c.mean_reward:.3f} "
            f"persistence {c.persistence:.3f}"
        )
        for entry in c.members[:3]:
            console.print(f"    - {entry.question}")

    # Round trip a fake skill through the skill store.
    fake_skill = Skill(
        name="clean-then-place",
        description="Sample skill for smoke test.",
        body="## When to use\nUse for clean then place tasks.\n\n## Procedure\n1. Go to sink.\n2. Use sink.\n3. Place at target.",
        category="general",
        ancestry=[e.id for e in entries[:3]],
        crystallized_from_cluster=0,
        crystallization_round=1,
        utility=0.7,
    )
    skill_store.add(fake_skill)
    loaded_back = SkillStore(store_dir=tmp / "skills")
    found = loaded_back.get(fake_skill.id)
    console.print(f"reloaded skill name: {found.name if found else 'missing'}")

    skill_retriever = SkillRetriever(loaded_back)
    sk_results = skill_retriever.retrieve("Clean the apple and put it on the table", top_k=3)
    sk_table = Table(title="skill retrieval check")
    sk_table.add_column("score")
    sk_table.add_column("name")
    for skill, score in sk_results:
        sk_table.add_row(f"{score:.3f}", skill.name)
    console.print(sk_table)

    console.print("offline smoke test completed", style="green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
