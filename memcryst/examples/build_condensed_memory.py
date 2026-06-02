"""Build the memory_condense baseline bank.

Takes a raw memory bank (the same one MEMCRYST crystallizes from), runs the
M1 PatternMiner to cluster it, asks an LLM to summarize each cluster into
one NL paragraph, and stores the result as a CondensedMemoryStore.

The output is NOT a skill bank. There is no capability tag, no procedure
section, no 9+3 audit, no ancestry feedback. It is plain LLM compression of
clustered experiences — the strong baseline that MEMCRYST must beat to
justify the crystallization step in the paper.

Usage:

    export OPENAI_API_KEY="sk-..."
    python -m memcryst.examples.build_condensed_memory \\
        --memory-dir /tmp/memcryst-alfworld/memory \\
        --output-dir /tmp/memcryst-alfworld/condensed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.baselines import build_condensed_memory
from memcryst.core import LLMClient
from memcryst.memory_bank import MemoryStore

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build memory_condense baseline bank")
    parser.add_argument("--memory-dir", required=True, help="raw memory bank dir")
    parser.add_argument("--output-dir", required=True, help="where to write condensed entries")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-mean-reward", type=float, default=0.4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    console.print(Panel.fit("memory_condense baseline builder", style="bold cyan"))
    console.print(f"Source memory: {args.memory_dir}")
    console.print(f"Output: {args.output_dir}\n")

    memory_store = MemoryStore(store_dir=Path(args.memory_dir))
    if len(memory_store) == 0:
        console.print(f"[red]memory store empty at {args.memory_dir}[/red]")
        return 2
    console.print(f"loaded {len(memory_store)} raw memory entries")

    llm = LLMClient(model=args.model, temperature=0.3)

    condensed_store = build_condensed_memory(
        memory_store=memory_store,
        llm=llm,
        output_dir=Path(args.output_dir),
        min_cluster_size=args.min_cluster_size,
        min_mean_reward=args.min_mean_reward,
    )

    console.rule("condensed bank")
    table = Table()
    table.add_column("id")
    table.add_column("cluster_size")
    table.add_column("summary length")
    table.add_column("summary preview", overflow="fold")
    for e in condensed_store:
        table.add_row(
            e.id,
            str(e.cluster_size),
            f"{len(e.summary)} chars",
            e.summary[:200] + ("..." if len(e.summary) > 200 else ""),
        )
    console.print(table)
    console.print(f"\n[green]wrote {len(condensed_store)} condensed entries to {args.output_dir}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
