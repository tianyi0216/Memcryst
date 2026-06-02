"""Day 2 smoke test. Run ReAct agent on ALFWorld task(s).

Modes:
- vanilla     no skill retrieval
- similarity  retrieve via O(N) cosine
- capability  retrieve via O(K) hash + relaxed fallback

Use --verbose to print each step's thought/action/obs.
Use --batch N to run a sequential batch and report success rate.

Usage:

    python -m memcryst.examples.alfworld_agent_smoke --task-idx 0 --verbose
    python -m memcryst.examples.alfworld_agent_smoke --batch 10
    python -m memcryst.examples.alfworld_agent_smoke --batch 5 --mode capability \\
        --skills-dir /tmp/memcryst-phase1/skills
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.agent import AlfworldEnv, ReactAgent, family_from_task_description
from memcryst.core import LLMClient
from memcryst.memory_bank import MemoryRetriever, MemoryStore
from memcryst.skill_bank import SkillRetriever, SkillStore


console = Console()


def make_step_printer():
    def cb(step_i: int, thought: str, action: str, obs: str, reward: float, done: bool, success: bool):
        console.print(
            f"[dim]step {step_i:02d}[/dim] "
            f"[yellow]thought:[/yellow] {thought[:120]}"
        )
        console.print(
            f"           [bold cyan]action:[/bold cyan] {action}"
        )
        console.print(
            f"           [dim]obs:[/dim] {obs[:200]}"
        )
        if done:
            tag = "[green]SUCCESS[/green]" if success else "[red]DONE (fail)[/red]"
            console.print(f"           {tag} reward={reward}")
    return cb


def run_one(env: AlfworldEnv, llm: LLMClient, skill_retriever, memory_retriever, args, task_idx: int):
    callback = make_step_printer() if args.verbose else None
    agent = ReactAgent(
        env=env,
        llm=llm,
        skill_retriever=skill_retriever,
        memory_retriever=memory_retriever,
        mode=args.mode,
        top_k_skills=args.top_k,
        max_steps=args.max_steps,
        step_callback=callback,
    )
    return agent.run(task_idx=task_idx)


def main() -> int:
    parser = argparse.ArgumentParser(description="ALFWorld agent smoke test")
    parser.add_argument("--split", default="eval_out_of_distribution")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0, help="run N sequential tasks and report aggregate")
    parser.add_argument("--mode", default="vanilla", choices=["vanilla", "memory", "similarity", "capability"])
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--memory-dir", default=None, help="required for --mode memory")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="print every step")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # keep step output clean
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    env = AlfworldEnv(split=args.split, max_steps=args.max_steps)
    llm = LLMClient(model=args.model, temperature=0.0)

    skill_retriever = None
    memory_retriever = None
    if args.mode in ("similarity", "capability"):
        if not args.skills_dir:
            console.print(f"[red]--skills-dir required for mode={args.mode}[/red]")
            return 2
        store = SkillStore(store_dir=Path(args.skills_dir))
        if len(store) == 0:
            console.print(f"[red]skill store at {args.skills_dir} is empty[/red]")
            return 2
        skill_retriever = SkillRetriever(store, signature_llm=llm)
        console.print(f"loaded {len(store)} skills from {args.skills_dir}\n")
    if args.mode == "memory":
        if not args.memory_dir:
            console.print(f"[red]--memory-dir required for mode=memory[/red]")
            return 2
        memstore = MemoryStore(store_dir=Path(args.memory_dir))
        if len(memstore) == 0:
            console.print(f"[red]memory store at {args.memory_dir} is empty[/red]")
            return 2
        memory_retriever = MemoryRetriever(memstore)
        console.print(f"loaded {len(memstore)} memory entries from {args.memory_dir}\n")

    if args.batch and args.batch > 1:
        results = []
        tasks = env.list_tasks(limit=args.batch)
        for tmeta in tasks:
            console.rule(f"task {tmeta.task_idx} family={tmeta.verb_family}")
            r = run_one(env, llm, skill_retriever, memory_retriever, args, tmeta.task_idx)
            console.print(
                f"  task_desc: {r.task_description}\n"
                f"  result: success={r.success} steps={r.steps} llm_calls={r.llm_calls} retries={r.invalid_action_retries}"
            )
            results.append((tmeta, r))

        # Aggregate by family detected from actual task description, not file path.
        console.rule("batch summary")
        agg_table = Table(title=f"{args.batch} tasks, mode={args.mode}")
        agg_table.add_column("verb_family")
        agg_table.add_column("n")
        agg_table.add_column("success")
        agg_table.add_column("avg_steps")
        agg_table.add_column("avg_llm_calls")
        by_family: dict[str, list] = {}
        for tmeta, r in results:
            true_family = family_from_task_description(r.task_description)
            by_family.setdefault(true_family, []).append(r)
        total_success = 0
        # Preserve family order alphabetically for readability.
        for fam in sorted(by_family.keys()):
            rs = by_family[fam]
            n = len(rs)
            succ = sum(1 for r in rs if r.success)
            total_success += succ
            avg_steps = sum(r.steps for r in rs) / n
            avg_calls = sum(r.llm_calls for r in rs) / n
            agg_table.add_row(fam, str(n), f"{succ}/{n} ({100*succ/n:.0f}%)", f"{avg_steps:.1f}", f"{avg_calls:.1f}")
        agg_table.add_row(
            "ALL", str(len(results)),
            f"{total_success}/{len(results)} ({100*total_success/len(results):.0f}%)",
            "", "",
        )
        console.print(agg_table)
        return 0

    # Single task
    console.rule(f"task {args.task_idx} mode={args.mode}")
    result = run_one(env, llm, skill_retriever, memory_retriever, args, args.task_idx)

    summary_table = Table(title="trajectory summary")
    summary_table.add_column("field")
    summary_table.add_column("value")
    summary_table.add_row("task_description", result.task_description)
    summary_table.add_row("mode", result.mode)
    summary_table.add_row("success", str(result.success))
    summary_table.add_row("steps", str(result.steps))
    summary_table.add_row("final_reward", f"{result.final_reward:.3f}")
    summary_table.add_row("llm_calls", str(result.llm_calls))
    summary_table.add_row("invalid_action_retries", str(result.invalid_action_retries))
    summary_table.add_row("used_skill_ids", ", ".join(result.used_skill_ids) or "<none>")
    summary_table.add_row("retrieval_paths", str(dict(result.retrieval_paths)) or "<none>")
    if result.error:
        summary_table.add_row("error", result.error)
    console.print(summary_table)

    style = "green" if result.success else "red"
    console.print(
        Panel(
            f"success={result.success} steps={result.steps}",
            title="result",
            style=style,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
