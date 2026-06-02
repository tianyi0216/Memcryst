"""Day 4. Three-mode ALFWorld evaluation.

For each task in a chosen range, run the same ReAct agent three times,
once per retrieval mode:
- vanilla     no skill retrieval
- similarity  O(N) cosine over skill descriptions
- capability  O(K) hash on capability signature plus fallback chain

Same task, same LLM, same prompt skeleton. Only the retrieval path changes.
We report per-family success rate, average steps, and retrieval path
distribution so we can attribute wins to the indexing mechanism vs the
plain presence of skills.

Usage:

    export OPENAI_API_KEY="sk-..."
    python -m memcryst.examples.alfworld_eval \\
        --skills-dir /tmp/memcryst-alfworld/skills \\
        --num-tasks 20

Cost estimate: ~$0.40 for 20 tasks * 3 modes, gpt-4o-mini.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional  # noqa: F401

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.agent import AlfworldEnv, ReactAgent, family_from_task_description
from memcryst.baselines import CondensedMemoryRetriever, CondensedMemoryStore
from memcryst.core import LLMClient
from memcryst.memory_bank import MemoryRetriever, MemoryStore
from memcryst.skill_bank import SkillRetriever, SkillStore


console = Console()


MODES = ["vanilla", "memory", "memory_condense", "similarity", "capability"]


def run_eval(
    split: str,
    llm: LLMClient,
    skill_retriever: Optional[SkillRetriever],
    memory_retriever: Optional[MemoryRetriever],
    condensed_retriever: Optional[CondensedMemoryRetriever],
    task_indices: list[int],
    modes: list[str],
    max_steps: int,
    top_k: int,
) -> dict:
    """Returns a nested dict: results[mode][task_idx] -> TrajectoryResult dict.

    Paired design: each mode runs on the same sequence of task_indices. We
    achieve this by instantiating a fresh env per mode, since ALFWorld 0.4
    reset() ignores task_idx and just auto-increments an internal counter.
    A fresh env restarts the counter at 0 so all modes see the same games.
    """
    results: dict[str, dict[int, dict]] = {m: {} for m in modes}
    for mode in modes:
        console.rule(f"mode={mode}")
        env = AlfworldEnv(split=split, max_steps=max_steps)
        # If task_indices[0] > 0, skip forward by consuming resets.
        skip = task_indices[0] if task_indices else 0
        for _ in range(skip):
            env.reset()
        for ti in task_indices:
            agent = ReactAgent(
                env=env,
                llm=llm,
                skill_retriever=skill_retriever if mode in ("similarity", "capability") else None,
                memory_retriever=memory_retriever if mode == "memory" else None,
                condensed_retriever=condensed_retriever if mode == "memory_condense" else None,
                mode=mode,
                top_k_skills=top_k,
                max_steps=max_steps,
            )
            console.print(f"[bold]task {ti:>3} mode={mode:<10}[/bold] ", end="")
            r = agent.run(task_idx=ti)
            true_family = family_from_task_description(r.task_description)
            console.print(
                f"family={true_family:<8} success={str(r.success):<5} "
                f"steps={r.steps:>2} llm={r.llm_calls:>2} retries={r.invalid_action_retries} "
                f"paths={dict(r.retrieval_paths) or '{}'} "
                f"desc={r.task_description[:50]}"
            )
            results[mode][ti] = {
                "task_idx": ti,
                "task_description": r.task_description,
                "verb_family": true_family,
                "mode": mode,
                "success": r.success,
                "steps": r.steps,
                "final_reward": r.final_reward,
                "llm_calls": r.llm_calls,
                "invalid_action_retries": r.invalid_action_retries,
                "used_skill_ids": r.used_skill_ids,
                "retrieval_paths": dict(r.retrieval_paths),
                "error": r.error,
            }
    return results


def render_summary(results: dict, modes: list[str]) -> None:
    # Collate per-family stats per mode.
    family_stats: dict[str, dict[str, dict]] = defaultdict(lambda: {m: {"n": 0, "succ": 0, "steps": 0, "llm": 0} for m in modes})
    overall: dict[str, dict] = {m: {"n": 0, "succ": 0, "steps": 0, "llm": 0} for m in modes}
    for mode in modes:
        for ti, r in results[mode].items():
            fam = r["verb_family"]
            family_stats[fam][mode]["n"] += 1
            family_stats[fam][mode]["succ"] += 1 if r["success"] else 0
            family_stats[fam][mode]["steps"] += r["steps"]
            family_stats[fam][mode]["llm"] += r["llm_calls"]
            overall[mode]["n"] += 1
            overall[mode]["succ"] += 1 if r["success"] else 0
            overall[mode]["steps"] += r["steps"]
            overall[mode]["llm"] += r["llm_calls"]

    # Main table.
    console.rule("success rate by family x mode")
    table = Table()
    table.add_column("family")
    for m in modes:
        table.add_column(m)
    for fam in sorted(family_stats.keys()):
        row = [fam]
        for m in modes:
            s = family_stats[fam][m]
            if s["n"] == 0:
                row.append("-")
            else:
                pct = 100 * s["succ"] / s["n"]
                row.append(f"{s['succ']}/{s['n']} ({pct:.0f}%)")
        table.add_row(*row)
    overall_row = ["ALL"]
    for m in modes:
        s = overall[m]
        if s["n"] == 0:
            overall_row.append("-")
        else:
            pct = 100 * s["succ"] / s["n"]
            overall_row.append(f"{s['succ']}/{s['n']} ({pct:.0f}%)")
    table.add_row(*overall_row, style="bold")
    console.print(table)

    # Step + cost table.
    console.rule("average steps and llm calls")
    table2 = Table()
    table2.add_column("mode")
    table2.add_column("avg_steps")
    table2.add_column("avg_llm_calls")
    for m in modes:
        s = overall[m]
        if s["n"] == 0:
            continue
        table2.add_row(m, f"{s['steps']/s['n']:.1f}", f"{s['llm']/s['n']:.1f}")
    console.print(table2)

    # Retrieval path distribution for capability mode.
    if "capability" in modes:
        console.rule("capability retrieval path distribution")
        path_counter: Counter = Counter()
        for r in results["capability"].values():
            for p, c in r["retrieval_paths"].items():
                path_counter[p] += c
        table3 = Table()
        table3.add_column("path")
        table3.add_column("count")
        for p, c in path_counter.most_common():
            table3.add_row(p, str(c))
        console.print(table3)

    # Where capability wins vs similarity, per task.
    console.rule("per-task delta: capability vs similarity (skills) vs vanilla")
    delta_table = Table()
    delta_table.add_column("task")
    delta_table.add_column("family")
    delta_table.add_column("task_desc", overflow="fold")
    for m in modes:
        delta_table.add_column(m)
    task_ids = sorted(set().union(*[set(results[m].keys()) for m in modes]))
    for ti in task_ids:
        row = [str(ti)]
        any_r = next((results[m][ti] for m in modes if ti in results[m]), None)
        if any_r is None:
            continue
        row.append(any_r["verb_family"])
        row.append(any_r["task_description"][:60])
        for m in modes:
            r = results[m].get(ti)
            if r is None:
                row.append("-")
            else:
                tag = "✓" if r["success"] else "✗"
                row.append(f"{tag} {r['steps']}s")
        delta_table.add_row(*row)
    console.print(delta_table)


def main() -> int:
    parser = argparse.ArgumentParser(description="ALFWorld N-mode eval")
    parser.add_argument("--split", default="eval_out_of_distribution")
    parser.add_argument("--skills-dir", default="")
    parser.add_argument("--memory-dir", default="", help="memory store dir, required for mode=memory")
    parser.add_argument("--condense-dir", default="", help="condensed memory dir, required for mode=memory_condense")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--modes", default="vanilla,memory,similarity,capability")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    if not modes:
        console.print("[red]no valid modes specified[/red]")
        return 2

    llm = LLMClient(model=args.model, temperature=0.0)

    skill_retriever: Optional[SkillRetriever] = None
    memory_retriever: Optional[MemoryRetriever] = None
    condensed_retriever: Optional[CondensedMemoryRetriever] = None
    skill_summary = "(none)"
    memory_summary = "(none)"
    condense_summary = "(none)"

    needs_skill = any(m in ("similarity", "capability") for m in modes)
    needs_memory = "memory" in modes
    needs_condense = "memory_condense" in modes

    if needs_skill:
        if not args.skills_dir:
            console.print(f"[red]--skills-dir required for skill modes[/red]")
            return 2
        store = SkillStore(store_dir=Path(args.skills_dir))
        if len(store) == 0:
            console.print(f"[red]skill store empty at {args.skills_dir}[/red]")
            return 2
        skill_retriever = SkillRetriever(store, signature_llm=llm)
        skill_summary = f"{len(store)} skills from {args.skills_dir}"

    if needs_memory:
        if not args.memory_dir:
            console.print(f"[red]--memory-dir required for mode=memory[/red]")
            return 2
        memstore = MemoryStore(store_dir=Path(args.memory_dir))
        if len(memstore) == 0:
            console.print(f"[red]memory store empty at {args.memory_dir}[/red]")
            return 2
        memory_retriever = MemoryRetriever(memstore)
        memory_summary = f"{len(memstore)} entries from {args.memory_dir}"

    if needs_condense:
        if not args.condense_dir:
            console.print(f"[red]--condense-dir required for mode=memory_condense[/red]")
            return 2
        condstore = CondensedMemoryStore(store_dir=Path(args.condense_dir))
        if len(condstore) == 0:
            console.print(f"[red]condensed store empty at {args.condense_dir}[/red]")
            return 2
        condensed_retriever = CondensedMemoryRetriever(condstore)
        condense_summary = f"{len(condstore)} clusters from {args.condense_dir}"

    console.print(
        Panel.fit(
            f"skills: {skill_summary}\n"
            f"memory: {memory_summary}\n"
            f"condense: {condense_summary}\n"
            f"modes: {modes}\n"
            f"num_tasks: {args.num_tasks}\n"
            f"max_steps: {args.max_steps}",
            title="eval config",
            style="cyan",
        )
    )

    task_indices = list(range(args.task_start, args.task_start + args.num_tasks))

    results = run_eval(
        split=args.split,
        llm=llm,
        skill_retriever=skill_retriever,
        memory_retriever=memory_retriever,
        condensed_retriever=condensed_retriever,
        task_indices=task_indices,
        modes=modes,
        max_steps=args.max_steps,
        top_k=args.top_k,
    )

    render_summary(results, modes)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        console.print(f"\nresults JSON dumped to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
