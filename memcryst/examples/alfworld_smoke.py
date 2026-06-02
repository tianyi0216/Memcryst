"""Smoke test for the ALFWorld adapter.

Loads the eval split, picks the first task, prints the initial observation,
and either runs a hardcoded baseline sequence OR lets you step manually.

Usage:

    export ALFWORLD_DATA="$HOME/.cache/alfworld"
    python -m memcryst.examples.alfworld_smoke

Optional flags:

    --task-idx N             pick task index N from the split
    --split SPLIT            "train" / "eval_in_distribution" / "eval_out_of_distribution"
    --interactive            prompt for actions instead of running canned sequence
    --list                   just list available task families and exit
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.agent import AlfworldEnv


console = Console()


def cmd_list(env: AlfworldEnv) -> None:
    tasks = env.list_tasks()
    table = Table(title=f"ALFWorld split summary: {len(tasks)} tasks")
    table.add_column("verb_family")
    table.add_column("count")
    counts = Counter(t.verb_family for t in tasks)
    for verb, count in counts.most_common():
        table.add_row(verb, str(count))
    console.print(table)
    console.print(f"\nFirst 5 game files:")
    for t in tasks[:5]:
        console.print(f"  [{t.task_idx}] {t.verb_family}  {t.game_file}")


_CANNED_LOOK = ["look", "inventory"]


def cmd_run(env: AlfworldEnv, task_idx: int, interactive: bool) -> None:
    reset = env.reset(task_idx=task_idx)
    console.rule(f"task {task_idx}")
    console.print(Panel(reset.obs, title="initial observation", style="cyan"))
    console.print(f"[bold]Parsed task[/bold]: {reset.task_description}\n")
    if reset.valid_actions:
        console.print(f"[dim]first 8 valid actions[/dim]: {reset.valid_actions[:8]}\n")

    if not interactive:
        # Canned smoke: just look + inventory to verify the env responds.
        for action in _CANNED_LOOK:
            console.print(f"[yellow]> {action}[/yellow]")
            step = env.step(action)
            console.print(step.obs)
            console.print(
                f"[dim]reward={step.reward} done={step.done} success={step.success}[/dim]\n"
            )
            if step.done:
                break
        return

    # Interactive: read actions until done or user hits ctrl-D.
    console.print("[bold]Interactive mode. Type actions, ctrl-D to quit.[/bold]")
    while True:
        try:
            action = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not action:
            continue
        if action == "actions":
            step_actions = env.step("look")  # noop refresh
            console.print(step_actions.valid_actions[:20])
            continue
        step = env.step(action)
        console.print(step.obs)
        console.print(
            f"[dim]reward={step.reward} done={step.done} success={step.success}[/dim]"
        )
        if step.done:
            console.print(
                Panel(
                    f"done. success={step.success}",
                    style="green" if step.success else "red",
                )
            )
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="ALFWorld smoke test")
    parser.add_argument("--split", default="eval_out_of_distribution")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    env = AlfworldEnv(split=args.split, max_steps=args.max_steps)

    if args.list:
        cmd_list(env)
        return 0

    cmd_run(env, task_idx=args.task_idx, interactive=args.interactive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
