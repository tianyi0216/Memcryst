"""Lightweight bidirectional feedback hooks (success distillation, refinement
notes on failing skills, utility-driven memory refinement) implemented as
heuristics so the mechanism can be ablated without extra LLM calls.
"""

from __future__ import annotations

import time
from typing import Iterable

from ..memory_bank import MemoryEntry, MemoryStore
from ..skill_bank import Skill, SkillStore


def n6_distill_success_entry(
    memory_store: MemoryStore,
    task_description: str,
    skill: Skill,
    used_memory_ids: list[str],
    step_count: int,
) -> MemoryEntry | None:
    """N6: after a successful agent run, write a compiled-lesson entry into
    memory. Heuristic: template-only, no LLM. The new entry summarizes
    what worked at the procedural level so future M1 cycles cluster it
    with raw episodes from the same family.
    """
    question = task_description.strip()
    parts = [
        f"compiled lesson from skill `{skill.name}`",
        f"task completed in {step_count} steps using procedural skill plus {len(used_memory_ids)} memory entries.",
    ]
    if skill.capability.is_complete():
        parts.append(
            f"key axes verb={skill.capability.verb_family}, "
            f"affordance={skill.capability.affordance}, "
            f"output={skill.capability.output}"
        )
    improvement = " | ".join(parts)
    entry = MemoryEntry(
        question=question,
        failure_reason="",
        improvement=improvement,
        category="distillation",
        weight=1.5,
    )
    memory_store.add(entry)
    return entry


def n8_append_failure_context(
    memory_store: MemoryStore,
    skill: Skill,
    ancestry_ids: Iterable[str],
    failure_rate: float,
) -> int:
    """N8: heuristic memory refinement. Append a short note to the
    failure_reason field of failing skill's ancestry entries so future
    retrievals carry the context. No weight change.
    """
    refined = 0
    note = (
        f"[refined: ancestor of skill `{skill.name}` "
        f"(fail rate {failure_rate:.2f}); review before re-applying]"
    )
    for eid in ancestry_ids:
        e = memory_store.get(eid)
        if e is None:
            continue
        if note in e.failure_reason:
            continue
        sep = "\n" if e.failure_reason else ""
        e.failure_reason = f"{e.failure_reason}{sep}{note}"
        e.updated_at = time.time()
        refined += 1
    if refined > 0:
        memory_store.save()
    return refined


def n13_joint_refine_low_utility(
    memory_store: MemoryStore,
    min_utility: float,
    min_uses: int,
) -> int:
    """N13: at sleep time, append a self-doubt annotation to memory entries
    that have low utility AND have been retrieved multiple times. They're
    being used and not helping, so flag the text. No weight change.
    """
    refined = 0
    note = "[refined: low utility, agent has cited this without improving]"
    for e in memory_store:
        # We don't track per-memory use_count yet; approximate with
        # verified_uses being 0 despite the entry being long-lived. Skip
        # if low-utility heuristic is not met.
        if e.utility >= min_utility:
            continue
        # Only refine entries that have been seen multiple cycles
        if e.verified_uses < min_uses:
            continue
        if note in e.failure_reason:
            continue
        sep = "\n" if e.failure_reason else ""
        e.failure_reason = f"{e.failure_reason}{sep}{note}"
        e.updated_at = time.time()
        refined += 1
    if refined > 0:
        memory_store.save()
    return refined
