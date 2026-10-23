"""N9 per-(skill, memory) pair tracking.

Stored as a sidecar JSON file in the memory bank directory:
   output_dir/memory/pair_stats.json
Format:
   { "<skill_id>|<memory_id>": [uses, successes], ... }

The MemoryRetriever can use this to add a co-occurrence bonus during
retrieval when the agent is also retrieving skills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _key(skill_id: str, memory_id: str) -> str:
    return f"{skill_id}|{memory_id}"


def load_pair_stats(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for k, v in raw.items():
        if "|" not in k:
            continue
        sid, mid = k.split("|", 1)
        out[(sid, mid)] = (int(v[0]), int(v[1]))
    return out


def save_pair_stats(path: Path, stats: dict[tuple[str, str], tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {_key(sid, mid): list(v) for (sid, mid), v in stats.items()}
    path.write_text(json.dumps(raw, indent=2))


def record_pair_outcome(
    stats: dict[tuple[str, str], tuple[int, int]],
    skill_ids: list[str],
    memory_ids: list[str],
    success: bool,
) -> None:
    """Mutate `stats` to record one trajectory's outcome on every pair."""
    succ_inc = 1 if success else 0
    for sid in skill_ids:
        for mid in memory_ids:
            uses, succs = stats.get((sid, mid), (0, 0))
            stats[(sid, mid)] = (uses + 1, succs + succ_inc)
