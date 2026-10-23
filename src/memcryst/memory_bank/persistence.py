"""Memory bank persistence. JSONL storage forked from live-evo.

The store keeps the full memory bank in memory for fast retrieval. JSONL on
disk acts as the durable record. We write the file on every mutating
operation so a crash does not lose data.

Default location is the directory ~/.memcryst/. Pass a custom directory at
construction time to keep multiple banks isolated, useful for testing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterator, Optional

from .schema import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    WEIGHT_DECREASE_RATE,
    WEIGHT_INCREASE_RATE,
    EntryStatus,
    MemoryEntry,
)

logger = logging.getLogger(__name__)


def default_store_dir() -> Path:
    return Path.home() / ".memcryst"


class MemoryStore:
    """JSONL backed memory bank with in memory cache."""

    def __init__(self, store_dir: Optional[Path] = None, autosave: bool = True) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else default_store_dir()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "memory_db.jsonl"
        self.autosave = autosave
        self._entries: dict[str, MemoryEntry] = {}
        self.load()

    # ----- load and save -----

    def load(self) -> None:
        self._entries.clear()
        if not self.db_path.exists():
            return
        with self.db_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s", self.db_path)
                    continue
                entry = MemoryEntry.from_dict(obj)
                self._entries[entry.id] = entry

    def save(self) -> None:
        tmp = self.db_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict()) + "\n")
        tmp.replace(self.db_path)

    # ----- CRUD -----

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        if not entry.env:
            from memcryst.envscope import current_env
            entry.env = current_env(content=True)
        self._entries[entry.id] = entry
        if self.autosave:
            self.save()
        return entry

    def add_many(self, entries: list[MemoryEntry]) -> None:
        from memcryst.envscope import current_env
        for entry in entries:
            if not entry.env:
                entry.env = current_env(content=True)
            self._entries[entry.id] = entry
        if self.autosave:
            self.save()

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def active(self) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.status == EntryStatus.ACTIVE]

    def by_status(self, status: EntryStatus) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.status == status]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            if self.autosave:
                self.save()
            return True
        return False

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[MemoryEntry]:
        return iter(self._entries.values())

    # ----- weight updates, mirrored from live-evo experience_manager -----

    def increase_weight(self, entry_id: str, factor: float = 1.0) -> Optional[MemoryEntry]:
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.weight += WEIGHT_INCREASE_RATE * factor
        entry.clamp_weight()
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def decrease_weight(self, entry_id: str, factor: float = 1.0) -> Optional[MemoryEntry]:
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.weight -= WEIGHT_DECREASE_RATE * factor
        entry.clamp_weight()
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    # ----- MEMCRYST specific state changes -----

    def flag_unreliable(self, entry_id: str, severity: float = 1.0) -> Optional[MemoryEntry]:
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.status = EntryStatus.FLAGGED
        entry.weight = max(MIN_WEIGHT, entry.weight * (1.0 - 0.5 * severity))
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def archive(self, entry_id: str, crystallized_into: str) -> Optional[MemoryEntry]:
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.status = EntryStatus.ARCHIVED
        entry.crystallized_into = crystallized_into
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def mark_verified(self, entry_id: str) -> Optional[MemoryEntry]:
        """N3 protect-verified: bump verified_uses when a retrieval led to success."""
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.verified_uses += 1
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def boost_weight(self, entry_id: str, amount: float = 0.05) -> Optional[MemoryEntry]:
        """N2 mem-boost: increase memory weight after a successful agent run."""
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.weight += amount
        entry.clamp_weight()
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def update_utility(self, entry_id: str, outcome: float, alpha: float = 0.1) -> Optional[MemoryEntry]:
        """N4 dual-utility EMA: utility := (1-alpha)*utility + alpha*outcome.
        outcome is 1.0 on success, 0.0 on failure."""
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.utility = (1.0 - alpha) * entry.utility + alpha * outcome
        entry.updated_at = time.time()
        if self.autosave:
            self.save()
        return entry

    def set_embedding(self, entry_id: str, embedding: list[float]) -> None:
        entry = self.get(entry_id)
        if entry is None:
            return
        entry.embedding = list(embedding)
        if self.autosave:
            self.save()

    def stats(self) -> dict:
        active = sum(1 for e in self if e.status == EntryStatus.ACTIVE)
        flagged = sum(1 for e in self if e.status == EntryStatus.FLAGGED)
        archived = sum(1 for e in self if e.status == EntryStatus.ARCHIVED)
        weights = [e.weight for e in self]
        return {
            "total": len(self),
            "active": active,
            "flagged": flagged,
            "archived": archived,
            "mean_weight": sum(weights) / len(weights) if weights else 0.0,
            "db_path": str(self.db_path),
        }
