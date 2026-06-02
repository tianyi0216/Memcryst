"""Unit tests for memory_bank."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memcryst.memory_bank import EntryStatus, MemoryEntry, MemoryStore


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "memory"


def test_memory_entry_id_stable(store_dir: Path) -> None:
    e1 = MemoryEntry(question="abc", failure_reason="x", improvement="y", created_at=100.0)
    assert e1.id != ""
    # Same content + same timestamp should give same id.
    e2 = MemoryEntry(question="abc", failure_reason="x", improvement="y", created_at=100.0)
    assert e1.id == e2.id


def test_memory_entry_round_trip(store_dir: Path) -> None:
    e = MemoryEntry(
        question="q1",
        failure_reason="fr",
        improvement="imp",
        category="general",
        weight=1.2,
        crystallized_into="abc",
        status=EntryStatus.FLAGGED,
    )
    d = e.to_dict()
    e2 = MemoryEntry.from_dict(d)
    assert e2.question == e.question
    assert e2.crystallized_into == "abc"
    assert e2.status == EntryStatus.FLAGGED
    assert e2.weight == 1.2


def test_store_add_save_load(store_dir: Path) -> None:
    store = MemoryStore(store_dir=store_dir)
    e = MemoryEntry(question="q1", failure_reason="x", improvement="y")
    store.add(e)
    assert len(store) == 1
    # Reload from disk.
    store2 = MemoryStore(store_dir=store_dir)
    assert len(store2) == 1
    assert store2.get(e.id) is not None


def test_weight_clamp(store_dir: Path) -> None:
    store = MemoryStore(store_dir=store_dir)
    e = MemoryEntry(question="q", failure_reason="x", improvement="y", weight=1.9)
    store.add(e)
    store.increase_weight(e.id)  # 1.9 + 0.3 = 2.2 -> clamp to 2.0
    assert store.get(e.id).weight == 2.0


def test_flag_unreliable_sets_status(store_dir: Path) -> None:
    store = MemoryStore(store_dir=store_dir)
    e = MemoryEntry(question="q", failure_reason="x", improvement="y", weight=1.0)
    store.add(e)
    store.flag_unreliable(e.id, severity=0.5)
    flagged = store.get(e.id)
    assert flagged.status == EntryStatus.FLAGGED
    assert flagged.weight < 1.0


def test_archive_sets_crystallized_into(store_dir: Path) -> None:
    store = MemoryStore(store_dir=store_dir)
    e = MemoryEntry(question="q", failure_reason="x", improvement="y")
    store.add(e)
    store.archive(e.id, crystallized_into="skill-abc")
    archived = store.get(e.id)
    assert archived.status == EntryStatus.ARCHIVED
    assert archived.crystallized_into == "skill-abc"


def test_stats(store_dir: Path) -> None:
    store = MemoryStore(store_dir=store_dir)
    for i in range(5):
        store.add(
            MemoryEntry(
                question=f"q{i}",
                failure_reason="",
                improvement="",
                weight=1.0 + 0.1 * i,
                created_at=100.0 + i,
            )
        )
    stats = store.stats()
    assert stats["total"] == 5
    assert stats["active"] == 5
    assert abs(stats["mean_weight"] - 1.2) < 0.05
