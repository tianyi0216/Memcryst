"""Unit tests for skill_bank."""

from __future__ import annotations

from pathlib import Path

import pytest

from memcryst.skill_bank import Capability, Skill, SkillStore, TaskSignatureExtractor


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "skills"


def test_skill_slug_kebab() -> None:
    s = Skill(name="Clean and Place", description="d", body="b")
    assert s.slug == "clean-and-place"


def test_skill_default_category() -> None:
    s = Skill(name="x", description="d", body="b", category="not-a-real-cat")
    assert s.category == "general"


def test_round_trip_skill_md(store_dir: Path) -> None:
    store = SkillStore(store_dir=store_dir)
    s = Skill(
        name="clean-then-place",
        description="When to clean and place",
        body="## When to use\n- clean and place tasks.",
        category="general",
        ancestry=["a1", "a2", "a3"],
        crystallized_from_cluster=2,
        crystallization_round=1,
        utility=0.6,
    )
    store.add(s)

    store2 = SkillStore(store_dir=store_dir)
    loaded = store2.get(s.id)
    assert loaded is not None
    assert loaded.name == s.name
    assert loaded.description == s.description
    assert loaded.ancestry == ["a1", "a2", "a3"]
    assert loaded.crystallized_from_cluster == 2
    assert "When to use" in loaded.body


def test_slug_collision_does_not_overwrite(store_dir: Path) -> None:
    """Two skills whose names slugify to the same string must both round trip."""
    store = SkillStore(store_dir=store_dir)
    s1 = Skill(
        name="Clean Object",
        description="first",
        body="body 1",
        category="general",
        created_at=100.0,
    )
    s2 = Skill(
        name="clean-object",
        description="second",
        body="body 2",
        category="general",
        created_at=101.0,
    )
    assert s1.slug == s2.slug
    assert s1.id != s2.id
    store.add(s1)
    store.add(s2)

    store2 = SkillStore(store_dir=store_dir)
    assert len(store2) == 2
    loaded1 = store2.get(s1.id)
    loaded2 = store2.get(s2.id)
    assert loaded1 is not None and loaded1.description == "first"
    assert loaded2 is not None and loaded2.description == "second"


def test_record_use_updates_stats(store_dir: Path) -> None:
    store = SkillStore(store_dir=store_dir)
    s = Skill(name="abc", description="d", body="b", utility=0.5)
    store.add(s)
    store.record_use(s.id, success=True)
    store.record_use(s.id, success=False)
    s2 = store.get(s.id)
    assert s2.use_count == 2
    assert s2.failure_count == 1
    assert abs(s2.failure_rate() - 0.5) < 1e-6


def test_capability_canonicalization_via_synonyms() -> None:
    """LLM emits 'wash' / 'microwave' / 'placed' which fold to canonical."""
    raw = {
        "verb_family": "wash",
        "target_type": "object",
        "affordance": "kitchen_sink",
        "output": "placed",
        "preconditions": ["object_at_source", "garbage_tag"],
        "task_pattern": "x",
    }
    cap = Capability.from_dict(raw)
    assert cap.verb_family == "clean"
    assert cap.affordance == "sink"
    assert cap.output == "object_placed"
    assert cap.preconditions == ["object_at_source"]  # bogus tag dropped
    assert cap.is_complete()


def test_capability_incomplete_when_outside_closed_set() -> None:
    raw = {
        "verb_family": "make_coffee",  # not in closed set
        "target_type": "object",
        "affordance": "sink",
        "output": "object_placed",
    }
    cap = Capability.from_dict(raw)
    assert cap.verb_family is None
    assert not cap.is_complete()


def test_capability_index_register_and_lookup(store_dir: Path) -> None:
    store = SkillStore(store_dir=store_dir)
    cap1 = Capability(
        verb_family="clean",
        target_type="object",
        affordance="sink",
        output="object_placed",
    )
    cap2 = Capability(
        verb_family="heat",
        target_type="object",
        affordance="microwave",
        output="object_placed",
    )
    cap3 = Capability(
        verb_family="examine",
        target_type="object",
        affordance="lamp",
        output="object_examined",
    )
    s1 = Skill(name="clean-and-place", description="d", body="b", capability=cap1)
    s2 = Skill(name="heat-and-place", description="d", body="b", capability=cap2)
    s3 = Skill(name="examine-with-light", description="d", body="b", capability=cap3)
    store.add(s1)
    store.add(s2)
    store.add(s3)

    assert len(store.capability_index) == 3

    # Exact lookup
    ids = store.capability_index.lookup(cap1.signature())
    assert s1.id in ids
    assert s2.id not in ids
    assert s3.id not in ids

    # Relaxed lookup on affordance
    sig_no_afford = ("clean", "object", "stove", "object_placed")  # affordance mismatch
    ids = store.capability_index.lookup(sig_no_afford)
    assert ids == []
    ids = store.capability_index.lookup_relaxed(sig_no_afford, relax_axes={"affordance"})
    assert s1.id in ids  # found via affordance relaxation


def test_task_signature_extractor_rule_path() -> None:
    extractor = TaskSignatureExtractor(llm=None)
    cap = extractor.extract("Clean the apple and place at the dining table")
    assert cap.verb_family == "clean"
    assert cap.affordance == "none"  # no affordance keyword in NL
    assert cap.output == "object_placed"

    cap = extractor.extract("Examine the book under the desk lamp")
    assert cap.verb_family == "examine"
    assert cap.affordance == "desklamp"
    assert cap.output == "object_examined"

    cap = extractor.extract("Heat the mug in the microwave")
    assert cap.verb_family == "heat"
    assert cap.affordance == "microwave"


def test_capability_index_persists_across_reload(store_dir: Path) -> None:
    store = SkillStore(store_dir=store_dir)
    cap = Capability(
        verb_family="clean",
        target_type="object",
        affordance="sink",
        output="object_placed",
        preconditions=["object_at_source"],
        task_pattern="clean <obj> and place at <dest>",
    )
    s = Skill(name="clean-place", description="d", body="b", capability=cap)
    store.add(s)

    store2 = SkillStore(store_dir=store_dir)
    reloaded = store2.get(s.id)
    assert reloaded is not None
    assert reloaded.capability.is_complete()
    assert reloaded.capability.verb_family == "clean"
    assert reloaded.capability.affordance == "sink"
    ids = store2.capability_index.lookup(cap.signature())
    assert s.id in ids


def test_record_axis_failure_tracks_dominant_axis(store_dir: Path) -> None:
    store = SkillStore(store_dir=store_dir)
    s = Skill(name="x", description="d", body="b")
    store.add(s)
    store.record_axis_failure(s.id, "affordance")
    store.record_axis_failure(s.id, "affordance")
    store.record_axis_failure(s.id, "verb_family")
    s2 = store.get(s.id)
    assert s2.axis_failure_counts["affordance"] == 2
    assert s2.axis_failure_counts["verb_family"] == 1
