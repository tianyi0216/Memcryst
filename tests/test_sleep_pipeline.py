"""Integration test for the sleep pipeline using a mock LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memcryst.core import LLMClient
from memcryst.memory_bank import EntryStatus, MemoryRetriever, MemoryStore
from memcryst.skill_bank import SkillStore
from memcryst.sleep import (
    FeedbackLoop,
    PatternMiner,
    SkillAuditor,
    SkillSynthesizer,
    SleepPipeline,
)
from memcryst.examples.synthetic_pilot import seed_entries


_FAKE_SKILL_MD = """\
---
name: clean-then-place-procedure
description: Use for clean and place tasks involving household objects.
category: general
capability:
  verb_family: clean
  target_type: object
  affordance: sink
  output: object_placed
  preconditions:
    - object_at_source
    - destination_reachable
  task_pattern: "clean <target_object> and place at <destination>"
parameters:
  target_object:
    type: object
    required: true
  source_location:
    type: receptacle
    required: true
  destination:
    type: receptacle
    required: true
---

## When to use
- Clean and place tasks for household objects.

## Preconditions
- Sink is reachable.

## Procedure
1. Locate <target_object>.
2. Carry <target_object> to the sink.
3. Use the sink action on <target_object>.
4. Carry <target_object> to <destination>.
5. Place <target_object> at <destination>.

## Anti patterns
- Do not skip the sink step.
"""


_FAKE_AUDIT = {
    "checks": [{"id": i, "verdict": "PASS", "reason": "ok"} for i in range(1, 14)],
    "pass_count": 13,
    "decision": "ACCEPT",
    "overall_reason": "looks fine",
}


class MockLLMClient(LLMClient):
    def __init__(self, **kwargs: Any) -> None:
        self.model = "mock"
        self.temperature = 0.0
        self.max_tokens = 2000
        self.max_retries = 1
        self._client = None

    def chat(self, messages, model=None, temperature=None, max_tokens=None, **kwargs):  # type: ignore[override]
        joined = "\n".join(m["content"] if isinstance(m, dict) else m.content for m in messages)
        if "JSON object" in joined or '"checks"' in joined:
            return json.dumps(_FAKE_AUDIT)
        return _FAKE_SKILL_MD


@pytest.fixture
def stores(tmp_path: Path):
    memory_store = MemoryStore(store_dir=tmp_path / "memory")
    skill_store = SkillStore(store_dir=tmp_path / "skills")
    memory_store.add_many(seed_entries())
    return memory_store, skill_store


def test_pipeline_produces_skill_and_archives(stores) -> None:
    memory_store, skill_store = stores
    retriever = MemoryRetriever(memory_store)
    mock_llm = MockLLMClient()
    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=retriever,
        skill_store=skill_store,
        miner=PatternMiner(retriever, min_cluster_size=3, min_mean_reward=0.4),
        synthesizer=SkillSynthesizer(mock_llm),
        auditor=SkillAuditor(mock_llm, min_pass=10),
        feedback=FeedbackLoop(skill_store, memory_store),
        max_candidates_per_run=5,
        archive_on_accept=True,
    )

    report = pipeline.run()
    assert len(report.accepted_skills) >= 1
    archived = memory_store.by_status(EntryStatus.ARCHIVED)
    assert len(archived) >= 3
    assert len(skill_store) >= 1


def test_feedback_does_not_reflag_on_repeat(stores) -> None:
    """M3 must skip entries that are already FLAGGED to avoid weight collapse."""
    memory_store, skill_store = stores
    retriever = MemoryRetriever(memory_store)
    mock_llm = MockLLMClient()
    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=retriever,
        skill_store=skill_store,
        miner=PatternMiner(retriever, min_cluster_size=3, min_mean_reward=0.4),
        synthesizer=SkillSynthesizer(mock_llm),
        auditor=SkillAuditor(mock_llm, min_pass=10),
        feedback=FeedbackLoop(skill_store, memory_store, failure_threshold=0.3, min_uses=3),
        max_candidates_per_run=5,
        archive_on_accept=False,
    )
    pipeline.run()
    assert len(skill_store) >= 1
    skill = skill_store.all()[0]
    for _ in range(5):
        skill_store.record_use(skill.id, success=False)
    skill_store.record_use(skill.id, success=True)

    # First flag pass should flag at least one entry.
    e1 = pipeline.feedback.run()
    assert any(e.flagged_entry_ids for e in e1)

    # Capture weights right after first flagging.
    from memcryst.memory_bank import EntryStatus
    flagged_after_first = [
        (e.id, e.weight) for e in memory_store if e.status == EntryStatus.FLAGGED
    ]

    # Run M3 again with no additional use signal. Already FLAGGED entries
    # must not be flagged again, so weights should be unchanged.
    pipeline.feedback.run()
    pipeline.feedback.run()
    weights_after = {e.id: e.weight for e in memory_store}
    for entry_id, w_before in flagged_after_first:
        assert weights_after[entry_id] == w_before, (
            f"entry {entry_id} weight changed from {w_before} to {weights_after[entry_id]}"
            " after a no-op M3 cycle"
        )


def test_feedback_flags_after_failures(stores) -> None:
    memory_store, skill_store = stores
    retriever = MemoryRetriever(memory_store)
    mock_llm = MockLLMClient()
    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=retriever,
        skill_store=skill_store,
        miner=PatternMiner(retriever, min_cluster_size=3, min_mean_reward=0.4),
        synthesizer=SkillSynthesizer(mock_llm),
        auditor=SkillAuditor(mock_llm, min_pass=10),
        feedback=FeedbackLoop(skill_store, memory_store, failure_threshold=0.3, min_uses=3),
        max_candidates_per_run=5,
        archive_on_accept=False,  # keep entries ACTIVE so M3 can flag them
    )
    pipeline.run()
    assert len(skill_store) >= 1

    skill = skill_store.all()[0]
    for _ in range(5):
        skill_store.record_use(skill.id, success=False)
    skill_store.record_use(skill.id, success=True)

    events = pipeline.feedback.run()
    assert len(events) >= 1
    flagged = memory_store.by_status(EntryStatus.FLAGGED)
    assert len(flagged) >= 1
