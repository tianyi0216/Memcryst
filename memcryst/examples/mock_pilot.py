"""End to end pipeline test with a mock LLM. No API key required.

We patch LLMClient.chat to return canned responses. The synthesizer gets a
hand crafted SKILL.md text. The auditor gets a hand crafted JSON verdict.
This verifies that M2 synthesizer, M2 audit, archive, and M3 feedback all
work together. Run before paying for any API call.

Usage

    python -m memcryst.examples.mock_pilot
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

console = Console()


_FAKE_SKILL_MD = """\
---
name: clean-then-place-procedure
description: Use when a task requires bringing a household object to the sink, cleaning it, then placing it at a target location.
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
- Tasks that contain the verb pair clean and place.
- Tasks where the agent picks up a household object and the final state involves the object at a different location.

## Preconditions
- The agent can navigate to the sink.
- The object can be carried by the agent.

## Procedure
1. go to <source_location>
2. take <target_object> from <source_location>
3. go to sink
4. clean <target_object> with sink
5. go to <destination>
6. put <target_object> in/on <destination>

## Anti patterns
- Do not attempt to clean using a cloth or other surrogate. Always use the sink.
- Do not put the object down before reaching the sink unless the navigation is blocked.
"""


_FAKE_AUDIT_JSON = {
    "checks": [
        {"id": 1, "verdict": "PASS", "reason": "Uses placeholders for object and destination."},
        {"id": 2, "verdict": "PASS", "reason": "Reads as generic, not a copy of one episode."},
        {"id": 3, "verdict": "PASS", "reason": "Includes locate step before pickup."},
        {"id": 4, "verdict": "PASS", "reason": "Placeholders present for variable values."},
        {"id": 5, "verdict": "PASS", "reason": "First step locates target which is the central action."},
        {"id": 6, "verdict": "PASS", "reason": "No silent bypass steps."},
        {"id": 7, "verdict": "PASS", "reason": "Procedure steps are observable."},
        {"id": 8, "verdict": "PASS", "reason": "Shape stays stable across invocations."},
        {"id": 9, "verdict": "PASS", "reason": "Covers majority of cluster episodes."},
        {"id": 10, "verdict": "PASS", "reason": "Covers more than 60 percent of episodes."},
        {"id": 11, "verdict": "PASS", "reason": "Does not claim to handle out of scope tasks."},
        {"id": 12, "verdict": "PASS", "reason": "Single model family in cluster, default PASS."},
        {"id": 13, "verdict": "PASS", "reason": "Capability tags are complete and consistent with procedure."},
    ],
    "pass_count": 13,
    "decision": "ACCEPT",
    "overall_reason": "Skill is grounded, abstract, and broadly covers the cluster.",
}


class MockLLMClient(LLMClient):
    """Subclass that returns canned outputs instead of hitting the API."""

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        # Avoid constructing the real OpenAI client. We do not call super.
        self.model = kwargs.get("model", "mock")
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        self.max_retries = 1
        self._client = None

    def chat(self, messages, model=None, temperature=None, max_tokens=None, **kwargs):  # type: ignore[override]
        joined = "\n".join(m["content"] if isinstance(m, dict) else m.content for m in messages)
        if "audit" in joined.lower() or "JSON object" in joined:
            return json.dumps(_FAKE_AUDIT_JSON)
        # Otherwise assume synthesis.
        return _FAKE_SKILL_MD


def main() -> int:
    console.rule("MEMCRYST mock pilot, no API key needed")
    tmp = Path(tempfile.mkdtemp(prefix="memcryst-mock-"))
    console.print(f"using temp dir {tmp}\n")

    memory_store = MemoryStore(store_dir=tmp / "memory")
    skill_store = SkillStore(store_dir=tmp / "skills")
    memory_store.add_many(seed_entries())
    console.print(f"seeded {len(memory_store)} memory entries")

    retriever = MemoryRetriever(memory_store)
    mock_llm = MockLLMClient(model="mock-gpt", temperature=0.0)

    miner = PatternMiner(retriever, min_cluster_size=3, min_mean_reward=0.4)
    synthesizer = SkillSynthesizer(mock_llm)
    auditor = SkillAuditor(mock_llm, min_pass=10)
    feedback = FeedbackLoop(skill_store, memory_store)

    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=retriever,
        skill_store=skill_store,
        miner=miner,
        synthesizer=synthesizer,
        auditor=auditor,
        feedback=feedback,
        max_candidates_per_run=5,
        archive_on_accept=True,
    )

    report = pipeline.run()

    console.rule("sleep report")
    console.print(f"candidates considered: {len(report.candidates)}")
    console.print(f"accepted skills:       {len(report.accepted_skills)}")
    console.print(f"flag events:           {len(report.flag_events)}\n")

    for outcome in report.candidates:
        console.rule(
            f"cluster {outcome.cluster.cluster_id} size {outcome.cluster.size}",
            characters="-",
        )
        if outcome.synthesis.skill is not None:
            console.print(
                Panel.fit(
                    f"name: {outcome.synthesis.skill.name}\n"
                    f"description: {outcome.synthesis.skill.description}",
                    title="synthesized skill",
                    style="green" if outcome.accepted else "yellow",
                )
            )
        if outcome.audit is not None:
            color = "green" if outcome.accepted else "red"
            console.print(
                Panel(
                    f"decision: {outcome.audit.decision}\n"
                    f"pass_count: {outcome.audit.pass_count} of 12\n"
                    f"overall_reason: {outcome.audit.overall_reason}",
                    title="audit",
                    style=color,
                )
            )

    console.rule("post sleep state")
    console.print("memory bank stats:")
    console.print(memory_store.stats())
    console.print("skill bank stats:")
    console.print(skill_store.stats())

    # Confirm that the accepted clusters archived their source memory entries.
    archived = memory_store.by_status(EntryStatus.ARCHIVED)
    console.print(f"\narchived memory entries after crystallization: {len(archived)}")
    for entry in archived[:5]:
        console.print(f"  [{entry.crystallized_into}] {entry.question}")

    # Simulate skill failure feedback for the new skill.
    if skill_store.all():
        first_skill = skill_store.all()[0]
        console.rule("simulating skill failures to trigger M3 feedback")
        for _ in range(5):
            skill_store.record_use(first_skill.id, success=False)
        skill_store.record_use(first_skill.id, success=True)
        feedback_events = feedback.run()
        console.print(f"M3 produced {len(feedback_events)} flag events")
        for event in feedback_events:
            console.print(
                f"  skill={event.skill_name}, failure_rate={event.failure_rate:.2f}, "
                f"severity={event.severity:.2f}, flagged={len(event.flagged_entry_ids)}"
            )

    console.print("\nmock pilot completed", style="bold green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
