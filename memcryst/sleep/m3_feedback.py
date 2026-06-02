"""M3 bidirectional feedback.

When a skill fails often enough during deployment, we trace through its
ancestry pointer back to the source memory entries that birthed the skill
and flag them as unreliable. The flag is a weight reduction plus a status
transition to FLAGGED.

In v1 with the capability index, we also attribute failures to specific
capability axes (verb_family, affordance, target_type, output). Ancestry
entries that match the dominant failure axis get amplified severity. This
is axis-level bidirectional feedback: skill failure → which axis broke →
flag only the relevant memory subspace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..memory_bank import EntryStatus, MemoryEntry, MemoryStore
from ..skill_bank import Skill, SkillStore
from ..skill_bank.constants import (
    canonicalize_affordance,
    canonicalize_verb,
)

logger = logging.getLogger(__name__)


@dataclass
class FlagEvent:
    skill_id: str
    skill_name: str
    failure_rate: float
    severity: float
    flagged_entry_ids: list[str] = field(default_factory=list)
    dominant_axis: Optional[str] = None
    axis_amplified_entry_ids: list[str] = field(default_factory=list)


def _entry_first_verb(entry: MemoryEntry) -> Optional[str]:
    text = (entry.question or "").strip().lower()
    if not text:
        return None
    first = text.split()[0].rstrip(".,!?;:")
    return canonicalize_verb(first)


def _entry_mentions_affordance(entry: MemoryEntry, affordance: str) -> bool:
    if affordance == "none":
        return False
    haystack = " ".join(
        [entry.question or "", entry.improvement or "", entry.failure_reason or ""]
    ).lower()
    if affordance in haystack:
        return True
    # Light synonym scan
    if affordance == "sink" and ("wash" in haystack or "rinse" in haystack):
        return True
    if affordance == "microwave" and ("heat" in haystack or "warm" in haystack):
        return True
    if affordance in ("lamp", "desklamp", "floorlamp") and "light" in haystack:
        return True
    return False


def _entry_matches_axis(entry: MemoryEntry, axis: str, skill: Skill) -> bool:
    """Decide whether a memory entry is relevant to the given failure axis.

    This is the v1 PoC heuristic. It is intentionally simple, easy to audit,
    and explainable in the paper. A v2 version would learn the match.
    """
    cap = skill.capability
    if axis == "verb_family" and cap.verb_family:
        entry_verb = _entry_first_verb(entry)
        return entry_verb == cap.verb_family
    if axis == "affordance" and cap.affordance:
        return _entry_mentions_affordance(entry, cap.affordance)
    if axis == "target_type":
        return True  # v1 PoC: target_type axis matches everything (all object tasks)
    if axis == "output":
        return True  # v1 PoC: output axis matches everything
    return False


class FeedbackLoop:
    """Trace skill failures back to source memory and flag entries."""

    def __init__(
        self,
        skill_store: SkillStore,
        memory_store: MemoryStore,
        failure_threshold: float = 0.3,
        min_uses: int = 3,
        max_severity: float = 0.9,
    ) -> None:
        self.skill_store = skill_store
        self.memory_store = memory_store
        self.failure_threshold = failure_threshold
        self.min_uses = min_uses
        self.max_severity = max_severity

    def _should_flag(self, skill: Skill) -> bool:
        if skill.use_count < self.min_uses:
            return False
        return skill.failure_rate() > self.failure_threshold

    def _severity_from(self, skill: Skill) -> float:
        rate = skill.failure_rate()
        if rate <= self.failure_threshold:
            return 0.0
        # Linear ramp from threshold to 1.0.
        scale = (rate - self.failure_threshold) / max(1e-6, 1.0 - self.failure_threshold)
        return min(self.max_severity, scale)

    def _dominant_failure_axis(self, skill: Skill) -> Optional[str]:
        """Pick the axis with the most attributed failures. Returns None if
        no axis failures have been recorded (skill failed without attribution)."""
        counts = skill.axis_failure_counts or {}
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def run(self) -> list[FlagEvent]:
        events: list[FlagEvent] = []
        for skill in self.skill_store:
            if not self._should_flag(skill):
                continue
            severity = self._severity_from(skill)
            dominant_axis = self._dominant_failure_axis(skill)
            flagged_ids: list[str] = []
            amplified_ids: list[str] = []
            dangling = 0
            already_flagged = 0
            for entry_id in skill.ancestry:
                entry = self.memory_store.get(entry_id)
                if entry is None:
                    dangling += 1
                    continue
                if entry.status == EntryStatus.ARCHIVED:
                    continue
                if entry.status == EntryStatus.FLAGGED:
                    already_flagged += 1
                    continue
                # Axis-level severity amplification. If this entry matches the
                # dominant failure axis, apply the full severity. Otherwise
                # apply a reduced fraction so we do not over-flag unrelated
                # ancestors.
                if dominant_axis is not None and _entry_matches_axis(
                    entry, dominant_axis, skill
                ):
                    entry_severity = severity
                    amplified_ids.append(entry_id)
                elif dominant_axis is not None:
                    # Axis is known, this entry does not match. Apply a
                    # fractional flag so we still track but do not over-decay.
                    entry_severity = severity * 0.3
                else:
                    # No axis attribution available, fall back to flat severity.
                    entry_severity = severity
                self.memory_store.flag_unreliable(entry_id, severity=entry_severity)
                flagged_ids.append(entry_id)
            if dangling:
                logger.warning(
                    "M3 saw %d dangling ancestry pointers for skill %s",
                    dangling,
                    skill.name,
                )
            events.append(
                FlagEvent(
                    skill_id=skill.id,
                    skill_name=skill.name,
                    failure_rate=skill.failure_rate(),
                    severity=severity,
                    flagged_entry_ids=flagged_ids,
                    dominant_axis=dominant_axis,
                    axis_amplified_entry_ids=amplified_ids,
                )
            )
            logger.info(
                "M3 flagged %d ancestry entries from skill %s (axis=%s, %d amplified), failure_rate=%.2f, severity=%.2f",
                len(flagged_ids),
                skill.name,
                dominant_axis or "none",
                len(amplified_ids),
                skill.failure_rate(),
                severity,
            )
        return events
