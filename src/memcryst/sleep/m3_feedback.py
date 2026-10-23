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
import time
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

    A simple, auditable heuristic.
    """
    cap = skill.capability
    if axis == "verb_family" and cap.verb_family:
        entry_verb = _entry_first_verb(entry)
        return entry_verb == cap.verb_family
    if axis == "affordance" and cap.affordance:
        return _entry_mentions_affordance(entry, cap.affordance)
    if axis == "target_type":
        return True
    if axis == "output":
        return True
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
        axis_aware: bool = True,
        gc_threshold: float = 0.0,
        target: str = "memory",
        protect_verified: bool = False,
        revive_archived_on_fail: bool = False,
        priority_cluster_on_fail: bool = False,
    ) -> None:
        self.skill_store = skill_store
        self.memory_store = memory_store
        self.failure_threshold = failure_threshold
        self.min_uses = min_uses
        self.max_severity = max_severity
        self.axis_aware = axis_aware
        self.gc_threshold = gc_threshold
        self.target = target
        # protect_verified: skip flagging entries with verified_uses > 0.
        self.protect_verified = protect_verified
        self.revive_archived_on_fail = revive_archived_on_fail
        self.priority_cluster_on_fail = priority_cluster_on_fail

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
            dominant_axis = (
                self._dominant_failure_axis(skill) if self.axis_aware else None
            )

            if self.target in ("skill", "both"):
                old_utility = skill.utility
                skill.utility = max(0.0, skill.utility * (1.0 - 0.5 * severity))
                self.skill_store.save_one(skill)
                logger.info(
                    "M3 (target=%s) decayed skill %s utility %.2f -> %.2f, severity=%.2f",
                    self.target, skill.name, old_utility, skill.utility, severity,
                )

            if self.revive_archived_on_fail:
                revived = 0
                for entry_id in skill.ancestry:
                    entry = self.memory_store.get(entry_id)
                    if entry is None:
                        continue
                    if entry.status == EntryStatus.ARCHIVED:
                        entry.status = EntryStatus.ACTIVE
                        entry.crystallized_into = None
                        entry.updated_at = time.time()
                        revived += 1
                if revived > 0:
                    self.memory_store.save()
                    logger.info("N7 revived %d archived ancestry entries for failing skill %s", revived, skill.name)

            if self.priority_cluster_on_fail:
                priority_marked = 0
                for entry_id in skill.ancestry:
                    entry = self.memory_store.get(entry_id)
                    if entry is None or entry.status == EntryStatus.ARCHIVED:
                        continue
                    if not entry.priority_next_cluster:
                        entry.priority_next_cluster = True
                        entry.updated_at = time.time()
                        priority_marked += 1
                if priority_marked > 0:
                    self.memory_store.save()
                    logger.info("N12 marked %d ancestry entries priority_next_cluster for failing skill %s", priority_marked, skill.name)

            # If target is "skill" only (or "none": mark-only mode for the
            # N12 generator-prioritization arm — no weight/utility change at
            # all), skip the memory branch entirely.
            if self.target in ("skill", "none"):
                events.append(
                    FlagEvent(
                        skill_id=skill.id,
                        skill_name=skill.name,
                        failure_rate=skill.failure_rate(),
                        severity=severity,
                        flagged_entry_ids=[],
                        dominant_axis=None,
                        axis_amplified_entry_ids=[],
                    )
                )
                continue

            flagged_ids: list[str] = []
            amplified_ids: list[str] = []
            dangling = 0
            already_flagged = 0
            protected = 0
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
                if self.protect_verified and entry.verified_uses > 0:
                    protected += 1
                    continue
                if not self.axis_aware:
                    # Paper Eq 9: uniform gamma scaled by f(s_j). All ancestry
                    # entries receive the same severity. No per-entry axis
                    # amplification or attenuation.
                    entry_severity = severity
                elif dominant_axis is not None and _entry_matches_axis(
                    entry, dominant_axis, skill
                ):
                    # Axis-level severity amplification. Matching ancestry
                    # gets the full severity.
                    entry_severity = severity
                    amplified_ids.append(entry_id)
                elif dominant_axis is not None:
                    # Axis is known, this entry does not match. Apply a
                    # fractional flag so we still track but do not over-decay.
                    entry_severity = severity * 0.3
                else:
                    # axis_aware=True but no axis attribution available.
                    # Fall back to flat severity.
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
                "M3 flagged %d ancestry entries from skill %s (axis=%s, %d amplified, %d protected), failure_rate=%.2f, severity=%.2f",
                len(flagged_ids),
                skill.name,
                dominant_axis or "none",
                len(amplified_ids),
                protected,
                skill.failure_rate(),
                severity,
            )

        # Paper Alg 1 line 11: garbage-collect entries whose weight dropped
        # below gc_threshold and that belong to some skill ancestry. This is
        # the GC step proposal §4.4 calls w^arch. Disabled when gc_threshold<=0.
        if self.gc_threshold > 0:
            ancestry_set: set[str] = set()
            for s in self.skill_store:
                ancestry_set.update(s.ancestry)
            gc_count = 0
            for entry_id in list(ancestry_set):
                entry = self.memory_store.get(entry_id)
                if entry is None:
                    continue
                if entry.status == EntryStatus.ARCHIVED:
                    continue
                if entry.weight < self.gc_threshold:
                    # Find the first skill that has this entry in ancestry
                    # for the crystallized_into reference.
                    crystal_id = ""
                    for s in self.skill_store:
                        if entry_id in s.ancestry:
                            crystal_id = s.id
                            break
                    self.memory_store.archive(entry_id, crystal_id)
                    gc_count += 1
            if gc_count > 0:
                logger.info(
                    "M3 GC archived %d entries with weight < %.2f",
                    gc_count, self.gc_threshold,
                )

        return events
