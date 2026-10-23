"""Memory entry schema for MEMCRYST.

Forked from live-evo experience_manager and extended with two MEMCRYST specific
fields. The crystallized_into field tracks the skill identifier when an entry
has been used as ancestry for a crystallized skill. The status field encodes
whether the entry is active, flagged unreliable by skill failure feedback, or
archived after crystallization.

The base fields are question, failure_reason, improvement, missed_information,
category and weight; the remaining fields track provenance, utility and
feedback state.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EntryStatus(str, Enum):
    ACTIVE = "active"
    FLAGGED = "flagged"
    ARCHIVED = "archived"


# Weight parameters borrowed from live-evo experience_manager.
INITIAL_WEIGHT = 1.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
WEIGHT_INCREASE_RATE = 0.3
WEIGHT_DECREASE_RATE = 0.2


@dataclass
class MemoryEntry:
    """A single episodic memory entry.

    Construct with the six base fields. Optional fields take defaults.
    Use to_dict and from_dict for JSONL round trip.
    """

    question: str
    failure_reason: str
    improvement: str
    missed_information: str = ""
    category: str = "other"
    weight: float = INITIAL_WEIGHT

    id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    env: str = ""

    # MEMCRYST extensions.
    crystallized_into: Optional[str] = None
    status: EntryStatus = EntryStatus.ACTIVE
    embedding: Optional[list[float]] = None

    # Optional task signature used by some retrieval modes.
    task_signature: str = ""

    verified_uses: int = 0
    utility: float = 1.0
    priority_next_cluster: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            seed = self.question + str(self.created_at)
            self.id = hashlib.md5(seed.encode()).hexdigest()[:8]

    def clamp_weight(self) -> None:
        self.weight = max(MIN_WEIGHT, min(MAX_WEIGHT, self.weight))

    def text_for_embedding(self) -> str:
        """Compose the text used for embedding based retrieval.

        We concatenate question, failure_reason, and improvement so the
        embedding captures task signature plus what the agent learned.
        """
        parts = [self.question, self.failure_reason, self.improvement]
        return " ".join([p for p in parts if p])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, EntryStatus) else self.status
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        d = dict(d)
        status = d.pop("status", EntryStatus.ACTIVE.value)
        if isinstance(status, str):
            status = EntryStatus(status)
        entry = cls(
            question=d.get("question", ""),
            failure_reason=d.get("failure_reason", ""),
            improvement=d.get("improvement", ""),
            missed_information=d.get("missed_information", ""),
            category=d.get("category", "other"),
            weight=float(d.get("weight", INITIAL_WEIGHT)),
            id=d.get("id", ""),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
            env=d.get("env", ""),
            crystallized_into=d.get("crystallized_into"),
            status=status,
            embedding=d.get("embedding"),
            task_signature=d.get("task_signature", ""),
            verified_uses=int(d.get("verified_uses", 0)),
            utility=float(d.get("utility", 1.0)),
            priority_next_cluster=bool(d.get("priority_next_cluster", False)),
        )
        return entry
