"""Skill schema for MEMCRYST.

Skills are stored as SKILL.md files with YAML frontmatter, following the
MUSE convention. We add a small number of MEMCRYST specific fields to track
provenance and utility.

Frontmatter format example.

    ---
    name: clean-object-procedure
    description: Use when an object must be brought to a sink and washed.
    category: general
    ancestry: [a1b2c3d4, e5f6g7h8]
    crystallized_from_cluster: 3
    crystallization_round: 1
    utility: 0.6
    use_count: 0
    failure_count: 0
    ---

    # Body of the skill in markdown follows the frontmatter.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .constants import (
    canonicalize_verb,
    canonicalize_affordance,
    canonicalize_output,
    canonicalize_target_type,
    canonicalize_precondition,
)


VALID_CATEGORIES = {
    "general",
    "coding",
    "research",
    "data_analysis",
    "security",
    "communication",
    "automation",
    "agentic",
    "productivity",
    "common_mistakes",
}


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "unnamed-skill"


@dataclass
class Capability:
    """Closed-set capability signature for indexed retrieval and axis-level feedback.

    The 5 axes are validated against the closed sets in constants.py at parse
    time. Tokens that fall outside the closed set go through synonym fallback;
    if that fails too, the axis is set to None and the skill cannot be
    indexed (M2 audit check #13 will catch it).
    """

    verb_family: Optional[str] = None
    target_type: Optional[str] = None
    affordance: Optional[str] = None
    output: Optional[str] = None
    preconditions: list[str] = field(default_factory=list)
    task_pattern: str = ""

    def is_complete(self) -> bool:
        return all([
            self.verb_family is not None,
            self.target_type is not None,
            self.affordance is not None,
            self.output is not None,
        ])

    def signature(self) -> tuple[str, str, str, str]:
        """The hash key used by CapabilityIndex. None-safe."""
        return (
            self.verb_family or "_unknown",
            self.target_type or "_unknown",
            self.affordance or "none",
            self.output or "_unknown",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb_family": self.verb_family,
            "target_type": self.target_type,
            "affordance": self.affordance,
            "output": self.output,
            "preconditions": list(self.preconditions),
            "task_pattern": self.task_pattern,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "Capability":
        if not data:
            return cls()
        # Canonicalize. Each token gets folded through the synonym map if it
        # is not already a member of the closed set. Returns None on failure
        # so the caller can decide whether to reject the skill.
        verb = data.get("verb_family")
        verb_canon = canonicalize_verb(verb) if verb is not None else None
        target = data.get("target_type")
        target_canon = canonicalize_target_type(target) if target is not None else None
        afford = data.get("affordance")
        afford_canon = canonicalize_affordance(afford) if afford is not None else None
        out = data.get("output")
        out_canon = canonicalize_output(out) if out is not None else None
        raw_preconds = data.get("preconditions") or []
        preconds = []
        for tag in raw_preconds:
            tag_canon = canonicalize_precondition(tag)
            if tag_canon is not None:
                preconds.append(tag_canon)
        return cls(
            verb_family=verb_canon,
            target_type=target_canon,
            affordance=afford_canon,
            output=out_canon,
            preconditions=preconds,
            task_pattern=data.get("task_pattern", "") or "",
        )


@dataclass
class Skill:
    """A procedural skill artifact.

    The body field holds the markdown body of the SKILL.md file. The other
    fields map to YAML frontmatter.
    """

    name: str
    description: str
    body: str
    category: str = "general"

    # Provenance and lifecycle.
    ancestry: list[str] = field(default_factory=list)
    crystallized_from_cluster: Optional[int] = None
    crystallization_round: int = 1

    # Utility tracking.
    utility: float = 0.5
    use_count: int = 0
    failure_count: int = 0

    # Identity and timing.
    id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    embedding: Optional[list[float]] = None

    # Capability signature for indexed retrieval. Defaults to an empty
    # Capability that fails is_complete(). Such skills can still live in the
    # store but cannot be reached through capability-indexed lookup, only
    # through the legacy similarity path.
    capability: Capability = field(default_factory=Capability)

    # Parameter schema. Keyed by placeholder name (e.g. "target_object").
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Optional Voyager / WebXSkill Grounded mode hook. v1 keeps this None.
    # When non-None, a Python file lives next to SKILL.md and can be invoked.
    script: Optional[str] = None

    # Axis-level failure counters from M3 axis-level feedback. Maps
    # axis name to count of failed uses attributed to that axis.
    axis_failure_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            self.category = "general"
        if not self.id:
            seed = self.name + str(self.created_at)
            self.id = hashlib.md5(seed.encode()).hexdigest()[:8]

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def text_for_embedding(self) -> str:
        return f"{self.name}. {self.description}"

    def failure_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.failure_count / self.use_count

    # ----- frontmatter and markdown round trip -----

    def to_frontmatter_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "ancestry": self.ancestry,
            "crystallized_from_cluster": self.crystallized_from_cluster,
            "crystallization_round": self.crystallization_round,
            "utility": round(self.utility, 4),
            "use_count": self.use_count,
            "failure_count": self.failure_count,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "capability": self.capability.to_dict(),
            "parameters": self.parameters,
            "script": self.script,
            "axis_failure_counts": self.axis_failure_counts,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_frontmatter(cls, meta: dict[str, Any], body: str) -> "Skill":
        return cls(
            name=meta.get("name", "unnamed-skill"),
            description=meta.get("description", ""),
            body=body,
            category=meta.get("category", "general"),
            ancestry=list(meta.get("ancestry", []) or []),
            crystallized_from_cluster=meta.get("crystallized_from_cluster"),
            crystallization_round=int(meta.get("crystallization_round", 1)),
            utility=float(meta.get("utility", 0.5)),
            use_count=int(meta.get("use_count", 0)),
            failure_count=int(meta.get("failure_count", 0)),
            id=meta.get("id", ""),
            created_at=float(meta.get("created_at", time.time())),
            updated_at=float(meta.get("updated_at", time.time())),
            capability=Capability.from_dict(meta.get("capability")),
            parameters=dict(meta.get("parameters") or {}),
            script=meta.get("script"),
            axis_failure_counts=dict(meta.get("axis_failure_counts") or {}),
        )
