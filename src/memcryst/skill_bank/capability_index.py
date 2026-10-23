"""Capability index and task signature extractor.

The CapabilityIndex maps a closed-set 4-tuple signature to the list of skill
ids that match. Lookup is O(1) on the hash, then O(K) over the bucket.

The TaskSignatureExtractor parses an incoming task description and returns a
canonical 4-tuple. It tries a rule-based first-verb lookup first, then falls
back to an LLM call when rules fail. The LLM call is rare in practice
because the closed verb set covers the common ALFWorld verbs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Iterable, Optional

from ..core import LLMClient
from .constants import (
    canonicalize_affordance,
    canonicalize_output,
    canonicalize_target_type,
    canonicalize_verb,
    closed_set_summary,
    detect_task_verb_family,
)
from .schema import Capability, Skill

logger = logging.getLogger(__name__)


# Default mapping of verbs to expected output shape. Used as a heuristic
# when the task signature does not name an output explicitly.
_VERB_TO_DEFAULT_OUTPUT: dict[str, str] = {
    "clean": "object_placed",
    "heat": "object_placed",
    "cool": "object_placed",
    "examine": "object_examined",
    "move": "object_placed",
    "search": "object_acquired",
    "open": "location_reached",
    "use": "object_examined",
}


# Words that commonly hint at an affordance inside the task NL.
_AFFORDANCE_HINTS: dict[str, str] = {
    "sink": "sink",
    "microwave": "microwave",
    "fridge": "fridge",
    "refrigerator": "fridge",
    "lamp": "lamp",
    "desklamp": "desklamp",
    "desk lamp": "desklamp",
    "floor lamp": "floorlamp",
    "floorlamp": "floorlamp",
    "stove": "stove",
    "toaster": "toaster",
}


class CapabilityIndex:
    """In-memory index from capability signature to skill ids."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    def register(self, skill: Skill) -> bool:
        """Register a skill in the index. Returns True if registered, False
        if the capability is incomplete (no signature available)."""
        if not skill.capability.is_complete():
            return False
        sig = skill.capability.signature()
        if skill.id not in self._buckets[sig]:
            self._buckets[sig].append(skill.id)
        return True

    def deregister(self, skill: Skill) -> None:
        if not skill.capability.is_complete():
            return
        sig = skill.capability.signature()
        if skill.id in self._buckets.get(sig, []):
            self._buckets[sig].remove(skill.id)
            if not self._buckets[sig]:
                del self._buckets[sig]

    def lookup(self, sig: tuple[str, str, str, str]) -> list[str]:
        return list(self._buckets.get(sig, []))

    def lookup_relaxed(
        self,
        sig: tuple[str, str, str, str],
        relax_axes: Iterable[str] = ("affordance",),
    ) -> list[str]:
        """Fallback lookup when the exact signature has no matches. We relax
        one or more axes to wildcards. The default relaxes affordance because
        a task may not name a specific appliance even when one applies.
        Returns deduplicated skill ids."""
        relax_set = set(relax_axes)
        axes = ("verb_family", "target_type", "affordance", "output")
        results: list[str] = []
        seen: set[str] = set()
        for bucket_sig, ids in self._buckets.items():
            ok = True
            for i, axis in enumerate(axes):
                if axis in relax_set:
                    continue
                if bucket_sig[i] != sig[i]:
                    ok = False
                    break
            if ok:
                for sid in ids:
                    if sid not in seen:
                        seen.add(sid)
                        results.append(sid)
        return results

    def stats(self) -> dict:
        return {
            "buckets": len(self._buckets),
            "skills_indexed": sum(len(v) for v in self._buckets.values()),
        }

    def __len__(self) -> int:
        return sum(len(v) for v in self._buckets.values())


def _extract_first_verb(text: str) -> Optional[str]:
    """Verb-family from the task NL, modifier-aware.

    Uses the shared detector so "Put a cool X in Y" → cool, matching the
    same logic that m1_miner uses for cluster purity splitting.
    """
    return detect_task_verb_family(text)


def _extract_affordance(text: str) -> Optional[str]:
    text_lower = text.lower()
    # Longer hints first so "desk lamp" beats "lamp".
    for hint in sorted(_AFFORDANCE_HINTS.keys(), key=len, reverse=True):
        if hint in text_lower:
            return canonicalize_affordance(_AFFORDANCE_HINTS[hint])
    return None


def _extract_target_type(text: str) -> str:
    # In v1 PoC we default to object. Location targets are rare in ALFWorld
    # household tasks.
    return "object"


def _extract_output(verb: Optional[str]) -> Optional[str]:
    if verb is None:
        return None
    return _VERB_TO_DEFAULT_OUTPUT.get(verb)


class TaskSignatureExtractor:
    """Extract a capability signature from a task NL string.

    Strategy:
      1. Rule-based first-verb + affordance hint scan. Fast, deterministic.
      2. If rule path fails on verb, LLM fallback (single call).
      3. Canonicalize each axis through the closed-set synonym map.
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm
        self.closed_set = closed_set_summary()

    def extract(self, task: str) -> Capability:
        verb = _extract_first_verb(task)
        afford = _extract_affordance(task)
        out = _extract_output(verb)
        target = _extract_target_type(task)

        if verb is None and self.llm is not None:
            verb, afford2, out2 = self._llm_extract(task)
            if afford is None and afford2 is not None:
                afford = afford2
            if out is None and out2 is not None:
                out = out2

        if afford is None:
            afford = "none"

        cap = Capability(
            verb_family=verb,
            target_type=target,
            affordance=afford,
            output=out,
            preconditions=[],
            task_pattern=task.strip(),
        )
        return cap

    def _llm_extract(self, task: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        prompt = (
            "Extract a capability signature for the following task. Return "
            "EXACTLY one JSON object with fields verb_family, affordance, "
            "output. Each value must come from the listed enums. If no "
            "affordance applies, use 'none'.\n\n"
            f"verb_family enum: {self.closed_set['verb_family']}\n"
            f"affordance enum: {self.closed_set['affordance']}\n"
            f"output enum: {self.closed_set['output']}\n\n"
            f"Task: {task}\n\n"
            'Respond with just JSON like {"verb_family":"...","affordance":"...","output":"..."}'
        )
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            data = _parse_first_json(raw)
            if not data:
                return None, None, None
            verb = canonicalize_verb(data.get("verb_family", ""))
            afford = canonicalize_affordance(data.get("affordance", ""))
            out = canonicalize_output(data.get("output", ""))
            return verb, afford, out
        except Exception as exc:
            logger.warning("LLM signature extraction failed, %s", exc)
            return None, None, None


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_first_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        match = _JSON_RE.search(text)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
