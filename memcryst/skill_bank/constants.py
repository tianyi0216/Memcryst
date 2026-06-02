"""Closed sets for the v1 capability index.

We hand-curate small enums for each capability axis and a synonym map that
folds LLM emissions into the canonical token. The closed-set design is what
makes O(K) hash lookup possible at retrieval time and what enables axis-level
discrete bidirectional feedback from M3.

The enums target ALFWorld household tasks because that is the v1 PoC
environment. Future versions will swap the env-specific tables but the
mechanism stays the same.
"""

from __future__ import annotations

# ----- Axis 1: verb family -----
VERB_FAMILY: frozenset[str] = frozenset({
    "clean",     # wash, rinse, scrub
    "heat",      # microwave, warm, cook, boil
    "cool",      # chill, freeze, refrigerate
    "examine",   # inspect, look at, view, read
    "move",      # place, put, bring, relocate
    "search",    # find, locate, look for
    "open",      # open, unlock
    "use",       # use, activate, operate
})

# ----- Axis 2: affordance (env-specific receptacle / tool) -----
AFFORDANCE: frozenset[str] = frozenset({
    "sink",
    "microwave",
    "fridge",
    "lamp",
    "desklamp",
    "floorlamp",
    "stove",
    "toaster",
    "none",
})

# ----- Axis 3: output shape -----
OUTPUT: frozenset[str] = frozenset({
    "object_placed",     # processed and placed at destination
    "object_examined",   # observed under a probe
    "location_reached",  # navigation only
    "object_processed",  # state change without placement
    "object_acquired",   # picked up only
})

# ----- Axis 4: target type -----
TARGET_TYPE: frozenset[str] = frozenset({
    "object",
    "location",
    "abstract",
})

# ----- Axis 5: precondition tags -----
PRECONDITION_TAGS: frozenset[str] = frozenset({
    "object_at_source",
    "destination_reachable",
    "appliance_available",
    "object_in_inventory",
    "light_source_present",
})


# ----- Synonym map. LLM emits one of these, we fold to canonical -----
# Keys are lowercased tokens. Used as a fallback before forcing a re-emission.
VERB_SYNONYMS: dict[str, str] = {
    "wash": "clean",
    "rinse": "clean",
    "scrub": "clean",
    "wipe": "clean",
    "microwave": "heat",
    "warm": "heat",
    "cook": "heat",
    "boil": "heat",
    "chill": "cool",
    "freeze": "cool",
    "refrigerate": "cool",
    "inspect": "examine",
    "look": "examine",
    "look_at": "examine",
    "view": "examine",
    "read": "examine",
    "observe": "examine",
    "place": "move",
    "put": "move",
    "bring": "move",
    "relocate": "move",
    "carry": "move",
    "deliver": "move",
    "find": "search",
    "locate": "search",
    "search_for": "search",
    "unlock": "open",
    "activate": "use",
    "operate": "use",
    "turn_on": "use",
}

AFFORDANCE_SYNONYMS: dict[str, str] = {
    "sinkbasin": "sink",
    "sink_basin": "sink",
    "washbasin": "sink",
    "kitchen_sink": "sink",
    "microwave_oven": "microwave",
    "oven_microwave": "microwave",
    "refrigerator": "fridge",
    "ice_box": "fridge",
    "desk_lamp": "desklamp",
    "table_lamp": "desklamp",
    "floor_lamp": "floorlamp",
    "side_lamp": "lamp",
    "reading_lamp": "lamp",
    "stovetop": "stove",
    "stove_top": "stove",
    "burner": "stove",
}

OUTPUT_SYNONYMS: dict[str, str] = {
    "placed": "object_placed",
    "examined": "object_examined",
    "reached": "location_reached",
    "processed": "object_processed",
    "acquired": "object_acquired",
    "picked_up": "object_acquired",
    "object_observed": "object_examined",
}


def canonicalize_verb(token: str) -> str | None:
    """Return the canonical verb family or None if it cannot be mapped."""
    t = token.strip().lower().replace(" ", "_")
    if t in VERB_FAMILY:
        return t
    return VERB_SYNONYMS.get(t)


def canonicalize_affordance(token: str) -> str | None:
    if token is None:
        return "none"
    t = token.strip().lower().replace(" ", "_")
    if t in AFFORDANCE:
        return t
    return AFFORDANCE_SYNONYMS.get(t)


def canonicalize_output(token: str) -> str | None:
    t = token.strip().lower().replace(" ", "_")
    if t in OUTPUT:
        return t
    return OUTPUT_SYNONYMS.get(t)


def canonicalize_target_type(token: str) -> str | None:
    t = token.strip().lower().replace(" ", "_")
    if t in TARGET_TYPE:
        return t
    return None


def canonicalize_precondition(token: str) -> str | None:
    t = token.strip().lower().replace(" ", "_")
    if t in PRECONDITION_TAGS:
        return t
    return None


def closed_set_summary() -> dict[str, list[str]]:
    """Used by the synthesizer prompt to list allowed values."""
    return {
        "verb_family": sorted(VERB_FAMILY),
        "affordance": sorted(AFFORDANCE),
        "output": sorted(OUTPUT),
        "target_type": sorted(TARGET_TYPE),
        "precondition_tags": sorted(PRECONDITION_TAGS),
    }


# Modifier patterns. ALFWorld tasks often phrase processing-then-place as
# "put a <modifier> X in Y" where the modifier carries the verb_family
# information, not the leading "put". We detect these explicitly so the
# clustering layer and the task signature extractor agree on family.
import re as _re

_MODIFIER_PATTERN = _re.compile(
    r"\b(?:a|an|some|the)\s+(clean|cool|cold|hot|heated|warm)\s+\w+",
    _re.IGNORECASE,
)


def detect_task_verb_family(text: str) -> str | None:
    """Map a task description NL to a canonical verb_family.

    Priority order:
      1. `examine` / `look at` keywords → examine
      2. Modifier inside "a <modifier> X" pattern → modifier-driven family
      3. Canonicalized first verb (via canonicalize_verb)
      4. None if nothing matches

    This is the SHARED canonical extraction used by m1_miner (verb purity
    split during clustering) and capability_index (task signature lookup at
    retrieval time). Keep them in sync by going through this one function.
    """
    text = (text or "").strip().lower()
    if not text:
        return None
    # 1. examine takes precedence — ALFWorld look_at_obj tasks always say one of these.
    if "examine" in text.split() or "look at" in text:
        return "examine"
    # 2. modifier inside a "a/some/the <modifier> X" pattern.
    m = _MODIFIER_PATTERN.search(text)
    if m:
        modifier = m.group(1).lower()
        if modifier == "clean":
            return "clean"
        if modifier in ("cool", "cold"):
            return "cool"
        if modifier in ("hot", "heated", "warm"):
            return "heat"
    # 3. fall back to first-verb canonicalization.
    first = text.split()[0].rstrip(".,!?;:")
    return canonicalize_verb(first)
