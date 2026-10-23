"""ReAct-style actor with pluggable context: retrieved skills (procedures) and
retrieved memory (episodic entries and semantic facts) are rendered into the
prompt; the action space, parser and loop are identical across modes.

Modes:
- vanilla     no retrieval; plain ReAct over the valid-action list
- similarity  retrieve top-K skills by embedding similarity
- capability  retrieve top-K skills through the capability index

Each run returns a TrajectoryResult (success, steps, used skill/memory ids)
that the wake-time writeback and the sleep-time feedback consume.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..memory_bank.condense import CondensedEntry, CondensedMemoryRetriever
from ..core import LLMClient
from ..memory_bank import MemoryEntry, MemoryRetriever
from ..skill_bank import Skill, SkillRetriever
from .alfworld_env import AlfworldEnv, ResetResult, StepResult, family_from_task_description

logger = logging.getLogger(__name__)


Mode = Literal["vanilla", "react", "memory", "memory_condense", "similarity", "capability", "capability_plus_memory", "capability_plus_condense", "capability_plus_semantic"]


@dataclass
class StepTrace:
    """One step of a trajectory, exposed for downstream writeback / attribution."""
    step_i: int
    thought: str
    action: str
    obs: str
    reward: float
    done: bool


@dataclass
class TrajectoryResult:
    task_idx: int
    task_description: str
    mode: Mode
    success: bool
    steps: int
    final_reward: float
    used_skill_ids: list[str] = field(default_factory=list)
    used_memory_ids: list[str] = field(default_factory=list)
    retrieval_paths: Counter = field(default_factory=Counter)
    invalid_action_retries: int = 0
    llm_calls: int = 0
    error: Optional[str] = None
    # Per-step trajectory exposed for trajectory writeback + axis attribution.
    # Populated in run(). Empty if max_steps reached without any step.
    step_trace: list[StepTrace] = field(default_factory=list)
    # The final observation before termination, useful for failure analysis.
    final_obs: str = ""
    # The RESET observation (task content lives here for qa/longqa streams —
    # step_trace[i].obs is the post-action obs, e.g. the env's "Noted." ack).
    initial_obs: str = ""
    diag: dict = field(default_factory=dict)


_ENV_KIND = os.environ.get("MEMCRYST_ENV", "alfworld").strip().lower()


def _taskstream_action_passthrough(action: str):
    """Taskstream envs consume free-form output (the env parses the answer),
    so no list matching is applied."""
    if _ENV_KIND != "taskstream":
        return None
    return (action or "").strip() or None


_TASKSTREAM_SYS_PROMPT = """You are an expert problem solver working through a stream of tasks.

Skill blocks and memory notes, when provided, are advisory: use them if they help,
ignore them if they do not apply to the current task.

Work each task IN FULL — reason step by step as deeply as the problem requires.
Follow the answer-format instructions given in the observation (e.g. include ONE
```python block when allowed, and finish with the requested answer line)."""

_SYS_PROMPT = """You are an autonomous agent solving an ALFWorld household task.

# Action policy

When a skill block is provided, follow its Procedure step by step. Identify which
step of the procedure you are currently executing and pick the action that advances
that step.

# Action priority (apply in order)

1. **If the target object is visible in the current observation AND you are not
   holding it yet, your next action MUST be `take <target_object> from <receptacle>`.**
   Do NOT navigate to another location once you have seen the target. Acting beats
   exploring.
2. If you are already holding the target object and the next procedure step is a
   manipulation primitive (`cool <obj> with fridge`, `heat <obj> with microwave`,
   `clean <obj> with sink`, `use <lamp>`), execute it directly.
3. **Process exactly once.** After you have run `cool` / `heat` / `clean` / `use`
   on the target ONE time, do NOT run another manipulation primitive on it. The
   next action is either `go to <destination>` or `put <object> in/on <destination>`.
4. **Destination vs appliance.** In a task like `put a cool tomato in microwave`,
   `microwave` is the DESTINATION, not the appliance. The cooling appliance is
   `fridge`. After cooling, your action at the microwave MUST be
   `put <target> in/on microwave`, NOT `heat <target> with microwave`. The same
   rule applies for sink, fridge, and lamp when they appear as destinations.
5. If you are holding the target object and at the destination, execute
   `put <object> in/on <destination>`.
6. Only emit `go to <X>` when you are not yet at the location required by the
   next procedure step. Pick `take` or `put` over `go to` whenever both are valid.
7. The target object in the task is the noun that follows "a/some/the". For
   example in "put a cool tomato in microwave" the target is `tomato`. The
   appliance words (microwave, fridge, sink, lamp) are NOT the target.

# Output

Output exactly one action from the valid actions list. Be efficient."""


def _format_condensed_entry(entry: CondensedEntry, idx: int) -> str:
    """Render a condensed memory summary (cluster + LLM summarize) as a
    prompt block (natural-language cluster summary, no skill structure).
    """
    return (
        f"Past experience cluster {idx} (consolidated from {entry.cluster_size} similar trajectories):\n"
        f"  {entry.summary.strip()}"
    )


def _format_memory_entry(entry: MemoryEntry, idx: int) -> str:
    """Render a raw memory entry as a prompt block. Used for the memory_raw
    baseline that retrieves directly from the memory bank without any
    crystallization into skill structure.
    """
    pieces = [f"Experience {idx} (weight={entry.weight:.2f}):"]
    pieces.append(f"  question: {entry.question}")
    if entry.improvement:
        pieces.append(f"  improvement: {entry.improvement.strip()[:400]}")
    if entry.failure_reason:
        pieces.append(f"  failure_reason: {entry.failure_reason.strip()[:300]}")
    if entry.missed_information:
        pieces.append(f"  missed_information: {entry.missed_information.strip()[:200]}")
    return "\n".join(pieces)


def _format_skill(skill: Skill, idx: int) -> str:
    procedure = _extract_section(skill.body, "Procedure") or ""
    when = _extract_section(skill.body, "When to use") or skill.description
    cap = skill.capability
    cap_line = ""
    if cap.is_complete():
        cap_line = (
            f"  capability: verb={cap.verb_family} affordance={cap.affordance} "
            f"output={cap.output}\n"
        )
    # Lessons the skill accumulated from its own failed uses.
    notes = getattr(skill, "failure_notes", None) or []
    notes_block = ""
    if notes:
        notes_block = "\n  known_failure_modes (avoid these):\n" + "".join(
            f"    - {n.strip()[:160]}\n" for n in notes[:4])
    return (
        f"Skill {idx}: {skill.name}\n"
        f"  description: {skill.description}\n"
        f"{cap_line}"
        f"  when_to_use: {when.strip()[:300]}\n"
        f"  procedure:\n{_indent(procedure.strip()[:600], '    ')}"
        f"{notes_block}"
    )


def _extract_section(body: str, section_name: str) -> Optional[str]:
    lines = (body or "").splitlines()
    collected: list[str] = []
    in_sec = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(f"## {section_name.lower()}"):
            in_sec = True
            continue
        if in_sec and stripped.startswith("## "):
            break
        if in_sec:
            collected.append(line)
    if not collected:
        return None
    return "\n".join(collected)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines() if line.strip())


_THOUGHT_RE = re.compile(r"thought\s*:\s*(.+?)(?=\n\s*action\s*:|$)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"action\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE | re.DOTALL)

# Parse "On the X, you see ..." / "In it, you see ..." / "The X is closed."
_CONTENTS_RE = re.compile(r"(?:you see|in it,? you see)\s+(.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL)
_CLOSED_RE = re.compile(r"(?:is closed|\bclosed\b)", re.IGNORECASE)
_RECEP_LIST_RE = re.compile(r"you see\s+(.+?)\.", re.IGNORECASE | re.DOTALL)


def _extract_contents(obs: str) -> str:
    """Pull a short summary of what is at the visited location from its obs."""
    if not obs:
        return "(no obs)"
    short = obs.strip().split("\n")[0]
    if _CLOSED_RE.search(short) and "you see" not in short.lower():
        return "closed, not opened"
    m = _CONTENTS_RE.search(short)
    if m:
        contents = m.group(1).strip()
        if len(contents) > 120:
            contents = contents[:120] + "..."
        if "nothing" in contents.lower():
            return "empty"
        return contents
    return short[:120]


def _action_priority(a: str) -> int:
    """Lower number = higher priority in the prompt listing.

    take and put share priority 0 because they are the endpoints of a task
    and rarely the wrong choice. Manipulation primitives (cool / heat /
    clean / use) get priority 1, but the system prompt enforces "process
    once only", so they should not dominate. Open/close are housekeeping
    and go-to is the most generic, so they go last.
    """
    al = a.lower().strip()
    if al.startswith("take ") or al.startswith("put ") or al.startswith("place "):
        return 0
    if al.startswith(("clean ", "heat ", "cool ", "use ")):
        return 1
    if al.startswith(("open ", "close ")):
        return 2
    if al.startswith("go to "):
        return 3
    return 4


def _extract_initial_receptacles(obs: str) -> list[str]:
    """Parse the initial observation listing of receptacles."""
    if not obs:
        return []
    m = _RECEP_LIST_RE.search(obs)
    if not m:
        return []
    items = m.group(1)
    # Strip leading "a " / "an " articles, split on commas and "and".
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", items)
    out = []
    for p in parts:
        p = p.strip().lstrip("a ").lstrip("an ").strip()
        if p and len(p) < 40:
            out.append(p)
    return out


def _parse_react(raw: str) -> tuple[str, str]:
    """Return (thought, action). Either may be empty string if not found."""
    thought_match = _THOUGHT_RE.search(raw or "")
    action_match = _ACTION_RE.search(raw or "")
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    # Strip backticks and quotes that LLMs love to add.
    action = action.strip("`'\" ")
    # Some LLMs prefix with "1." or bullets.
    action = re.sub(r"^[\d\.\)\-\*\s]+", "", action).strip()
    return thought, action


_PUT_TO_MOVE_RE = re.compile(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", re.IGNORECASE)

# Action-grammar adapter for weak open-source backbones. Strong models (e.g.
# invalid `take <obj> from <recep>` without first navigating, then repeat it
# every step until max_steps -> task fails. Controlled per-run via the env var:
#   MEMCRYST_ACTION_GRAMMAR = none     (default; original behavior)
#                             redirect (turn an invalid manipulation into the
#                                       'go to <recep>' it presupposes)
#                             guided   (vLLM guided_choice: model MUST pick a
#                                       valid action; fixation impossible)
_ACTION_GRAMMAR = os.environ.get("MEMCRYST_ACTION_GRAMMAR", "none").lower().strip()

# Variant 3: inject an explicit navigation-first precondition rule into the
# prompt. Counteracts the skill-priming + manipulation-first action ordering
# that make weak models attempt `take`/`put` before navigating.
_NAV_HINT = os.environ.get("MEMCRYST_NAV_HINT", "0").strip().lower() not in ("", "0", "false", "no")
# In-chain retrieval (R4R-inspired): re-retrieve skill/memory every K steps using
# the PARTIAL TRAJECTORY as query, so injected guidance tracks the current phase.
# 0 = off (one-shot retrieval = current behavior). Gate/family decision stays on
# the original task description; only the retrieved CONTENT follows the phase.
_INCHAIN = int(os.environ.get("MEMCRYST_INCHAIN", "0") or "0")
_WM_ENFORCE = os.environ.get("MEMCRYST_WM_ENFORCE", "0").strip().lower() not in ("", "0", "false", "no")
# WM revisit threshold: redirect a revisit only after the target has ALREADY been
# visited >= this many times. Default 1 = block on the first revisit (original
# behavior). 2 = allow ONE legitimate return (the agent re-checking a place it
# saw the object at) and only break the 3rd+ revisit (the pathological 4+ loop) —
# fixes WM's over-block of clean/move tasks that need a legitimate return.
_WM_REVISIT_MIN = int(os.environ.get("MEMCRYST_WM_REVISIT_MIN", "1"))
# skill-framed fact verification (DIAGNOSTIC only — records verified/refuted on each
# retrieved location fact, changes NO behavior). When on, the run loop tracks where
# each fact's entity is actually observed and, at episode end, scores whether the
# fact's predicted receptacle-type matched reality.
_FACT_VERIFY = os.environ.get("MEMCRYST_FACT_VERIFY", "0").strip().lower() not in ("", "0", "false", "no")
# Use an LLM to classify location categories instead of the hardcoded RECEPTACLE_CATEGORY
# table (domain-agnostic — the mechanism generalizes to QA/other domains by swapping
# the taxonomy prompt). ~25 cached calls total.
_FACT_CAT_LLM = os.environ.get("MEMCRYST_FACT_CAT_LLM", "0").strip().lower() not in ("", "0", "false", "no")
# Act on the verification signal: skip facts the environment has repeatedly refuted
# (read-only quarantine). Implies _FACT_VERIFY (must record to act). This is the
# step that turns the clean truth signal into a behavior — the bidirectional
# co-evolution loop (skill-framed test -> correct the grounding memory).
_FACT_QUARANTINE = os.environ.get("MEMCRYST_FACT_QUARANTINE", "0").strip().lower() not in ("", "0", "false", "no")
# (a) soft down-rank: >0 multiplies a refuted fact's score instead of skipping it, so
#     partially-correct multi-location priors stay available (default 0 = hard skip).
_FACT_QUAR_PENALTY = float(os.environ.get("MEMCRYST_FACT_QUAR_PENALTY", "0"))
# (b') verified-aware: only quarantine facts verified at most this many times. 0 = drop
#     ONLY never-confirmed spurious facts (tomato-in-fridge 0/3) and spare real modes
#     (pencil-on-sidetable 5/8). Default huge = quarantine regardless of verified.
_FACT_QUAR_MAXV = int(os.environ.get("MEMCRYST_FACT_QUAR_MAXV", "1000000000"))
# Judge quarantine at COARSE category granularity (surface/container/appliance) — the
# signal that actually tracks usefulness (r=-0.40 vs type's -0.12). Removes wrong-
# category hallucinations, spares right-category-wrong-instance priors.
_FACT_QUAR_CATEGORY = os.environ.get("MEMCRYST_FACT_QUAR_CATEGORY", "0").strip().lower() not in ("", "0", "false", "no")
# Unified DUAL-LAYER memory: add a gated, de-instantiated RAW episodic block next to
# the fact layer. Channel isolation (raw never enters confidence feedback) +
# relevance gate (inject only chunks scoring >= tau; cross-room stale replays filter
# out naturally) + de-instantiation (mask instance ids like 'desk 1' -> 'a desk',
# killing the prior-welding failure mode diagnosed in June). Defaults OFF.
_RAW_LAYER = os.environ.get("MEMCRYST_RAW_LAYER", "0").strip().lower() not in ("", "0", "false", "no")
_RAW_GATE_TAU = float(os.environ.get("MEMCRYST_RAW_GATE_TAU", "0.55"))
_RAW_TOPK = int(os.environ.get("MEMCRYST_RAW_TOPK", "2"))
_RAW_DEINST = os.environ.get("MEMCRYST_RAW_DEINST", "1").strip().lower() not in ("", "0", "false", "no")


def _deinstantiate(text: str) -> str:
    """Mask numbered instance ids: 'go to desk 1' -> 'go to a desk'. Keeps the
    PROCEDURE, removes the weldable instance grounding."""
    return re.sub(r"\b([a-z]+)\s+\d+\b", r"a \1", text)
# A — skill-conditioned semantic retrieval: when on, facts whose supports_skills
# ancestry edge intersects the task's active skills get boosted in fact ranking
# (read-only skill→memory coupling). Default off (preserves topk4).
_SKILL_COND_RETRIEVAL = os.environ.get("MEMCRYST_SKILL_COND_RETRIEVAL", "0").strip().lower() not in ("", "0", "false", "no")
_SKILL_COND_BOOST = float(os.environ.get("MEMCRYST_SKILL_COND_BOOST", "1.3"))
# Skill-retrieval fallback: when capability-index buckets are empty (e.g. WebShop,
# whose queries do not match ALFWorld-style verb signatures), fall back to cosine
# similarity so skills still get injected. Default off (preserves ALFWorld results).
_SKILL_FALLBACK = os.environ.get("MEMCRYST_SKILL_FALLBACK","0").strip().lower() not in ("","0","false","no")
# Complementary memory filtering (scheme 3): drop retrieved memory entries whose
# text is semantically redundant with the already-retrieved skill block (cosine
# >= threshold), so memory only adds what the skill does not already cover.
# Default off. Threshold via MEMCRYST_COMPL_TAU (default 0.6).
_COMPL_MEM = os.environ.get("MEMCRYST_COMPL_MEM","0").strip().lower() not in ("","0","false","no")
_COMPL_TAU = float(os.environ.get("MEMCRYST_COMPL_TAU","0.6") or "0.6")
# Attribution experiment: force a specific skill (by id) into every task's
# retrieved skill context, so a planted-bad component has a large denominator of
# failures in which it was actually present. Default off (empty).
# NOTE: these two knobs are read LIVE from os.environ at call time (not at import)
# because experiment harnesses set them per-eval inside one process; a module-level
# constant captured at import would never update across evals.
def _force_skill_id() -> str:
    return os.environ.get("MEMCRYST_FORCE_SKILL_ID", "").strip()


def _dormant_skill_ids() -> set:
    return {x.strip() for x in os.environ.get("MEMCRYST_DORMANT_SKILL_IDS", "").split(",") if x.strip()}


def _drop_dormant_skills(skill_results):
    """Remove skill results whose id is in the dormant set (measure-vs-attribute)."""
    dormant = _dormant_skill_ids()
    if not dormant:
        return skill_results
    return [t for t in skill_results if t[0].id not in dormant]

# Stuck-breaker: when an action is invalid after retries, the default fallback
# is `look` — but look does NOT change location, so a fixating weak model sees
# the same observation and re-emits the same invalid action every step until
# max_steps (observed: single invalid take repeated 15-17x). When enabled, fall
# back to a state-CHANGING navigation (an unvisited `go to <recep>`) instead,
# which mutates obs + valid_actions and breaks the death-spiral.
_STUCK_BREAK = os.environ.get("MEMCRYST_STUCK_BREAK", "0").strip().lower() not in ("", "0", "false", "no")

# De-prime: the prompt normally (a) sorts manipulation actions (take/clean/...)
# ABOVE navigation and (b) surfaces a "[PRIORITY: target visible, take now]"
# OVER-take they backfire — priming premature `take` before navigating. When
# enabled, keep the env's natural action order and drop the take-PRIORITY hint.
_DEPRIME = os.environ.get("MEMCRYST_DEPRIME", "0").strip().lower() not in ("", "0", "false", "no")

# Prompt-engineering controls (isolate retrieval value from generic prompt effects):
#   ALL_SKILLS  - inject EVERY (non-dormant) skill for every task, no retrieval =
#                 "the strongest static prompt distillable from our bank". If
#                 top-k retrieval doesn't beat this, instance-conditioning is
#                 worthless at this library size.
#   RANDOM_RETRIEVAL - inject k random-family skills + random memories (token-
#                 matched placebo). If real retrieval ~= random, the benefit is
#                 generic scaffolding, not content relevance.
_ALL_SKILLS = os.environ.get("MEMCRYST_ALL_SKILLS", "0").strip().lower() not in ("", "0", "false", "no")
_RANDOM_RETRIEVAL = os.environ.get("MEMCRYST_RANDOM_RETRIEVAL", "0").strip().lower() not in ("", "0", "false", "no")

# Full per-family injection gate: comma-separated families measured to have NO
# competence gap (Δ<=0). Tasks of these families get NO injected context at all
# (skill AND memory) — fixes the leak where dormant skills still left raw-memory
# injection dragging no-gap families (dual-examine 57 vs capability-examine 100).
_GATED_FAMILIES = {f.strip() for f in os.environ.get("MEMCRYST_GATED_FAMILIES", "").split(",") if f.strip()}


def set_gated_families(fams) -> None:
    """Update the per-family injection gate at runtime. Used by the per-cycle
    orchestrator gap-gate so each sleep's measured decision governs the next
    cycle's wake/eval retrieval (env var alone is static per-process)."""
    global _GATED_FAMILIES
    _GATED_FAMILIES = {str(f).strip() for f in fams if str(f).strip()}

# Tier-1 of two-tier crystallization: a single GENERAL meta-skill (cross-family
# generic tactics distilled from the bank) injected for EVERY task, low-authority
# advisory. Bypasses the family gate (tier-2 specifics are what the gate controls).
_GENERAL_SKILL_PATH = os.environ.get("MEMCRYST_GENERAL_SKILL", "").strip()
_GENERAL_TEXT = ""
if _GENERAL_SKILL_PATH and os.path.isfile(_GENERAL_SKILL_PATH):
    _GENERAL_TEXT = open(_GENERAL_SKILL_PATH).read().strip()
# Families listed here do not receive the general block.
_GENERAL_GATED_FAMILIES = {f.strip() for f in os.environ.get("MEMCRYST_GENERAL_GATED_FAMILIES", "").split(",") if f.strip()}


def _build_action_regex(valid: list[str]) -> str:
    """Regex for vLLM guided_regex: a free-form Thought line followed by an
    Action that is EXACTLY one of the valid actions. Unlike guided_choice this
    preserves the ReAct reasoning step (guided_choice dropped it -> 0% success).
    """
    alts = "|".join(re.escape(a) for a in valid) or "look"
    return r"Thought: [^\n]{1,200}\nAction: (?:" + alts + r")"

_RECEP_AFTER_PREP_RE = re.compile(
    r"\b(?:from|in/on|into|onto|in|on|to|at)\s+(.+)$", re.IGNORECASE
)
_OPEN_CLOSE_RE = re.compile(r"^(?:open|close|go\s+to)\s+(.+)$", re.IGNORECASE)


def _redirect_to_goto(action: str, valid: list[str]) -> Optional[str]:
    """Map an invalid manipulation action to the navigation it presupposes.

    `take pillow 2 from armchair 1` (invalid, agent not there) -> `go to
    armchair 1` when that is a valid action. Breaks the take-before-navigate
    fixation without spending another LLM call. Returns None if no sensible
    redirect exists.
    """
    if not action:
        return None
    a = action.lower().strip()
    verb = a.split()[0] if a.split() else ""
    if verb not in ("take", "put", "move", "open", "close", "use",
                    "clean", "heat", "cool", "examine"):
        return None
    m = _RECEP_AFTER_PREP_RE.search(a) or _OPEN_CLOSE_RE.match(a)
    recep = m.group(1).strip() if m else ""
    if not recep:
        return None
    valid_lower = {v.lower(): v for v in valid}
    if f"go to {recep}" in valid_lower:
        return valid_lower[f"go to {recep}"]
    # receptacle string may carry extra tokens; fall back to fuzzy contains
    for vl, v in valid_lower.items():
        if vl.startswith("go to ") and recep in vl:
            return v
    return None


def _best_action_match(candidate: str, valid: list[str]) -> Optional[str]:
    if not candidate:
        return None
    cand = candidate.lower().strip()

    # Skill literals often write `put <obj> in/on <recep>` to cover both
    # container and surface receptacles. ALFWorld emits one or the other,
    # never both. Expand the slash before matching.
    cand_variants = {cand}
    if "in/on" in cand:
        cand_variants.add(cand.replace("in/on", "in"))
        cand_variants.add(cand.replace("in/on", "on"))

    # ALFWorld 0.4 uses `move <obj> to <recep>` rather than `put <obj> in <recep>`.
    # When the LLM emits a put (from the skill procedure literal), translate
    # to the canonical move form for matching.
    put_m = _PUT_TO_MOVE_RE.match(cand)
    if put_m:
        obj, recep = put_m.group(1).strip(), put_m.group(2).strip()
        cand_variants.add(f"move {obj} to {recep}")

    # Exact match
    for a in valid:
        if a.lower().strip() in cand_variants:
            return a
    # Substring match where the candidate fully contains a valid action.
    # Use the longest valid action that fits to avoid matching a short
    # prefix like `take` when the candidate is `take egg`.
    candidates_substring = []
    for a in valid:
        a_lower = a.lower().strip()
        for c in cand_variants:
            if a_lower in c:
                candidates_substring.append(a)
                break
    if candidates_substring:
        return max(candidates_substring, key=lambda a: len(a))
    # Verb-anchored fuzzy match. Require that the first token (the action
    # verb) matches, then take the valid action with the longest token
    # overlap with the candidate.
    cand_tokens = cand.split()
    if not cand_tokens:
        return None
    cand_verb = cand_tokens[0]
    cand_set = set(cand_tokens)
    best: Optional[str] = None
    best_overlap = 0
    for a in valid:
        a_tokens = a.lower().strip().split()
        if not a_tokens or a_tokens[0] != cand_verb:
            continue
        overlap = len(set(a_tokens) & cand_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best = a
    return best


@dataclass
class _StepRecord:
    obs: str
    action: str
    reward: float
    done: bool


class ReactAgent:
    """ReAct loop over ALFWorld with optional skill retrieval."""

    def __init__(
        self,
        env: AlfworldEnv,
        llm: LLMClient,
        skill_retriever: Optional[SkillRetriever] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        condensed_retriever: Optional[CondensedMemoryRetriever] = None,
        semantic_retriever=None,
        mode: Mode = "vanilla",
        top_k_skills: int = 2,
        top_k_memory: int = 3,
        history_window: int = 5,
        max_steps: int = 50,
        max_invalid_retries: int = 1,
        step_callback=None,
    ) -> None:
        self.env = env
        self.llm = llm
        self.skill_retriever = skill_retriever
        self.memory_retriever = memory_retriever
        self.condensed_retriever = condensed_retriever
        self.semantic_retriever = semantic_retriever
        self.mode = mode
        self.top_k_skills = top_k_skills
        self.top_k_memory = top_k_memory
        self.history_window = history_window
        self.max_steps = max_steps
        self.max_invalid_retries = max_invalid_retries
        self.step_callback = step_callback
        if mode in ("similarity", "capability", "capability_plus_memory", "capability_plus_condense", "capability_plus_semantic") and skill_retriever is None:
            raise ValueError(f"mode={mode} requires a skill_retriever")
        if mode in ("memory", "capability_plus_memory") and memory_retriever is None:
            raise ValueError(f"mode={mode} requires a memory_retriever")
        if mode in ("memory_condense", "capability_plus_condense") and condensed_retriever is None:
            raise ValueError(f"mode={mode} requires a condensed_retriever")
        if mode == "capability_plus_semantic" and semantic_retriever is None:
            raise ValueError(f"mode={mode} requires a semantic_retriever")

    def _cat_llm_fn(self):
        """A prompt->str callable over this agent's LLM, for LLM category classification."""
        def _fn(prompt: str) -> str:
            return self.llm.chat([{"role": "user", "content": prompt}],
                                 max_tokens=8, temperature=0.0)
        return _fn

    def _control_skill_results(self, task_description: str):
        """Prompt-engineering controls; returns override skill triples or None."""
        if not (_ALL_SKILLS or _RANDOM_RETRIEVAL) or self.skill_retriever is None:
            return None
        skills = [s for s in self.skill_retriever.store if not getattr(s, "dormant", False)]
        if not skills:
            return []
        if _ALL_SKILLS:
            return [(s, 1.0, {"path": "all_static"}) for s in skills]
        import random as _rnd
        rng = _rnd.Random(hash(task_description) & 0xFFFF)
        pick = rng.sample(skills, min(self.top_k_skills, len(skills)))
        return [(s, 1.0, {"path": "random_placebo"}) for s in pick]

    def _retrieve_context(self, task_description: str, gate_desc: str = None) -> tuple[list[str], list[str], list[str], str]:
        blocks, ids, paths, title = self._retrieve_context_inner(task_description, gate_desc)
        _gd = gate_desc if gate_desc is not None else task_description
        if _GENERAL_TEXT and not (
            _GENERAL_GATED_FAMILIES
            and family_from_task_description(_gd) in _GENERAL_GATED_FAMILIES
        ):
            gen = ("## General strategies (distilled from past experience; advisory — "
                   "apply only when a trigger matches)\n\n" + _GENERAL_TEXT)
            blocks = [gen] + blocks
            ids = ["general"] + ids
            paths = ["general"] + paths
            if not title:
                title = "# Retrieved context (general strategies)"
        return blocks, ids, paths, title

    def _retrieve_context_inner(self, task_description: str, gate_desc: str = None) -> tuple[list[str], list[str], list[str], str]:
        if _GATED_FAMILIES and family_from_task_description(gate_desc if gate_desc is not None else task_description) in _GATED_FAMILIES:
            return [], [], [], ""  # no-gap family: inject nothing (vanilla behavior)
        """Returns (formatted_blocks, item_ids, paths, block_title) for the
        prompt. Block content depends on mode:

        - vanilla   : empty
        - memory    : top-K raw memory entries via cosine
        - similarity: top-K skills via cosine
        - capability: top-K skills via capability index (no fallback to cosine)
        """
        if self.mode in ("vanilla", "react"):   # "react" = paper-table alias of vanilla
            return [], [], [], ""
        if self.mode == "memory":
            results = self.memory_retriever.retrieve(task_description, top_k=self.top_k_memory)
            blocks = [_format_memory_entry(e, i + 1) for i, (e, _) in enumerate(results)]
            ids = [e.id for e, _ in results]
            return blocks, ids, ["memory_cosine"] * len(results), (
                "# Related past experiences (raw memory entries)"
            )
        if self.mode == "memory_condense":
            results = self.condensed_retriever.retrieve(task_description, top_k=self.top_k_memory)
            blocks = [_format_condensed_entry(e, i + 1) for i, (e, _) in enumerate(results)]
            ids = [e.id for e, _ in results]
            return blocks, ids, ["condense_cosine"] * len(results), (
                "# Related past experience clusters (consolidated summaries)"
            )
        if self.mode == "similarity":
            results = self.skill_retriever.retrieve(task_description, top_k=self.top_k_skills)
            blocks = [_format_skill(s, i + 1) for i, (s, _) in enumerate(results)]
            ids = [s.id for s, _ in results]
            return blocks, ids, ["similarity"] * len(results), (
                "# Available skill (follow the procedure step by step)"
            )
        if self.mode == "capability":
            _ctrl = self._control_skill_results(task_description)
            results = _ctrl if _ctrl is not None else self.skill_retriever.retrieve_by_capability(
                task_description,
                top_k=self.top_k_skills,
                rerank_embeddings=True,
                fallback_similarity=_SKILL_FALLBACK,
            )
            results = _drop_dormant_skills(results)
            blocks = [_format_skill(s, i + 1) for i, (s, _, _) in enumerate(results)]
            ids = [s.id for s, _, _ in results]
            paths = [r[2]["path"] for r in results]
            return blocks, ids, paths, "# Available skill (follow the procedure step by step)"

        if self.mode == "capability_plus_semantic":
            # SKILL (procedure / HOW) + SEMANTIC MEMORY (declarative grounding /
            # WHERE). Orthogonal axes: the skill's abstract find(<target>) step is
            # grounded by the semantic facts for the entities in this task. Skill
            # carries no locations; memory carries no procedure.
            skill_results = self.skill_retriever.retrieve_by_capability(
                task_description, top_k=self.top_k_skills,
                rerank_embeddings=True, fallback_similarity=_SKILL_FALLBACK)
            skill_results = _drop_dormant_skills(skill_results)
            # P1b fact-type routing: MEMCRYST_SEMANTIC_TYPES=affordance,location
            # restricts which fact types are injected (e.g. affordance-only kills the
            # low-support location facts (MEMCRYST_SEMANTIC_MIN_SUPPORT).
            _types_env = os.environ.get("MEMCRYST_SEMANTIC_TYPES", "").strip()
            _allowed = set(t.strip() for t in _types_env.split(",") if t.strip()) or None
            _min_sup = int(os.environ.get("MEMCRYST_SEMANTIC_MIN_SUPPORT", "1"))
            # Optional per-family gate on location facts.
            _locgate = os.environ.get("MEMCRYST_LOCATION_GATE_FAMILIES", "").strip()
            if _locgate:
                _gated_fams = {f.strip() for f in _locgate.split(",") if f.strip()}
                _fam = family_from_task_description(gate_desc if gate_desc is not None else task_description)
                if _fam in _gated_fams:
                    base = _allowed if _allowed is not None else {"affordance", "location", "other"}
                    _allowed = base - {"location"}
            # ① ENTITY-PROFILE retrieval (MEMCRYST_SEMANTIC_PROFILE=1): retrieve by
            # the task's entities + family appliance instead of flat cosine top-k, so
            # the target's locations + the appliance's affordance always surface.
            if os.environ.get("MEMCRYST_SEMANTIC_PROFILE", "0").strip().lower() not in ("", "0", "false", "no"):
                _pfam = family_from_task_description(gate_desc if gate_desc is not None else task_description)
                facts = self.semantic_retriever.retrieve_profiles(
                    task_description, family=_pfam, top_k=max(self.top_k_memory, 6),
                    allowed_types=_allowed, min_support_location=_min_sup)
            else:
                # A: pass the active skills so facts that empirically grounded them
                # rank up (skill→memory edge, read-only).
                _boost = ([s.id for s, _, _ in skill_results]
                          if _SKILL_COND_RETRIEVAL else None)
                facts = self.semantic_retriever.retrieve(
                    task_description, top_k=self.top_k_memory,
                    allowed_types=_allowed, min_support_location=_min_sup,
                    boost_skills=_boost, skill_boost=_SKILL_COND_BOOST,
                    quarantine=_FACT_QUARANTINE,
                    quarantine_penalty=_FACT_QUAR_PENALTY,
                    quarantine_maxv=_FACT_QUAR_MAXV,
                    quarantine_category=_FACT_QUAR_CATEGORY)
            skill_blocks = [_format_skill(s, i + 1) for i, (s, _, _) in enumerate(skill_results)]
            ids = [s.id for s, _, _ in skill_results] + [f.id for f in facts]
            paths = [r[2]["path"] for r in skill_results] + ["semantic"] * len(facts)
            # DIAGNOSTIC: stash retrieved LOCATION facts (the testable ones) with their
            # entity + predicted receptacle-types, so the run loop can score whether the
            # env confirms each fact's claim this episode. Pure recording, no behavior.
            if _FACT_VERIFY:
                from ..memory_bank.semantic import receptacle_types_in, receptacle_categories_in, receptacle_categories_llm
                _td = (task_description or "").lower()
                _catfn = ((lambda t: receptacle_categories_llm(t, self._cat_llm_fn()))
                          if _FACT_CAT_LLM else receptacle_categories_in)
                self._vfacts = [
                    (f, f.entity.lower(), receptacle_types_in(f.fact),
                     _catfn(f.fact), f.entity.lower() in _td)
                    for f in facts
                    if f.ftype == "location" and receptacle_types_in(f.fact)
                ]
            blocks, tparts = [], []
            if skill_blocks:
                blocks.append("## Procedure to follow (HOW — the step order; fill in WHERE from below + the room)\n\n"
                              + "\n\n".join(skill_blocks))
                tparts.append("skill")
            aff = [f for f in facts if f.ftype == "affordance"]
            loc = [f for f in facts if f.ftype != "affordance"]
            if aff:
                blocks.append(
                    "## Constraints — rules about how things work (RESPECT THESE)\n"
                    "These are reliable facts about how appliances/containers behave. Violating them causes failure.\n\n"
                    + self.semantic_retriever.format_block(aff))
                tparts.append("afford")
            if loc:
                blocks.append(
                    "## Where to look — location priors (HINTS, verify against the room)\n"
                    "Where these object types are usually found. Use to decide where to search first, "
                    "then trust what THIS room actually shows over the prior.\n\n"
                    + self.semantic_retriever.format_block(loc))
                tparts.append("loc")
            # Unified dual-layer: gated + de-instantiated RAW episodic block.
            # Channel-isolated: raw entry ids are NOT added to `ids`, so the
            # confidence feedback loop (record_outcome) never sees the raw layer.
            if _RAW_LAYER and self.memory_retriever is not None:
                try:
                    raw_hits = self.memory_retriever.retrieve(
                        task_description, top_k=_RAW_TOPK * 3)
                except Exception:
                    raw_hits = []
                gated = [(e, s) for e, s in raw_hits if s >= _RAW_GATE_TAU][:_RAW_TOPK]
                # MEMCRYST_RAW_RECENCY=N: recency prior — the N most recently
                # stored verbatim excerpts ride along regardless of cosine
                # (LoCoMo questions cluster around the latest session; the
                # June open-book probe showed raw-text-in-context lifts even
                # adversarial calibration, 0.35 vs 0.076). Memory policy, not
                # a leak: entries all come from the agent's own store.
                _rec_n = int(os.environ.get("MEMCRYST_RAW_RECENCY", "0") or 0)
                if _rec_n > 0:
                    try:
                        from memcryst.envscope import env_ok as _env_ok
                        _have = {e.id for e, _ in gated}
                        _exc = [e for e in self.memory_retriever.store
                                if getattr(e, "category", "") == "session_excerpt"
                                and e.id not in _have
                                and _env_ok(getattr(e, "env", ""))]
                        _exc.sort(key=lambda e: -getattr(e, "created_at", 0.0))
                        gated = [(e, 1.0) for e in _exc[:_rec_n]] + gated
                    except Exception:
                        pass
                if gated:
                    body = "\n\n".join(_format_memory_entry(e, i + 1)
                                       for i, (e, _) in enumerate(gated))
                    if _RAW_DEINST:
                        body = _deinstantiate(body)
                    blocks.append(
                        "## Reference episodes (from PAST situations — the procedure "
                        "may transfer; specific places/objects WILL differ here. "
                        "Trust the current observations over these.)\n\n" + body)
                    tparts.append("raw")
            title = "# Retrieved context (" + "+".join(tparts) + ")" if tparts else ""
            return blocks, ids, paths, title

        if self.mode == "capability_plus_condense":
            # conditional skills + condensed memory summaries
            skill_results = self.skill_retriever.retrieve_by_capability(
                task_description, top_k=self.top_k_skills,
                rerank_embeddings=True, fallback_similarity=_SKILL_FALLBACK)
            cond_results = self.condensed_retriever.retrieve(
                task_description, top_k=self.top_k_memory)
            skill_blocks = [_format_skill(s, i + 1) for i, (s, _, _) in enumerate(skill_results)]
            cond_blocks = [_format_condensed_entry(e, i + 1) for i, (e, _) in enumerate(cond_results)]
            ids = [s.id for s, _, _ in skill_results] + [e.id for e, _ in cond_results]
            paths = [r[2]["path"] for r in skill_results] + ["condense_cosine"] * len(cond_results)
            blocks, tparts = [], []
            if skill_blocks:
                blocks.append("## Available skill (follow the procedure step by step)\n\n" + "\n\n".join(skill_blocks))
                tparts.append("skill")
            if cond_blocks:
                blocks.append("## Related past experience clusters (consolidated summaries)\n\n" + "\n\n".join(cond_blocks))
                tparts.append("condense")
            title = "# Retrieved context (" + "+".join(tparts) + ")" if tparts else ""
            return blocks, ids, paths, title

        # capability_plus_memory: paper Eq 5. Agent sees BOTH skill block and
        # memory block in the same prompt. Skill block goes first because the
        # procedure should drive action and memory provides edge-case context.
        _ctrl = self._control_skill_results(task_description)
        skill_results = _ctrl if _ctrl is not None else self.skill_retriever.retrieve_by_capability(
            task_description,
            top_k=self.top_k_skills,
            rerank_embeddings=True,
            fallback_similarity=_SKILL_FALLBACK,
        )
        skill_results = _drop_dormant_skills(skill_results)
        # Attribution experiment: force a specific skill into context on every
        # task (large denominator for the planted-bad component). Prepend if absent.
        _force_id = _force_skill_id()
        if _force_id and getattr(self, "skill_retriever", None) is not None:
            present = {s.id for s, _, _ in skill_results}
            if _force_id not in present:
                try:
                    fs = self.skill_retriever.store.get(_force_id)
                    if fs is not None:
                        skill_results = [(fs, 1.0, {"path": "force_inject", "signature": None})] + list(skill_results)
                except Exception:
                    pass
        if hasattr(self.memory_retriever, "pair_partner_skill_ids"):
            self.memory_retriever.pair_partner_skill_ids = [s.id for s, _, _ in skill_results]
        if _RANDOM_RETRIEVAL:
            import random as _rnd
            _entries = [e for e in self.memory_retriever.store
                        if str(getattr(e, "status", "active")).lower().endswith("active")]
            _rng = _rnd.Random(hash(task_description) & 0xFFFF)
            mem_results = [(e, 0.0) for e in _rng.sample(_entries, min(self.top_k_memory, len(_entries)))] if _entries else []
        else:
            mem_results = self.memory_retriever.retrieve(
                task_description, top_k=self.top_k_memory
            )
        if _COMPL_MEM and skill_results and mem_results and self.memory_retriever is not None:
            try:
                import numpy as _np
                skill_txt = " ".join((getattr(s, "body", "") or "") + " " + (getattr(s, "description", "") or "")
                                     for s, _, _ in skill_results)
                emb = self.memory_retriever.embed
                sv = emb(skill_txt); sv = sv / (_np.linalg.norm(sv) + 1e-9)
                kept = []
                for e, sc in mem_results:
                    mt = (e.improvement or e.failure_reason or e.question or "")
                    mv = emb(mt); mv = mv / (_np.linalg.norm(mv) + 1e-9)
                    if float(sv @ mv) < _COMPL_TAU:   # keep only NON-redundant memory
                        kept.append((e, sc))
                mem_results = kept
            except Exception:
                pass
        skill_blocks = [_format_skill(s, i + 1) for i, (s, _, _) in enumerate(skill_results)]
        mem_blocks = [_format_memory_entry(e, i + 1) for i, (e, _) in enumerate(mem_results)]

        ids: list[str] = [s.id for s, _, _ in skill_results] + [e.id for e, _ in mem_results]
        paths: list[str] = [r[2]["path"] for r in skill_results] + [
            "memory_cosine"
        ] * len(mem_results)

        blocks: list[str] = []
        title_parts: list[str] = []
        # Dual-framing (MEMCRYST_DUAL_FRAMING=1): give the agent an explicit ROLE
        # for skill vs memory so it does not literally replay past trajectories
        # (whose objects/receptacles are from DIFFERENT tasks). Skill = authoritative
        # procedure; memory = non-replayable supplementary hints only.
        _framing = os.environ.get("MEMCRYST_DUAL_FRAMING", "0").strip().lower() not in ("", "0", "false", "no")
        if skill_blocks:
            hdr = ("## Your procedure — FOLLOW THIS (authoritative)\n\n" if _framing
                   else "## Available skill (follow the procedure step by step)\n\n")
            blocks.append(hdr + "\n\n".join(skill_blocks))
            title_parts.append("skill")
        if mem_blocks:
            if _framing:
                hdr = ("## Supplementary hints — NOT instructions\n"
                       "These are notes from DIFFERENT past tasks. Their specific objects/receptacles do NOT exist "
                       "in the current room. Follow the procedure above; use these ONLY as rough hints about where "
                       "object TYPES are usually found and mistakes to avoid. NEVER copy their specific actions, "
                       "object ids, or locations — adapt to what THIS room actually shows.\n\n")
            else:
                hdr = "## Related past experiences (raw memory entries)\n\n"
            blocks.append(hdr + "\n\n".join(mem_blocks))
            title_parts.append("memory")
        title = "# Retrieved context (" + "+".join(title_parts) + ")" if title_parts else ""
        return blocks, ids, paths, title

    def _build_prompt(
        self,
        task_description: str,
        context_blocks: list[str],
        context_title: str,
        recent: list[_StepRecord],
        current_obs: str,
        valid_actions: list[str],
        visited: dict[str, str],
        all_receptacles: list[str],
    ) -> list[dict]:
        context_block = ""
        if context_blocks:
            context_block = f"\n\n{context_title}\n" + "\n\n".join(context_blocks)
        # places_visited tracking is essential for ALL modes. We restore the
        # verbose headers because compact headers ("# Already checked") were
        # revisit". With verbose headers + explicit instruction the agent
        # follows the constraint.
        visited_block = ""
        unvisited_block = ""
        if visited:
            visited_lines = []
            for loc, obs in visited.items():
                contents = _extract_contents(obs)
                visited_lines.append(f"- {loc}: {contents}")
            visited_block = (
                "\n\n# Locations you have already checked (do NOT revisit unless retrieving)\n"
                + "\n".join(visited_lines)
            )
        if all_receptacles:
            unvisited = [r for r in all_receptacles if r not in visited]
            if unvisited:
                unvisited_block = (
                    "\n\n# Locations you have NOT yet checked (try these for the target)\n"
                    + ", ".join(unvisited[:25])
                )
        history_block = ""
        if recent:
            history_lines = []
            for r in recent[-self.history_window:]:
                history_lines.append(f"obs: {r.obs[:200]}")
                history_lines.append(f"action: {r.action}")
            history_block = "\n\n# Recent history\n" + "\n".join(history_lines)
        # Reorder valid actions so manipulation primitives (take / clean / heat /
        # cool / use / put) appear above navigation (go to). This counteracts the
        # obs but keeps emitting `go to <X>` because go-to fills the action
        # list and is read first.
        # _DEPRIME keeps the env's natural order (no manipulation-first reorder)
        # to avoid priming weak models toward premature `take`.
        sorted_valid = list(valid_actions) if _DEPRIME else sorted(valid_actions, key=_action_priority)
        # If the current observation already mentions an object as part of a
        # take action that is in valid_actions, surface that take at the very
        # top with a hint so the agent cannot miss it. Skipped under _DEPRIME.
        primary_take: Optional[str] = None
        if not _DEPRIME:
            obs_lower = (current_obs or "").lower()
            for a in sorted_valid:
                if a.lower().startswith("take ") and a.lower().split(" from ")[0].replace("take ", "").strip() in obs_lower:
                    primary_take = a
                    break
        valid_show = sorted_valid[:30]
        lines = []
        if primary_take is not None:
            lines.append(f"- {primary_take}  [PRIORITY: target visible, take now]")
            for a in valid_show:
                if a == primary_take:
                    continue
                lines.append(f"- {a}")
        else:
            for a in valid_show:
                lines.append(f"- {a}")
        valid_block = "\n".join(lines)
        if len(sorted_valid) > len(valid_show):
            valid_block += f"\n- ... and {len(sorted_valid) - len(valid_show)} more"
        nav_rule = ""
        if _NAV_HINT:
            nav_rule = (
                "# Rule\n"
                "You can only take / put / clean / heat / cool / use an object at a "
                "receptacle you are CURRENTLY at. If the action you want is not in the "
                "Valid actions list above, you are not there yet — `go to <receptacle>` "
                "first, then act.\n\n"
            )
        steps_left_block = ""
        user = (
            f"# Task\n{task_description}"
            f"{context_block}"
            f"{visited_block}"
            f"{unvisited_block}"
            f"{history_block}"
            f"{steps_left_block}\n\n"
            f"# Current observation\n{current_obs}\n\n"
            f"# Valid actions\n{valid_block}\n\n"
            f"{nav_rule}"
            + (
                # taskstream = free-form solving
                "# Your response format\n"
                "Free form. Reason step by step as deeply as needed, then follow "
                "the answer format asked for in the observation."
                if _ENV_KIND == "taskstream" else
                "# Your response format\n"
                "Thought: <one sentence about the next subgoal>\n"
                "Action: <exactly one action copied verbatim from the valid actions list>"
            )
        )
        sys_prompt = _TASKSTREAM_SYS_PROMPT if _ENV_KIND == "taskstream" else _SYS_PROMPT
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]

    def run(self, task_idx: int, reset_result=None) -> TrajectoryResult:
        # reset_result: reuse an already-performed env.reset(). ALFWorld resets are
        # sequential, so resetting twice for one task would play the next game.
        reset = reset_result if reset_result is not None else self.env.reset(task_idx=task_idx)
        self._vfacts = []   # DIAGNOSTIC: retrieved location facts to verify this episode
        context_blocks, used_ids, paths, context_title = self._retrieve_context(
            reset.task_description
        )
        retrieval_paths = Counter(paths)

        # Split skill vs memory ids using the path tag. Memory retrievals use
        # path "memory_cosine" or "condense_cosine"; skill retrievals use
        # "exact"/"relaxed_*"/"similarity"/"fallback_*".
        _MEM_PATHS = ("memory_cosine", "condense_cosine", "semantic")
        skill_ids_split = [i for i, p in zip(used_ids, paths) if p not in _MEM_PATHS]
        memory_ids_split = [i for i, p in zip(used_ids, paths) if p in _MEM_PATHS]

        result = TrajectoryResult(
            task_idx=task_idx,
            task_description=reset.task_description,
            mode=self.mode,
            success=False,
            steps=0,
            final_reward=0.0,
            used_skill_ids=skill_ids_split,
            used_memory_ids=memory_ids_split,
            retrieval_paths=retrieval_paths,
            initial_obs=reset.obs or "",
        )

        if (os.environ.get("MEMCRYST_QA_ANSCALL", "0") == "1"
                and getattr(self.env, "current_kind", "") == "question"):
            _date = getattr(self.env, "_last_session_date", "") or "unknown"
            _mem = "\n\n".join(context_blocks) if context_blocks else "(no memory yet)"
            _qa_prompt = (
                "You answer questions about people from your accumulated memory "
                f"of their conversations. Today is {_date}.\n\n"
                "## Answer style (follow these formats exactly)\n"
                "- \"When did X attend the workshop?\" -> 7 May 2023\n"
                "- If a session dated 27 June 2023 says something happened \"last week\" "
                "-> The week before 27 June 2023\n"
                "- \"What did X cook for the party?\" -> spaghetti with meatballs\n"
                "- \"How long did X's trip last?\" -> two weeks\n\n"
                f"{_mem}\n\n"
                f"Question: {reset.task_description}\n"
                "Answer with ONLY a short phrase (a few words, a date, or a name). "
                "Give your best short answer whenever ANY fact or excerpt relates "
                "to the question — only if absolutely nothing relates, answer: "
                "No information available.\nAnswer:")
            try:
                _ans = self.llm.chat([{"role": "user", "content": _qa_prompt}],
                                     max_tokens=48, temperature=0.0).strip()
                result.llm_calls += 1
            except Exception as exc:
                result.error = f"QA answer call failed: {exc!r}"
                return result
            _ans = _ans.splitlines()[0].strip() if _ans else ""
            step = self.env.step(f"Answer: {_ans}")
            result.step_trace.append(StepTrace(
                step_i=0, thought="", action=f"Answer: {_ans}", obs=step.obs,
                reward=step.reward, done=True))
            result.steps = 1
            result.final_reward = step.reward
            result.success = step.success
            return result

        recent: list[_StepRecord] = []
        current_obs = reset.obs
        valid_actions = reset.valid_actions or ["look"]
        visited: dict[str, str] = {}
        visit_count: dict[str, int] = {}   # WM: how many times each receptacle was entered
        seen_at: dict[str, set] = {}       # DIAGNOSTIC: entity -> receptacle-types observed at
        find_goto: dict[str, int] = {}     # DIAGNOSTIC: entity -> go-to# when first observed
        goto_count = 0                     # total go-tos so far (search cost denominator)
        all_receptacles = _extract_initial_receptacles(reset.obs)
        holding = False   # working-memory: is the agent carrying an object (search done)?

        for step_i in range(self.max_steps):
            # ② STATE-CONDITIONED (in-chain) retrieval: every _INCHAIN steps, re-retrieve
            # skill/facts using the PARTIAL TRAJECTORY as query so the injected guidance
            # tracks the current phase (gate/family stays on the original task).
            if _INCHAIN and step_i > 0 and step_i % _INCHAIN == 0:
                try:
                    _prog = " ".join(r.action for r in recent[-6:])
                    _q = reset.task_description + " | progress: " + _prog + " | now: " + current_obs[:200]
                    context_blocks, _ids2, _p2, context_title = self._retrieve_context(
                        _q, gate_desc=reset.task_description)
                except Exception:
                    pass
            messages = self._build_prompt(
                reset.task_description,
                context_blocks,
                context_title,
                recent,
                current_obs,
                valid_actions,
                visited,
                all_receptacles,
            )
            try:
                if _ACTION_GRAMMAR == "guided":
                    # vLLM guided_choice forces the output to be exactly one of
                    # the valid actions — no Thought, but fixation is impossible.
                    raw = self.llm.chat(
                        messages, max_tokens=40, temperature=0.0,
                        extra_body={"guided_choice": list(valid_actions)},
                    )
                elif _ACTION_GRAMMAR == "guided_regex":
                    # Keep the Thought, constrain only the Action to a valid one.
                    raw = self.llm.chat(
                        messages, max_tokens=220, temperature=0.0,
                        extra_body={"guided_regex": _build_action_regex(valid_actions)},
                    )
                else:
                    # taskstream envs need full solutions (reasoning + code + answer),
                    # not one-line actions — 200 tokens truncated every math solve
                    _mt = (int(os.environ.get("MEMCRYST_TS_MAXTOKENS", "2400"))
                           if _ENV_KIND == "taskstream" else 200)
                    raw = self.llm.chat(messages, max_tokens=_mt, temperature=0.0)
                result.llm_calls += 1
            except Exception as exc:
                result.error = f"LLM call failed: {exc!r}"
                logger.exception("agent LLM call crashed")
                break
            logger.debug("step %d raw LLM output: %r", step_i, raw)
            if _ACTION_GRAMMAR == "guided":
                thought, action = "", raw.strip().strip("`'\" ")
            else:
                thought, action = _parse_react(raw)
            if _ENV_KIND == "taskstream":
                # free-form env: the WHOLE model output is the action
                action = (raw or "").strip()
            matched = _taskstream_action_passthrough(action) or _best_action_match(action, valid_actions)
            # If matcher returned a result that has nothing to do with the
            # candidate verb (e.g., agent said `put` but we matched a `go to`
            # via prefix tie), treat as no-match so we retry. We allow the
            # known translation `put → move` because ALFWorld's place
            # primitive is `move <obj> to <recep>`.
            if matched is not None and action:
                cand_verb = action.lower().strip().split()[0] if action.strip() else ""
                match_verb = matched.lower().strip().split()[0]
                if cand_verb and match_verb and cand_verb != match_verb:
                    if not (cand_verb == "put" and match_verb == "move"):
                        matched = None

            # redirect mode: convert an invalid manipulation action into the
            # 'go to <recep>' it presupposes, instead of burning a retry call.
            if matched is None and _ACTION_GRAMMAR == "redirect":
                redir = _redirect_to_goto(action, valid_actions)
                if redir is not None:
                    logger.info("step %d, redirect %r -> %r", step_i, action, redir)
                    matched = redir
                    result.invalid_action_retries += 1

            # One retry if action invalid.
            retry = 0
            while matched is None and retry < self.max_invalid_retries:
                retry += 1
                result.invalid_action_retries += 1
                cand_verb = action.lower().strip().split()[0] if action.strip() else ""
                related = [a for a in valid_actions if a.lower().startswith(cand_verb)] if cand_verb else []
                logger.warning(
                    "step %d, raw LLM action %r did not map. Valid actions with verb %r: %r. Retrying.",
                    step_i,
                    action,
                    cand_verb,
                    related[:10],
                )
                hint = (
                    "Your Action was not in the valid actions list. "
                    "Pick ONE action copied verbatim from the list."
                )
                if cand_verb == "put":
                    hint += (
                        " If you intended `put <obj> in <recep>` and it is "
                        "not available, the destination receptacle probably "
                        "contains another object you must clear first. "
                        "Use `take <existing_object> from <destination>` to "
                        "clear it, then retry put."
                    )
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": hint},
                ]
                raw = self.llm.chat(retry_messages, max_tokens=120, temperature=0.0)
                result.llm_calls += 1
                _, action = _parse_react(raw)
                matched = _taskstream_action_passthrough(action) or _best_action_match(action, valid_actions)

            if matched is None:
                # Give up on this step. Default `look` keeps the agent in place,
                # so a fixating model re-emits the same invalid action forever.
                if _STUCK_BREAK:
                    # Force a state change: navigate to an unvisited receptacle
                    # (rotate by step so consecutive give-ups explore different
                    # places). Falls back to look only if no go-to exists.
                    gotos = [a for a in valid_actions if a.lower().startswith("go to ")]
                    unvisited = [g for g in gotos if g.lower()[len("go to "):].strip() not in visited]
                    pool = unvisited or gotos
                    if pool:
                        matched = pool[step_i % len(pool)]
                    else:
                        matched = "look" if "look" in valid_actions else valid_actions[0]
                else:
                    matched = "look" if "look" in valid_actions else valid_actions[0]
                cand_verb = action.lower().strip().split()[0] if action.strip() else ""
                related = [a for a in valid_actions if a.lower().startswith(cand_verb)] if cand_verb else []
                # Also dump the full valid_actions list so we can see what
                # the env actually offers at this state.
                logger.warning(
                    "task %d step %d, LLM action %r not in valid set, defaulting to %s.\n"
                    "  Related (verb=%r): %r\n"
                    "  FULL valid_actions (%d total): %r",
                    task_idx, step_i, action, matched,
                    cand_verb, related[:10],
                    len(valid_actions), valid_actions,
                )

            # ① WORKING-MEMORY ENFORCEMENT: during search (not holding), redirect a
            # revisit of an already-checked receptacle to an unvisited one, so the
            # agent covers the room instead of bouncing (the 21/21 timeout pattern).
            if (_WM_ENFORCE and not holding and matched
                    and matched.lower().startswith("go to ")):
                _tgt = matched.lower()[len("go to "):].strip()
                # Only break a revisit once the target has been entered >= the
                # threshold (default 1 = first revisit). >=2 spares a legitimate
                # single return and only breaks the pathological 3rd+ loop.
                if visit_count.get(_tgt, 0) >= _WM_REVISIT_MIN:
                    _goto = [a for a in valid_actions if a.lower().startswith("go to ")]
                    _unv = [a for a in _goto if a.lower()[len("go to "):].strip() not in visited]
                    if _unv:
                        logger.info("step %d, WM-enforce redirect %r -> %r", step_i, matched, _unv[0])
                        matched = _unv[0]
            step = self.env.step(matched)
            # track holding (search phase ends when an object is picked up)
            _mlh = matched.lower().strip()
            if _mlh.startswith("take "):
                holding = True
            elif _mlh.startswith("put ") or _mlh.startswith("move "):
                holding = False
            recent.append(
                _StepRecord(obs=current_obs, action=matched, reward=step.reward, done=step.done)
            )
            result.step_trace.append(
                StepTrace(
                    step_i=step_i,
                    thought=thought,
                    action=matched,
                    obs=step.obs,
                    reward=step.reward,
                    done=step.done,
                )
            )
            # Track visited locations. Capture the observation after a `go to`
            # or `open` action so the prompt can show "already checked X".
            ml = matched.lower().strip()
            if ml.startswith("go to "):
                loc = ml[len("go to ") :].strip()
                visited[loc] = step.obs
                visit_count[loc] = visit_count.get(loc, 0) + 1
                goto_count += 1
                # DIAGNOSTIC: record where each retrieved fact's entity is observed
                # (receptacle-type we arrived at) + the go-to# it was first found at
                # (search cost). No behavior change.
                if _FACT_VERIFY and self._vfacts:
                    from ..memory_bank.semantic import receptacle_types_in
                    _loc_types = receptacle_types_in(loc)
                    _obs_l = (step.obs or "").lower()
                    for _f, _ent, _pred, _pcat, _istgt in self._vfacts:
                        if _ent and _ent in _obs_l:
                            if _loc_types:
                                seen_at.setdefault(_ent, set()).update(_loc_types)
                            if _ent not in find_goto:
                                find_goto[_ent] = goto_count
            elif ml.startswith("open "):
                loc = ml[len("open ") :].strip()
                # Overwrite, since opening reveals new contents.
                visited[loc] = step.obs
            if self.step_callback is not None:
                try:
                    self.step_callback(step_i, thought, matched, step.obs, step.reward, step.done, step.success)
                except Exception:
                    pass
            current_obs = step.obs
            valid_actions = step.valid_actions or valid_actions
            result.steps = step_i + 1
            result.final_reward = step.reward
            result.final_obs = step.obs
            if step.done:
                result.success = step.success
                break

        if _FACT_VERIFY:
            result.diag = {
                "revisit_extra": sum(max(0, c - 1) for c in visit_count.values()),
                "kn_refuted": 0, "kn_verified": 0,
            }
        if _FACT_VERIFY and self._vfacts:
            from ..memory_bank.semantic import RECEPTACLE_CATEGORY, llm_receptacle_category
            for _f, _ent, _pred, _pcat, _istgt in self._vfacts:
                _obs_types = seen_at.get(_ent)
                if _obs_types:
                    # record where the entity was actually observed (correction target)
                    for _t in _obs_types:
                        _f.seen_types[_t] = _f.seen_types.get(_t, 0) + 1
                    # type-level verdict (original signal)
                    if _obs_types & _pred:
                        _f.verified += 1
                    else:
                        _f.refuted += 1
                    # category-level verdict (coarser — surface/container/appliance)
                    if _FACT_CAT_LLM:
                        _obs_cats = {llm_receptacle_category(t, self._cat_llm_fn())
                                     for t in _obs_types}
                        _obs_cats.discard(None)
                    else:
                        _obs_cats = {RECEPTACLE_CATEGORY[t] for t in _obs_types
                                     if t in RECEPTACLE_CATEGORY}
                    if _pcat and _obs_cats:
                        if _obs_cats & _pcat:
                            _f.cat_verified += 1
                            result.diag["kn_verified"] += 1
                        else:
                            _f.cat_refuted += 1
                            result.diag["kn_refuted"] += 1
                # search-cost (target-conditioned): only when this fact's entity is the
                # task target. Lower go-to# to find it = the grounding usefully narrowed
                # the search. Never found -> full search effort (goto_count).
                if _istgt:
                    _f.find_cost_sum += find_goto.get(_ent, goto_count)
                    _f.find_cost_n += 1

        return result
