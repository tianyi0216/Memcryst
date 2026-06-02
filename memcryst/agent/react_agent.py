"""ReAct-style agent with three skill-retrieval modes.

Modes:
- vanilla     no skill retrieval at all, raw ReAct over valid actions
- similarity  retrieve top-K skills via SkillRetriever.retrieve() (O(N) cosine)
- capability  retrieve top-K skills via SkillRetriever.retrieve_by_capability() (O(K) hash)

The three modes share the same prompt and parser. Only the skill block in
the prompt context changes. That makes the comparison architecturally clean:
identical agent, identical action space, only the retrieval path differs.

Each run produces a TrajectoryResult with success flag, step count, retrieval
path counts, and the used skill ids so M3 can later attribute failures.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..baselines.condense import CondensedEntry, CondensedMemoryRetriever
from ..core import LLMClient
from ..memory_bank import MemoryEntry, MemoryRetriever
from ..skill_bank import Skill, SkillRetriever
from .alfworld_env import AlfworldEnv, ResetResult, StepResult

logger = logging.getLogger(__name__)


Mode = Literal["vanilla", "memory", "memory_condense", "similarity", "capability"]


@dataclass
class TrajectoryResult:
    task_idx: int
    task_description: str
    mode: Mode
    success: bool
    steps: int
    final_reward: float
    used_skill_ids: list[str] = field(default_factory=list)
    retrieval_paths: Counter = field(default_factory=Counter)
    invalid_action_retries: int = 0
    llm_calls: int = 0
    error: Optional[str] = None


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
    prompt block. memory_condense is the AWM / Live-Evo style baseline:
    NL summary of a cluster, no skill structure.
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
    return (
        f"Skill {idx}: {skill.name}\n"
        f"  description: {skill.description}\n"
        f"{cap_line}"
        f"  when_to_use: {when.strip()[:300]}\n"
        f"  procedure:\n{_indent(procedure.strip()[:600], '    ')}"
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
        self.mode = mode
        self.top_k_skills = top_k_skills
        self.top_k_memory = top_k_memory
        self.history_window = history_window
        self.max_steps = max_steps
        self.max_invalid_retries = max_invalid_retries
        self.step_callback = step_callback
        if mode in ("similarity", "capability") and skill_retriever is None:
            raise ValueError(f"mode={mode} requires a skill_retriever")
        if mode == "memory" and memory_retriever is None:
            raise ValueError(f"mode=memory requires a memory_retriever")
        if mode == "memory_condense" and condensed_retriever is None:
            raise ValueError(f"mode=memory_condense requires a condensed_retriever")

    def _retrieve_context(self, task_description: str) -> tuple[list[str], list[str], list[str], str]:
        """Returns (formatted_blocks, item_ids, paths, block_title) for the
        prompt. Block content depends on mode:

        - vanilla   : empty
        - memory    : top-K raw memory entries via cosine
        - similarity: top-K skills via cosine
        - capability: top-K skills via capability index (no fallback to cosine)
        """
        if self.mode == "vanilla":
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
        # capability
        results = self.skill_retriever.retrieve_by_capability(
            task_description,
            top_k=self.top_k_skills,
            rerank_embeddings=True,
            fallback_similarity=False,
        )
        blocks = [_format_skill(s, i + 1) for i, (s, _, _) in enumerate(results)]
        ids = [s.id for s, _, _ in results]
        paths = [r[2]["path"] for r in results]
        return blocks, ids, paths, "# Available skill (follow the procedure step by step)"

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
        # not strong enough signals for gpt-4o-mini to actually obey "do not
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
        # observed failure mode where gpt-4o-mini sees the target object in
        # obs but keeps emitting `go to <X>` because go-to fills the action
        # list and is read first.
        sorted_valid = sorted(valid_actions, key=_action_priority)
        # If the current observation already mentions an object as part of a
        # take action that is in valid_actions, surface that take at the very
        # top with a hint so the agent cannot miss it.
        primary_take: Optional[str] = None
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
        user = (
            f"# Task\n{task_description}"
            f"{context_block}"
            f"{visited_block}"
            f"{unvisited_block}"
            f"{history_block}\n\n"
            f"# Current observation\n{current_obs}\n\n"
            f"# Valid actions\n{valid_block}\n\n"
            f"# Your response format\n"
            f"Thought: <one sentence about the next subgoal>\n"
            f"Action: <exactly one action copied verbatim from the valid actions list>"
        )
        return [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": user},
        ]

    def run(self, task_idx: int) -> TrajectoryResult:
        reset = self.env.reset(task_idx=task_idx)
        context_blocks, used_ids, paths, context_title = self._retrieve_context(
            reset.task_description
        )
        retrieval_paths = Counter(paths)

        result = TrajectoryResult(
            task_idx=task_idx,
            task_description=reset.task_description,
            mode=self.mode,
            success=False,
            steps=0,
            final_reward=0.0,
            used_skill_ids=used_ids,
            retrieval_paths=retrieval_paths,
        )

        recent: list[_StepRecord] = []
        current_obs = reset.obs
        valid_actions = reset.valid_actions or ["look"]
        visited: dict[str, str] = {}
        all_receptacles = _extract_initial_receptacles(reset.obs)

        for step_i in range(self.max_steps):
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
                raw = self.llm.chat(messages, max_tokens=200, temperature=0.0)
                result.llm_calls += 1
            except Exception as exc:
                result.error = f"LLM call failed: {exc!r}"
                logger.exception("agent LLM call crashed")
                break
            logger.debug("step %d raw LLM output: %r", step_i, raw)
            thought, action = _parse_react(raw)
            matched = _best_action_match(action, valid_actions)
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
                matched = _best_action_match(action, valid_actions)

            if matched is None:
                # Give up on this step and try `look` to keep the loop alive.
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

            step = self.env.step(matched)
            recent.append(
                _StepRecord(obs=current_obs, action=matched, reward=step.reward, done=step.done)
            )
            # Track visited locations. Capture the observation after a `go to`
            # or `open` action so the prompt can show "already checked X".
            ml = matched.lower().strip()
            if ml.startswith("go to "):
                loc = ml[len("go to ") :].strip()
                visited[loc] = step.obs
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
            if step.done:
                result.success = step.success
                break

        return result
