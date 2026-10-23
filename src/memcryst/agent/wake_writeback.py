"""Wake-time trajectory writeback and failure attribution.

``trajectory_to_memory_entry`` summarizes a completed trajectory into a
MemoryEntry (canonical question shape, what worked or what the procedure
should have been, failure reason). ``attribute_failure_axis`` decides which
capability axis a failed trajectory should be attributed to; the sleep-time
feedback uses it for axis-level flagging of ancestry entries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..core import LLMClient
from ..memory_bank import MemoryEntry
from ..skill_bank import Skill
from .react_agent import TrajectoryResult

logger = logging.getLogger(__name__)


_WRITEBACK_PROMPT = """An autonomous agent just attempted an ALFWorld household task. Summarize the trajectory into a single memory entry that future agents can learn from.

Task: {task_description}
Outcome: {outcome}
Total steps: {steps}

Trajectory (last {n_steps} steps):
{trajectory}

Output a single JSON object with these fields:
{{
  "question": "<short canonical phrasing of the task, USING PLACEHOLDERS NOT INSTANCES>",
  "improvement": "<one paragraph describing the procedure that worked, or the procedure that should have worked. Use ALFWorld primitives: go to, take X from Y, clean X with sink, heat X with microwave, cool X with fridge, use lamp, move X to Y. Do not include long-form narration.>",
  "failure_reason": "<if the task failed, ONE sentence describing what went wrong. Empty string if succeeded.>",
  "category": "general"
}}

CRITICAL NORMALIZATION RULES for the "question" field:
- Replace specific object names with <target> (e.g., "mug 1", "potato 2", "soapbar 3" → "<target>").
- Replace specific receptacle names with <destination> (e.g., "coffeemachine 1", "drawer 2", "countertop 1" → "<destination>").
- Replace specific appliance names with <appliance> if relevant.
- Use the verb family verbatim (clean, heat, cool, examine, move).
- Examples:
  - "put a clean mug in coffeemachine" → "clean a <target> and place it in a <destination>"
  - "cool potato 2 and put it in fridge" → "cool a <target> and place it in a <destination>"
  - "examine the book under the desklamp" → "examine a <target> using a desklamp"
- DO NOT include the literal instance id (no "mug 1", no "1", no "the red one").
- DO NOT make the question more specific than the verb family + placeholders.

The improvement field MAY contain specific object names from the trajectory (those are the recipe details). Only the question field must be normalized.

Keep all fields concise. The improvement field should read like a recipe future agents can follow. Output only the JSON, no preamble."""


def _writeback_prompt() -> str:
    return _WRITEBACK_PROMPT


_AXIS_ATTRIBUTION_PROMPT = """An agent attempted an ALFWorld task using a crystallized skill, and the task failed. Decide which capability axis of the skill should be flagged for re-crystallization.

Task: {task_description}
Skill used: {skill_name}
Skill verb_family: {verb_family}
Skill affordance: {affordance}
Skill output: {output}
Skill procedure:
{procedure}

What the agent actually did (last {n_steps} steps):
{trajectory}

Final observation: {final_obs}

The four capability axes are:
- verb_family: the agent picked the wrong overall action category (e.g., heated when should have cooled, cleaned when should have moved)
- affordance: the agent used the wrong appliance / receptacle (e.g., went to microwave when should have gone to fridge, looked at countertop when should have looked at drawer)
- target_type: the agent confused the target object with another object (took the wrong item, or never located the right item)
- output: the agent located the correct target AND reached the correct destination AND tried the placement primitive but it never registered as done

CRITICAL DECISION RULES — apply in order:
1. If the agent's actions show it never went to the correct appliance/receptacle, attribute to **affordance**.
2. If the agent's actions show it took or examined the wrong object, attribute to **target_type**.
3. If the agent's verb sequence does not match the task (cleaned instead of heated, moved instead of cooled), attribute to **verb_family**.
4. Only attribute to **output** if the agent visibly performed the correct intermediate actions on the correct objects at the correct receptacle AND only the final placement primitive failed to fire. This is rare. If unsure between output and any of the other three axes, DO NOT pick output.

Most failures are caused by affordance or target_type confusion, not by output. Default away from output.

Pick the SINGLE axis that best explains why the task failed. Output a JSON object with this shape:
{{"axis": "<one of verb_family|affordance|target_type|output>", "reason": "<one short sentence pointing to the specific step or missing action that justifies this axis>"}}

Output only the JSON, no preamble."""


_VALID_AXES = {"verb_family", "affordance", "target_type", "output"}


def _format_trajectory_for_prompt(result: TrajectoryResult, last_n: int = 10) -> str:
    if not result.step_trace:
        return "(no steps executed)"
    relevant = result.step_trace[-last_n:]
    lines = []
    for s in relevant:
        lines.append(f"  step {s.step_i}: thought={s.thought[:100]} | action={s.action} | obs={s.obs[:120]}")
    return "\n".join(lines)


def _parse_first_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
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
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def trajectory_to_memory_entry(
    result: TrajectoryResult,
    llm: LLMClient,
    last_n_steps: int = 12,
) -> Optional[MemoryEntry]:
    """Summarize an agent trajectory into a MemoryEntry.

    Returns None if the LLM call fails or the output cannot be parsed.
    The returned entry has weight scaled by outcome (success > failure).
    """
    outcome = "SUCCESS" if result.success else f"FAILURE (reached {result.steps} of max steps)"
    trajectory = _format_trajectory_for_prompt(result, last_n=last_n_steps)
    prompt = _writeback_prompt().format(
        task_description=result.task_description,
        outcome=outcome,
        steps=result.steps,
        n_steps=min(last_n_steps, len(result.step_trace)),
        trajectory=trajectory,
    )
    try:
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": "You write concise procedural memory entries from agent trajectories. Output one JSON object only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("trajectory_to_memory_entry LLM call failed: %s", exc)
        return None

    data = _parse_first_json(raw)
    if not data:
        logger.warning("trajectory_to_memory_entry could not parse LLM output: %r", raw[:200])
        return None

    question = (data.get("question") or "").strip()
    if not question:
        return None
    improvement = (data.get("improvement") or "").strip()
    failure_reason = (data.get("failure_reason") or "").strip()
    category = (data.get("category") or "general").strip() or "general"

    # MEMCRYST_WB_VERBATIM=1 (QA streams): append a verbatim excerpt of the first
    # observation (the session text) so the raw layer carries NAMES AND DATES —
    # the reflection LLM masks them ("<target>") following the embodied prompt.
    # text_for_embedding includes improvement, so questions match by content.
    import os as _os2
    if (result.success
            and _os2.environ.get("MEMCRYST_WB_VERBATIM", "0") not in ("", "0")):
        _obs0 = (getattr(result, "initial_obs", "") or "")[:1800]
        if _obs0:
            improvement = (improvement + "\n\nVERBATIM EXCERPT:\n" + _obs0).strip()

    # MEMCRYST_STORE_SOLUTION=1 (solve streams): append the final WORKING code
    # to the entry so the raw layer can inject it as a worked example for
    # archetype-similar problems. Problem statements don't repeat but
    # algorithmic archetypes do — this is the one content form the three
    # substrates were missing (facts=trivia, skills=prose, raw=reflections).
    import os as _os
    if result.success and _os.environ.get("MEMCRYST_STORE_SOLUTION", "0") not in ("", "0"):
        for st in reversed(result.step_trace):
            blocks = re.findall(r"```(?:python)?\s*(.*?)```", st.action or "", re.DOTALL)
            if blocks:
                improvement = (improvement + "\n\nWORKED SOLUTION (this exact code "
                               "solved that task):\n```python\n"
                               + blocks[-1].strip()[:1200] + "\n```").strip()
                break

    # Weight encodes the writeback signal strength. Successful entries are
    # boosted (mirrors Live-Evo's positive empirical gap), failures get
    # below-default weight so they do not dominate clustering by mass.
    weight = 1.3 if result.success else 0.7

    return MemoryEntry(
        question=question,
        improvement=improvement,
        failure_reason=failure_reason,
        category=category,
        weight=weight,
    )


def attribute_failure_axis(
    result: TrajectoryResult,
    skill: Skill,
    llm: LLMClient,
    last_n_steps: int = 12,
) -> Optional[str]:
    """Pick the capability axis that best explains a failed trajectory.

    Returns one of the four axis names, or None if the LLM call fails or
    the returned axis is not in the closed set.
    """
    if result.success:
        return None
    if skill is None or not skill.capability.is_complete():
        return None

    procedure = ""
    for line in (skill.body or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## procedure"):
            continue
        if stripped.startswith("## "):
            break
        if stripped:
            procedure += line + "\n"

    trajectory = _format_trajectory_for_prompt(result, last_n=last_n_steps)
    prompt = _AXIS_ATTRIBUTION_PROMPT.format(
        task_description=result.task_description,
        skill_name=skill.name,
        verb_family=skill.capability.verb_family,
        affordance=skill.capability.affordance,
        output=skill.capability.output,
        procedure=procedure.strip()[:600],
        n_steps=min(last_n_steps, len(result.step_trace)),
        trajectory=trajectory,
        final_obs=result.final_obs[:200],
    )

    try:
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": "You attribute agent failures to one capability axis. Output one JSON object only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("attribute_failure_axis LLM call failed: %s", exc)
        return None

    data = _parse_first_json(raw)
    if not data:
        return None
    axis = (data.get("axis") or "").strip().lower()
    if axis not in _VALID_AXES:
        logger.warning("axis_attribution returned invalid axis %r", axis)
        return None
    return axis
