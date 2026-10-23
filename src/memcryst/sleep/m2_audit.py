"""Skill audit gate.

A fresh auditor reads the candidate skill and the source cluster summary, runs
twelve binary checks and returns a structured verdict; a candidate is
accepted when at least nine checks pass. The check definitions live in
``memcryst/prompts/audit_checks.md``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core import LLMClient
from ..memory_bank import MemoryEntry
from ..prompts import load_prompt
from ..skill_bank import Skill
from .m1_miner import ClusterCandidate

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AuditResult:
    decision: str
    pass_count: int
    checks: list[dict[str, Any]] = field(default_factory=list)
    overall_reason: str = ""
    raw_text: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision.upper() == "ACCEPT"


def _format_episode_short(entry: MemoryEntry, idx: int) -> str:
    pieces = [f"  Episode {idx} weight {entry.weight:.2f}"]
    pieces.append(f"    question: {entry.question}")
    if entry.failure_reason:
        pieces.append(f"    failure_reason: {entry.failure_reason}")
    if entry.improvement:
        pieces.append(f"    improvement: {entry.improvement}")
    return "\n".join(pieces)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip().rstrip("`")
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


class SkillAuditor:
    """Audit a candidate Skill against its source cluster."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_name: str = "audit_checks",
        min_pass: int = 10,
        total_checks: int = 13,
    ) -> None:
        self.llm = llm
        import os as _os
        prompt_name = _os.environ.get("MEMCRYST_AUDIT_PROMPT", prompt_name)
        self.prompt_template = load_prompt(prompt_name)
        self.min_pass = min_pass
        self.total_checks = total_checks

    def audit(self, skill: Skill, cluster: ClusterCandidate) -> AuditResult:
        episodes_block = "\n\n".join(
            _format_episode_short(entry, i + 1)
            for i, entry in enumerate(sorted(cluster.members, key=lambda e: e.weight, reverse=True)[:5])
        )

        # Render the candidate skill as full SKILL.md text for grounding.
        skill_md = self._render_skill_md(skill)

        prompt = self.prompt_template.format(
            skill_md=skill_md,
            cluster_size=cluster.size,
            mean_reward=round(cluster.mean_reward, 3),
            category=cluster.category_mode,
            episodes_block=episodes_block,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a conservative fresh agent auditor. You output exactly one JSON object.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, max_tokens=1500, temperature=0.0)

        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("auditor returned non JSON output, rejecting by default")
            return AuditResult(
                decision="REJECT",
                pass_count=0,
                checks=[],
                overall_reason="auditor output could not be parsed as JSON",
                raw_text=raw,
            )

        checks_raw = parsed.get("checks") or []
        checks = []
        pass_count = 0
        for item in checks_raw:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "")).upper().strip()
            if verdict not in {"PASS", "FAIL"}:
                verdict = "FAIL"
            if verdict == "PASS":
                pass_count += 1
            checks.append(
                {
                    "id": item.get("id"),
                    "verdict": verdict,
                    "reason": str(item.get("reason", "")).strip(),
                }
            )

        # Prefer model declared decision if it matches the rule, otherwise enforce.
        declared = str(parsed.get("decision", "")).upper().strip()
        if declared in {"ACCEPT", "REJECT"}:
            enforced = "ACCEPT" if pass_count >= self.min_pass else "REJECT"
            decision = enforced if enforced == declared else enforced
        else:
            decision = "ACCEPT" if pass_count >= self.min_pass else "REJECT"

        return AuditResult(
            decision=decision,
            pass_count=pass_count,
            checks=checks,
            overall_reason=str(parsed.get("overall_reason", "")).strip(),
            raw_text=raw,
        )

    @staticmethod
    def _render_skill_md(skill: Skill) -> str:
        meta = skill.to_frontmatter_dict()
        frontmatter_lines = ["---"]
        for key in ("name", "description", "category"):
            frontmatter_lines.append(f"{key}: {meta.get(key, '')}")
        # Render capability so the auditor can run check #13.
        cap = skill.capability
        frontmatter_lines.append("capability:")
        frontmatter_lines.append(f"  verb_family: {cap.verb_family or 'MISSING'}")
        frontmatter_lines.append(f"  target_type: {cap.target_type or 'MISSING'}")
        frontmatter_lines.append(f"  affordance: {cap.affordance or 'MISSING'}")
        frontmatter_lines.append(f"  output: {cap.output or 'MISSING'}")
        if cap.preconditions:
            frontmatter_lines.append(
                "  preconditions: [" + ", ".join(cap.preconditions) + "]"
            )
        if cap.task_pattern:
            frontmatter_lines.append(f"  task_pattern: \"{cap.task_pattern}\"")
        if skill.parameters:
            frontmatter_lines.append(f"parameters: {list(skill.parameters.keys())}")
        frontmatter_lines.append("---")
        return "\n".join(frontmatter_lines) + "\n\n" + skill.body.strip()
