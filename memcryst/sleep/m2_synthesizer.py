"""M2 skill synthesizer.

Given a candidate cluster, format the synthesis prompt and ask the language
model to produce a SKILL.md text. Parse the result back into a Skill object
ready for the audit gate.

The prompt template lives in memcryst/prompts/synthesize_skill.md so we can
iterate on wording without touching the code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import frontmatter

from ..core import LLMClient
from ..memory_bank import MemoryEntry
from ..prompts import load_prompt
from ..skill_bank import Skill
from .m1_miner import ClusterCandidate

logger = logging.getLogger(__name__)


def _format_episode(entry: MemoryEntry, idx: int) -> str:
    lines = [
        f"Episode {idx}.",
        f"  question: {entry.question}",
    ]
    if entry.failure_reason:
        lines.append(f"  failure_reason: {entry.failure_reason}")
    if entry.improvement:
        lines.append(f"  improvement: {entry.improvement}")
    if entry.missed_information:
        lines.append(f"  missed_information: {entry.missed_information}")
    lines.append(f"  category: {entry.category}")
    lines.append(f"  weight: {entry.weight:.2f}")
    return "\n".join(lines)


def _format_episodes_block(members: list[MemoryEntry], top_k: int = 5) -> str:
    # Sort by weight so the most successful examples come first.
    sorted_members = sorted(members, key=lambda e: e.weight, reverse=True)[:top_k]
    return "\n\n".join(_format_episode(entry, i + 1) for i, entry in enumerate(sorted_members))


@dataclass
class SynthesisResult:
    skill: Optional[Skill]
    raw_text: str
    error: Optional[str] = None


class SkillSynthesizer:
    """Wrap a language model client to turn a cluster into a Skill."""

    def __init__(self, llm: LLMClient, prompt_name: str = "synthesize_skill") -> None:
        self.llm = llm
        self.prompt_template = load_prompt(prompt_name)

    def synthesize(self, cluster: ClusterCandidate, round_index: int = 1) -> SynthesisResult:
        episodes_block = _format_episodes_block(cluster.members, top_k=5)
        prompt = self.prompt_template.format(
            cluster_id=cluster.cluster_id,
            cluster_size=cluster.size,
            mean_reward=round(cluster.mean_reward, 3),
            category=cluster.category_mode,
            episodes_block=episodes_block,
        )

        messages = [
            {
                "role": "system",
                "content": "You produce reusable skills in MUSE SKILL.md format. Be precise and abstract.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, max_tokens=2000, temperature=0.3)
        skill, err = self._parse(raw, cluster, round_index)
        return SynthesisResult(skill=skill, raw_text=raw, error=err)

    @staticmethod
    def _extract_procedure(body: str) -> Optional[str]:
        """Pull the Procedure section out of a SKILL.md body for placeholder check."""
        lines = body.splitlines()
        in_proc = False
        collected: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("## procedure"):
                in_proc = True
                continue
            if in_proc and stripped.startswith("## "):
                break
            if in_proc:
                collected.append(line)
        if not collected:
            return None
        return "\n".join(collected)

    @staticmethod
    def _parse(
        raw: str,
        cluster: ClusterCandidate,
        round_index: int,
    ) -> tuple[Optional[Skill], Optional[str]]:
        text = raw.strip()
        # Trim accidental outer code fences if the model wrapped the whole file.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Common GPT-4o pattern. It wraps the YAML header in a ```yaml ... ```
        # block instead of using the --- delimiters that python-frontmatter
        # expects. Detect that and rewrite to the canonical form.
        if text.startswith("```yaml") or text.startswith("```yml"):
            lines = text.splitlines()
            # Drop the opening ```yaml line.
            lines = lines[1:]
            # Find the closing fence for the yaml block.
            close_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    close_idx = i
                    break
            if close_idx is not None:
                yaml_block = "\n".join(lines[:close_idx]).strip()
                body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")
                text = f"---\n{yaml_block}\n---\n{body}"

        try:
            post = frontmatter.loads(text)
        except Exception as exc:
            return None, f"frontmatter parse failed, {exc}"

        meta = dict(post.metadata or {})
        if "name" not in meta or "description" not in meta:
            return None, "frontmatter missing required name or description field"

        # Parametricity hard check. A skill with zero angle-bracket placeholders in
        # the procedure section is rejected at parse time. This catches the
        # contract violation we saw with GPT-4o-mini producing skills that just
        # paraphrase the source episodes without abstracting axes of variation.
        procedure_text = SkillSynthesizer._extract_procedure(post.content)
        if procedure_text is not None and "<" not in procedure_text:
            return None, "procedure section contains zero placeholder slots, contract violation"

        skill = Skill.from_frontmatter(meta, post.content)

        # Capability completeness check. If the model emitted partial or
        # invalid capability tags, after canonicalization we end up with
        # None values on some axes. We do NOT reject the skill at parse time
        # because audit check #13 will catch it; we just log so the caller
        # can see the issue. The skill will fail to register in the
        # capability index until fixed.
        if not skill.capability.is_complete():
            logger.warning(
                "skill %s has incomplete capability after canonicalization: %s",
                skill.name,
                skill.capability.to_dict(),
            )
        # Stamp provenance from the cluster.
        skill.ancestry = [entry.id for entry in cluster.members]
        skill.crystallized_from_cluster = cluster.cluster_id
        skill.crystallization_round = round_index
        skill.utility = max(0.4, min(0.95, cluster.mean_reward))
        return skill, None
