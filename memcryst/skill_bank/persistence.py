"""Skill bank persistence.

Each skill lives in a directory named after its slug. The directory contains a
SKILL.md file with YAML frontmatter and a markdown body. Optional script
artifacts can be added by future versions but are not required for v1.

    ~/.memcryst/skills/
        clean-object-procedure/
            SKILL.md
        heat-object-procedure/
            SKILL.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import frontmatter

from .capability_index import CapabilityIndex
from .schema import Skill, slugify

logger = logging.getLogger(__name__)


def default_skills_dir() -> Path:
    return Path.home() / ".memcryst" / "skills"


def _format_skill_md(skill: Skill) -> str:
    """Render a skill to SKILL.md text using YAML frontmatter."""
    post = frontmatter.Post(skill.body)
    for key, value in skill.to_frontmatter_dict().items():
        post[key] = value
    return frontmatter.dumps(post)


def _parse_skill_md(text: str) -> Skill:
    post = frontmatter.loads(text)
    return Skill.from_frontmatter(dict(post.metadata), post.content)


class SkillStore:
    """Filesystem backed skill bank with in memory cache."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else default_skills_dir()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self.capability_index = CapabilityIndex()
        self.load()

    # ----- load and save -----

    def load(self) -> None:
        self._skills.clear()
        self.capability_index = CapabilityIndex()
        for skill_dir in self.store_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            try:
                text = md_path.read_text()
                skill = _parse_skill_md(text)
                self._skills[skill.id] = skill
                self.capability_index.register(skill)
            except Exception as exc:
                logger.warning("could not parse %s, %s", md_path, exc)

    def _path_for(self, skill: Skill) -> Path:
        # Include id suffix in the directory name so two skills whose names
        # collide after slugify (for example "Clean Object" and "clean-object")
        # do not overwrite each other on disk.
        return self.store_dir / f"{skill.slug}-{skill.id}" / "SKILL.md"

    def save_one(self, skill: Skill) -> Path:
        path = self._path_for(skill)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_format_skill_md(skill))
        return path

    # ----- CRUD -----

    def add(self, skill: Skill) -> Skill:
        self._skills[skill.id] = skill
        self.capability_index.register(skill)
        self.save_one(skill)
        return skill

    def update(self, skill: Skill) -> Skill:
        import time as _t

        skill.updated_at = _t.time()
        old = self._skills.get(skill.id)
        if old is not None:
            self.capability_index.deregister(old)
        self._skills[skill.id] = skill
        self.capability_index.register(skill)
        self.save_one(skill)
        return skill

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def get_by_name(self, name: str) -> Optional[Skill]:
        slug = slugify(name)
        for skill in self._skills.values():
            if skill.slug == slug:
                return skill
        return None

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    # ----- usage stats updates -----

    def record_use(self, skill_id: str, success: bool) -> Optional[Skill]:
        import time as _t

        skill = self.get(skill_id)
        if skill is None:
            return None
        skill.use_count += 1
        if not success:
            skill.failure_count += 1
        # Simple EMA on utility, alpha 0.2.
        reward = 1.0 if success else 0.0
        alpha = 0.2
        skill.utility = (1.0 - alpha) * skill.utility + alpha * reward
        skill.updated_at = _t.time()
        self.save_one(skill)
        return skill

    def record_axis_failure(self, skill_id: str, axis: str) -> Optional[Skill]:
        """Attribute a failure to a specific capability axis. Used by M3 to
        identify the dominant failure axis when reweighting ancestry memory."""
        import time as _t

        skill = self.get(skill_id)
        if skill is None:
            return None
        counts = dict(skill.axis_failure_counts or {})
        counts[axis] = counts.get(axis, 0) + 1
        skill.axis_failure_counts = counts
        skill.updated_at = _t.time()
        self.save_one(skill)
        return skill

    def stats(self) -> dict:
        total = len(self)
        used = sum(1 for s in self if s.use_count > 0)
        indexed = sum(1 for s in self if s.capability.is_complete())
        return {
            "total": total,
            "used_at_least_once": used,
            "mean_utility": sum(s.utility for s in self) / total if total else 0.0,
            "capability_indexed": indexed,
            "capability_buckets": self.capability_index.stats()["buckets"],
            "store_dir": str(self.store_dir),
        }
