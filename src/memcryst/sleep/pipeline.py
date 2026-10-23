"""SleepPipeline. Wires M1, M2, M2 audit, and M3 together for one sleep step.

A single call to run mines clusters, synthesizes a draft skill for each
qualifying cluster, runs the audit gate, commits accepted skills, archives
crystallized memory entries, and applies feedback for any skill that has
crossed the failure threshold during the previous wake period.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..core import LLMClient
from ..memory_bank import MemoryRetriever, MemoryStore
from ..skill_bank import SkillStore

from .m1_miner import ClusterCandidate, PatternMiner
from .m2_audit import AuditResult, SkillAuditor
from .m2_synthesizer import SkillSynthesizer, SynthesisResult
from .m3_feedback import FeedbackLoop, FlagEvent

logger = logging.getLogger(__name__)


@dataclass
class CandidateOutcome:
    cluster: ClusterCandidate
    synthesis: SynthesisResult
    audit: Optional[AuditResult] = None
    accepted: bool = False


@dataclass
class SleepReport:
    started_at: float
    finished_at: float
    candidates: list[CandidateOutcome] = field(default_factory=list)
    accepted_skills: list[str] = field(default_factory=list)
    refined_skills: list[str] = field(default_factory=list)
    flag_events: list[FlagEvent] = field(default_factory=list)


class SleepPipeline:
    def __init__(
        self,
        memory_store: MemoryStore,
        memory_retriever: MemoryRetriever,
        skill_store: SkillStore,
        miner: PatternMiner,
        synthesizer: SkillSynthesizer,
        auditor: SkillAuditor,
        feedback: FeedbackLoop,
        max_candidates_per_run: int = 5,
        archive_on_accept: bool = True,
        refine_enabled: bool = True,
        semantic_store=None,
    ) -> None:
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.skill_store = skill_store
        self.miner = miner
        self.synthesizer = synthesizer
        self.auditor = auditor
        self.feedback = feedback
        self.max_candidates_per_run = max_candidates_per_run
        self.archive_on_accept = archive_on_accept
        # B (provenance-guided re-synthesis): when a semantic_store is supplied AND
        # MEMCRYST_PROVENANCE_SYNTH=1, re-crystallization of an existing skill is fed
        # the high-confidence facts whose ancestry edge points at it (memory→skill).
        self.semantic_store = semantic_store
        self._provenance_synth = (
            semantic_store is not None
            and os.environ.get("MEMCRYST_PROVENANCE_SYNTH", "0").strip().lower()
            not in ("", "0", "false", "no"))
        # refine_enabled=True (default): dedup-then-refine. Same-signature
        # cluster triggers re-synthesis on combined ancestry, replace existing.
        # refine_enabled=False: dedup-then-skip (Fix A only). Same-signature
        # cluster is rejected and skill bank stays frozen on that signature.
        self.refine_enabled = refine_enabled

    def _provenance_block(self, skill_id: str, top_k: int = 6, min_conf: float = 0.5):
        """B: gather the high-confidence facts whose ancestry edge points at this
        skill, formatted for the re-synthesis prompt. Returns None when disabled or
        no qualifying facts (so synthesize falls back to its normal behavior)."""
        if not self._provenance_synth:
            return None
        facts = [
            f for f in self.semantic_store
            if skill_id in getattr(f, "supports_skills", []) and f.confidence() >= min_conf
        ]
        if not facts:
            return None
        facts.sort(key=lambda f: -f.confidence())
        lines = [f"- {f.entity}: {f.fact} (helped {f.helped}/{f.used})"
                 for f in facts[:top_k]]
        return "\n".join(lines)

    def run(self) -> SleepReport:
        started = time.time()
        outcomes: list[CandidateOutcome] = []
        accepted_ids: list[str] = []
        refined_ids: list[str] = []

        try:
            candidates = self.miner.mine()
        except Exception as exc:
            logger.exception("M1 miner crashed, abandoning sleep step")
            return SleepReport(
                started_at=started,
                finished_at=time.time(),
                candidates=[],
                accepted_skills=[],
                flag_events=[],
            )
        logger.info("SleepPipeline mined %d candidate clusters", len(candidates))

        for cluster in candidates[: self.max_candidates_per_run]:
            try:
                synth = self.synthesizer.synthesize(cluster)
            except Exception as exc:
                logger.exception(
                    "M2 synthesis crashed for cluster %s", cluster.cluster_id
                )
                outcomes.append(
                    CandidateOutcome(
                        cluster=cluster,
                        synthesis=SynthesisResult(
                            skill=None, raw_text="", error=f"synth_exception: {exc!r}"
                        ),
                    )
                )
                continue

            if synth.skill is None:
                logger.warning(
                    "synthesis failed for cluster %s, err=%s",
                    cluster.cluster_id,
                    synth.error,
                )
                from pathlib import Path
                try:
                    debug_path = Path(self.memory_store.db_path).parent.parent / f"_debug_synth_cluster_{cluster.cluster_id}.txt"
                    debug_path.write_text(synth.raw_text or "")
                    logger.warning("raw synth output saved to %s", debug_path)
                except Exception as dbg_exc:
                    logger.warning("could not save raw synth output, %s", dbg_exc)
                outcomes.append(CandidateOutcome(cluster=cluster, synthesis=synth))
                continue

            try:
                audit = self.auditor.audit(synth.skill, cluster)
            except Exception as exc:
                logger.exception(
                    "M2 audit crashed for cluster %s", cluster.cluster_id
                )
                outcomes.append(
                    CandidateOutcome(
                        cluster=cluster,
                        synthesis=synth,
                        audit=None,
                        accepted=False,
                    )
                )
                continue

            accepted = audit.accepted
            if accepted:
                # Dedup-then-refine: if an existing skill already covers the
                # same capability signature, run a refinement synthesis on
                # combined evidence (existing ancestry + new cluster). If the
                # refined skill passes audit, REPLACE the existing skill
                # (same id, bumped crystallization round, merged ancestry).
                # This unblocks evolution after the verb x affordance space
                # is filled in cycle 0 and lets later cycles improve skills
                # in place rather than just being skipped.
                if synth.skill.capability.is_complete():
                    new_sig = synth.skill.capability.signature()
                    existing = next(
                        (
                            s
                            for s in self.skill_store
                            if s.capability.is_complete()
                            and s.capability.signature() == new_sig
                        ),
                        None,
                    )
                    if existing is not None and not self.refine_enabled:
                        # Fix A skip semantics: same-signature → reject.
                        logger.info(
                            "skipping duplicate skill (refine disabled): sig=%s",
                            new_sig,
                        )
                        accepted = False
                        outcomes.append(
                            CandidateOutcome(
                                cluster=cluster,
                                synthesis=synth,
                                audit=audit,
                                accepted=False,
                            )
                        )
                        continue
                    if existing is not None:
                        ancestry_entries = []
                        for eid in existing.ancestry:
                            entry = self.memory_store.get(eid)
                            if entry is not None:
                                ancestry_entries.append(entry)
                        seen_ids = {e.id for e in cluster.members}
                        combined_members = list(cluster.members) + [
                            e for e in ancestry_entries if e.id not in seen_ids
                        ]
                        from .m1_miner import ClusterCandidate as _CC
                        refine_cluster = _CC(
                            cluster_id=cluster.cluster_id + 1_000_000,
                            members=combined_members,
                            mean_reward=cluster.mean_reward,
                            persistence=cluster.persistence,
                            category_mode=cluster.category_mode,
                            centroid=cluster.centroid,
                        )
                        _grounding = self._provenance_block(existing.id)
                        try:
                            refined = self.synthesizer.synthesize(
                                refine_cluster, extra_grounding=_grounding)
                        except Exception:
                            logger.exception(
                                "refine synthesis crashed for cluster %s vs skill %s",
                                cluster.cluster_id, existing.id,
                            )
                            refined = None
                        if refined is None or refined.skill is None:
                            logger.info(
                                "refine failed for signature %s, keeping existing skill %s",
                                new_sig, existing.id,
                            )
                            accepted = False
                            outcomes.append(
                                CandidateOutcome(
                                    cluster=cluster,
                                    synthesis=synth,
                                    audit=audit,
                                    accepted=False,
                                )
                            )
                            continue
                        try:
                            refined_audit = self.auditor.audit(refined.skill, refine_cluster)
                        except Exception:
                            logger.exception(
                                "refine audit crashed for skill %s", existing.id,
                            )
                            refined_audit = None
                        if refined_audit is None or not refined_audit.accepted:
                            logger.info(
                                "refine audit rejected for signature %s, keeping existing %s",
                                new_sig, existing.id,
                            )
                            accepted = False
                            outcomes.append(
                                CandidateOutcome(
                                    cluster=cluster,
                                    synthesis=refined,
                                    audit=refined_audit,
                                    accepted=False,
                                )
                            )
                            continue
                        # Replace existing skill in place.
                        refined.skill.id = existing.id
                        refined.skill.crystallization_round = existing.crystallization_round + 1
                        new_ancestry_ids = [m.id for m in cluster.members]
                        merged_ancestry = list(existing.ancestry)
                        for aid in new_ancestry_ids:
                            if aid not in merged_ancestry:
                                merged_ancestry.append(aid)
                        refined.skill.ancestry = merged_ancestry
                        refined.skill.use_count = existing.use_count
                        refined.skill.failure_count = existing.failure_count
                        refined.skill.utility = existing.utility
                        refined.skill.created_at = existing.created_at
                        # (else the MUSE-style failure_notes are wiped every sleep).
                        refined.skill.failure_notes = list(existing.failure_notes)
                        # Force re-embedding because the body text changed.
                        refined.skill.embedding = None
                        self.skill_store.update(refined.skill)
                        accepted_ids.append(refined.skill.id)
                        refined_ids.append(refined.skill.id)
                        if self.archive_on_accept:
                            for entry in cluster.members:
                                self.memory_store.archive(entry.id, refined.skill.id)
                        logger.info(
                            "refined skill %s with cluster %s (round %d -> %d, ancestry %d -> %d)",
                            existing.id, cluster.cluster_id,
                            existing.crystallization_round, refined.skill.crystallization_round,
                            len(existing.ancestry), len(merged_ancestry),
                        )
                        outcomes.append(
                            CandidateOutcome(
                                cluster=cluster,
                                synthesis=refined,
                                audit=refined_audit,
                                accepted=True,
                            )
                        )
                        continue

                self.skill_store.add(synth.skill)
                accepted_ids.append(synth.skill.id)
                if self.archive_on_accept:
                    for entry in cluster.members:
                        self.memory_store.archive(entry.id, synth.skill.id)

            outcomes.append(
                CandidateOutcome(
                    cluster=cluster,
                    synthesis=synth,
                    audit=audit,
                    accepted=accepted,
                )
            )

        try:
            flag_events = self.feedback.run()
        except Exception as exc:
            logger.exception("M3 feedback crashed, returning empty flag list")
            flag_events = []

        finished = time.time()
        return SleepReport(
            started_at=started,
            finished_at=finished,
            candidates=outcomes,
            accepted_skills=accepted_ids,
            refined_skills=refined_ids,
            flag_events=flag_events,
        )
