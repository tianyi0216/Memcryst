"""SleepPipeline. Wires M1, M2, M2 audit, and M3 together for one sleep step.

A single call to run mines clusters, synthesizes a draft skill for each
qualifying cluster, runs the audit gate, commits accepted skills, archives
crystallized memory entries, and applies feedback for any skill that has
crossed the failure threshold during the previous wake period.
"""

from __future__ import annotations

import logging
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

    def run(self) -> SleepReport:
        started = time.time()
        outcomes: list[CandidateOutcome] = []
        accepted_ids: list[str] = []

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
            # Each cluster is processed independently. A failure in synthesis
            # or audit for one cluster does not abort the sleep run. We log
            # the error and move on so M3 feedback always gets to run.
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
                # Commit the skill to the skill bank.
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
            flag_events=flag_events,
        )
