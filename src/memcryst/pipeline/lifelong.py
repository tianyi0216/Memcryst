"""Lifelong wake-sleep loop.

The agent is deployed over a task stream in cycles of (wake: attempt N tasks
with the current memory and skill banks -> sleep: mine, synthesize, audit and
revise skills -> repeat). Every task is scored before the system learns from
it (prequential scoring), so the per-task record is the lifelong learning
curve. Wake writes episodic entries and semantic facts and returns task
outcomes to the facts and skills that were served; sleep clusters episodes
into skill candidates, audits them and applies the skill->memory feedback.

Run through ``src/run.py`` with a YAML config (see README), or directly:

    python -m memcryst.pipeline.lifelong --input-dir <empty-dir> --output-dir <run-dir> ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcryst.agent import (
    AlfworldEnv,
    ReactAgent,
    StepTrace,
    TrajectoryResult,
    attribute_failure_axis,
    family_from_task_description,
    trajectory_to_memory_entry,
)
from memcryst.agent.wake_writeback import _format_trajectory_for_prompt
from memcryst.agent.env_factory import make_env
from memcryst.core import LLMClient
from memcryst.memory_bank import (
    MemoryEntry, MemoryRetriever, MemoryStore,
    load_pair_stats, save_pair_stats, record_pair_outcome,
    SemanticMemoryStore, SemanticMemoryRetriever, extract_grounding_facts,
)



# Ablation knobs (additive; defaults preserve production behavior exactly).
# MEMCRYST_WAKE_MODE: agent mode during wake episodes (default dual retrieval)
# MEMCRYST_EVAL_MODE: agent mode during per-cycle eval (default dual retrieval)
import os as _os
_WAKE_MODE = _os.environ.get("MEMCRYST_WAKE_MODE", "capability_plus_memory")
_EVAL_MODE = _os.environ.get("MEMCRYST_EVAL_MODE", "capability_plus_memory")
# MEMCRYST_WAKE_RETRIES: train-time-only best-of-k (retry failed wake tasks)
# MEMCRYST_WB_SUCCESS_ONLY: skip writeback of failure trajectories
_WAKE_RETRIES = int(_os.environ.get("MEMCRYST_WAKE_RETRIES", "0"))
_WB_SUCCESS_ONLY = _os.environ.get("MEMCRYST_WB_SUCCESS_ONLY", "0").strip().lower() not in ("", "0", "false", "no")
# MEMCRYST_SEMANTIC_FROM_FAILURE: also distill affordance facts from FAILED episodes.
# Default OFF — failure-derived facts proved noisy (examine -25 in the v2 run);
# success-only is the clean default.
_SEM_FROM_FAILURE = _os.environ.get("MEMCRYST_SEMANTIC_FROM_FAILURE", "0").strip().lower() not in ("", "0", "false", "no")
_SKILL_FAILMEM = _os.environ.get("MEMCRYST_SKILL_FAILMEM", "0").strip().lower() not in ("", "0", "false", "no")


def _extract_skill_failure_note(task_description: str, trajectory_text: str, skill, llm) -> str:
    """Distill ONE short, actionable, transferable lesson from a FAILED use of `skill`.
    Returns '' on any failure. The lesson goes into the skill's failure_notes and is
    shown to future agents using this skill."""
    proc = ""
    try:
        from memcryst.agent.react_agent import _extract_section
        proc = (_extract_section(skill.body, "Procedure") or "")[:500]
    except Exception:
        pass
    prompt = (
        "An agent FAILED a task while following this skill. Write ONE short, actionable "
        "lesson (max 25 words) that would help a FUTURE agent using this SAME skill avoid "
        "the SAME failure. State it as a general rule (e.g. 'if the target is not on the "
        "first surface, check shelves and sidetables before giving up'), NOT a description "
        "of this episode. If nothing transferable, output NONE.\n\n"
        f"Skill: {skill.name} (family={getattr(skill.capability,'verb_family',None)})\n"
        f"Procedure:\n{proc}\n\nTask: {task_description}\nTrajectory (last steps):\n{trajectory_text}\n\n"
        "Output ONLY the one-sentence lesson, or NONE."
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content": "You write one short transferable skill-failure lesson. Output one sentence only."},
             {"role": "user", "content": prompt}], max_tokens=60, temperature=0.2).strip()
    except Exception:
        return ""
    if not raw or raw.strip().upper().startswith("NONE"):
        return ""
    return raw.strip().strip('"').splitlines()[0][:160]

def build_memory_retriever(memory_store, args=None, pair_stats_path=None):
    """Construct a MemoryRetriever, threading N9 + N10 boosts when active."""
    kwargs = {}
    if args is not None:
        if getattr(args, "n10_verified_rank_boost", 0.0) > 0:
            kwargs["verified_boost"] = args.n10_verified_rank_boost
        if getattr(args, "n9_pair_tracking", False) and pair_stats_path is not None:
            kwargs["pair_stats"] = load_pair_stats(pair_stats_path)
            kwargs["pair_boost"] = args.n9_pair_boost
    return MemoryRetriever(memory_store, **kwargs)
from memcryst.agent.bi_hooks import (
    n6_distill_success_entry,
    n8_append_failure_context,
    n13_joint_refine_low_utility,
)
from memcryst.memory_bank.schema import EntryStatus
from memcryst.skill_bank import SkillRetriever, SkillStore
from memcryst.sleep import (
    FeedbackLoop,
    PatternMiner,
    SkillAuditor,
    SkillSynthesizer,
    SleepPipeline,
)

console = Console()


def restore_archived_memory(memory_store: MemoryStore) -> int:
    """Re-activate archived entries before re-crystallization.

    After a sleep cycle the miner only sees ACTIVE entries; restoring first
    lets re-crystallization see the whole bank rather than the residue.
    """
    restored = 0
    for entry in memory_store:
        if entry.status == EntryStatus.ARCHIVED:
            entry.status = EntryStatus.ACTIVE
            restored += 1
    if restored:
        memory_store.save()
    return restored


def run_wake_phase(
    env: AlfworldEnv,
    agent_llm: LLMClient,
    writeback_llm: LLMClient,
    memory_store: MemoryStore,
    memory_retriever: MemoryRetriever,
    skill_store: SkillStore,
    skill_retriever: SkillRetriever,
    task_indices: list[int],
    max_steps: int,
    m3_mode: str = "axis",
    top_k_skills: int = 2,
    top_k_memory: int = 3,
    mem_boost_on_success: float = 0.0,
    mark_verified_on_success: bool = False,
    memory_utility_alpha: float = 0.0,
    n6_distill: bool = False,
    n9_pair_stats_path: Optional[Path] = None,
    condensed_retriever=None,
    semantic_retriever=None,
    semantic_store=None,
    writeback_enabled: bool = True,
) -> dict:
    """Run agent on a batch of training tasks.

    For each task:
      1. agent.run() with capability retrieval, producing a TrajectoryResult.
      2. trajectory_to_memory_entry() writes a new memory entry.
      3. For each used skill, SkillStore.record_use() with the outcome.
      4. If failed AND a skill was used, attribute_failure_axis() picks an
         axis and SkillStore.record_axis_failure() is called.

    Returns a report dict.
    """
    report = {
        "tasks_run": 0,
        "successes": 0,
        "memory_writeback": 0,
        "skill_uses_recorded": 0,
        "axis_failures_recorded": 0,
        "writeback_failures": 0,
        "attribution_failures": 0,
        "by_family": {},
        "per_task": [],   # prequential per-task scores for the lifelong/deployment curve
    }

    for ti in task_indices:
        # best-of-k at TRAIN time only: rare successes on hard families seed the bank
        for _attempt in range(1 + max(0, _WAKE_RETRIES)):
            agent = ReactAgent(
                env=env,
                llm=agent_llm,
                skill_retriever=skill_retriever,
                memory_retriever=memory_retriever,
                condensed_retriever=condensed_retriever,
                semantic_retriever=semantic_retriever,
                mode=_WAKE_MODE,
                top_k_skills=top_k_skills,
                top_k_memory=top_k_memory,
                max_steps=max_steps,
            )
            r = agent.run(task_idx=ti)
            if r.success:
                break
        report["tasks_run"] += 1
        if r.success:
            report["successes"] += 1
        fam = family_from_task_description(r.task_description)
        # prequential record: attempted with the bank state BEFORE learning from
        # this task (writeback/crystallize come after) -> leak-free lifelong score.
        _pt = {"task_idx": ti, "family": fam,
               "success": bool(r.success), "steps": r.steps,
               "reward": round(float(r.final_reward), 3)}
        _gf = getattr(env, "last_gamefile", None)          # ALFWorld: the game actually played (order audit)
        if _gf:
            _pt["gamefile"] = _gf
        if getattr(r, "diag", None):
            _pt.update(r.diag)   # failure-taxonomy tags (revisit_extra / kn_refuted / kn_verified)
        report["per_task"].append(_pt)
        bf = report["by_family"].setdefault(fam, {"n": 0, "succ": 0})
        # prequential family base-rate (over PRIOR tasks of this family) for the
        # base-rate-relative lift signal; 0.5 prior before any data on this family.
        _fam_base = (bf["succ"] / bf["n"]) if bf["n"] > 0 else 0.5
        # demand-driven capture (MEMCRYST_CAPTURE_DEMAND=1): a failed question
        # records its entities as extraction priorities for future session
        # writebacks — selection acting on the WRITE path (skills/queries teach
        # memory what to capture). QA streams only (fact_style=qa).
        if (os.environ.get("MEMCRYST_CAPTURE_DEMAND", "0") == "1"
                and fam != "session" and float(r.final_reward) < 0.5):
            import re as _re_cd
            _cd_stop = {"What", "When", "Where", "Who", "Whom", "Why", "How",
                        "Did", "Does", "Do", "Is", "Was", "Were", "Would",
                        "Could", "Should", "Which", "The", "And", "But", "For",
                        "Has", "Have", "Had", "Are", "Will", "Answer"}
            _cd_ents = [w for w in _re_cd.findall(
                r"\b[A-Z][a-z]{2,}\b", r.task_description) if w not in _cd_stop]
            if _cd_ents:
                from memcryst.memory_bank import semantic as _semantic_mod
                _semantic_mod.add_capture_demand(_cd_ents)
                report.setdefault("capture_demand_adds", 0)
                report["capture_demand_adds"] += len(_cd_ents)
        bf["n"] += 1
        if r.success:
            bf["succ"] += 1

        console.print(
            f"  [dim]wake task {ti:>3}[/dim] family={fam:<8} "
            f"success={str(r.success):<5} steps={r.steps:>2} "
            f"skills={r.used_skill_ids} desc={r.task_description[:50]}"
        )

        # Step 1: trajectory writeback. WB_SUCCESS_ONLY skips failures
        # (weak authors mis-diagnose their own failures). writeback_enabled=False
        # (freeze-memory intervention): NO new memory content is written — retrieval
        # and the rank-only confidence feedback keep working on the frozen bank.
        _wb_skip = (os.environ.get("MEMCRYST_WB_SESSION_ONLY", "0") not in ("", "0")
                    and fam != "session")
        if (_WB_SUCCESS_ONLY and not r.success) or not writeback_enabled or _wb_skip:
            new_entry = None
        else:
            new_entry = trajectory_to_memory_entry(r, writeback_llm)
        if new_entry is not None:
            memory_store.add(new_entry)
            report["memory_writeback"] += 1
            # MEMCRYST_M2W_EXEMPLAR=1: store each verified-successful step as a worked example.
            if (os.environ.get("MEMCRYST_M2W_EXEMPLAR", "0") == "1"
                    and r.success and r.step_trace):
                import re as _re2
                _act = (r.step_trace[0].action or "").strip()
                _m2 = _re2.search(r"answer\s*:\s*([A-T])\b", _act, _re2.IGNORECASE)
                _obs0_lines = (getattr(r, "initial_obs", "") or "").splitlines()
                _cand = next((ln for ln in _obs0_lines
                              if _m2 and ln.startswith(f"{_m2.group(1).upper()}. ")), "")
                _goal = next((ln for ln in _obs0_lines if ln.startswith("Task goal:")), "")
                from memcryst.memory_bank.schema import MemoryEntry as _ME2
                memory_store.add(_ME2(
                    question=r.task_description,
                    failure_reason="",
                    improvement=(f"WORKED EXAMPLE (verified) — {_goal}\n"
                                 f"Correct element: {_cand[:160]}\n"
                                 f"Correct reply: {_act[:80]}"),
                    category="m2w_exemplar"))
            if (os.environ.get("MEMCRYST_WB_VERBATIM", "0") == "2"
                    and r.success):
                _obs0 = getattr(r, "initial_obs", "") or ""
                for _ci in range(1, min(12, (len(_obs0) + 1799) // 1800)):
                    _chunk = _obs0[_ci * 1800:(_ci + 1) * 1800]
                    if len(_chunk) < 200:
                        break
                    from memcryst.memory_bank.schema import MemoryEntry as _ME
                    memory_store.add(_ME(
                        question=f"{r.task_description} (part {_ci + 1})",
                        failure_reason="",
                        improvement="VERBATIM EXCERPT:\n" + _chunk,
                        category="session_excerpt"))
        else:
            report["writeback_failures"] += 1

        if semantic_store is not None:
            # ③ + ②④ feedback FIRST (on the facts retrieved DURING this task): update
            # their calibrated confidence (used/helped) and link co-retrieved skills
            # (the supports_skills ancestry edge). r.used_memory_ids are the semantic
            # facts that were in context; r.used_skill_ids the co-retrieved skills.
            try:
                # backward-edge lesion (MEMCRYST_NO_FEEDBACK=1): content still
                # appended, but outcomes stop re-ranking facts (③ confidence /
                # supports_skills lineage edges) — isolates what outcome
                # selection is worth on top of pure accumulation.
                # update (fact used/helped + supports_skills co-retrieval edges);
                # FB_SKILL gates skill utility below. NO_FEEDBACK=1 forces both
                _fb_mem = (os.environ.get("MEMCRYST_FB_MEM", "1") == "1"
                           and os.environ.get("MEMCRYST_NO_FEEDBACK", "0") != "1")
                if r.used_memory_ids and _fb_mem:
                    _wdecay = float(os.environ.get("MEMCRYST_SEMANTIC_WDECAY", "1.0"))
                    semantic_store.record_outcome(
                        r.used_memory_ids, r.success, r.used_skill_ids,
                        weight_decay=_wdecay, family_base=_fam_base)
            except Exception as exc:
                console.print(f"[dim]semantic feedback failed: {exc}[/dim]")
            # write-back NEW facts (with episode provenance ②). SUCCESS -> location +
            # affordance; FAILURE -> affordance only (opt-in via _SEM_FROM_FAILURE).
            if (r.success or _SEM_FROM_FAILURE) and writeback_enabled:
                try:
                    traj_txt = _format_trajectory_for_prompt(r, last_n=12)
                    outcome = "SUCCESS" if r.success else f"FAILURE (reached {r.steps} steps)"
                    facts = extract_grounding_facts(
                        r.task_description, outcome, traj_txt, writeback_llm, success=r.success)
                    if facts:
                        semantic_store.add_many(facts, episode_id=ti)
                        report.setdefault("semantic_facts_added", 0)
                        report["semantic_facts_added"] += len(facts)
                except Exception as exc:
                    console.print(f"[dim]semantic extraction failed: {exc}[/dim]")

        # Optional second-pass extraction (MEMCRYST_SECONDPASS=1): after a failed
        # question is scored, re-extract facts from stored verbatim excerpts.
        if (os.environ.get("MEMCRYST_SECONDPASS", "0") == "1"
                and semantic_store is not None and fam != "session"
                and float(r.final_reward) < 0.5 and memory_retriever is not None):
            try:
                _sp_hits = memory_retriever.retrieve(r.task_description, top_k=4)
                _sp_txt = "\n\n".join(
                    (getattr(e, "improvement", "") or "")[:1600]
                    for e, _ in _sp_hits)[:6000]
                if len(_sp_txt) > 400:
                    _sp_prompt = (
                        "A question could NOT be answered from extracted memory:\n"
                        f"QUESTION: {r.task_description}\n\n"
                        "Below are stored conversation excerpts. Extract up to 8 facts "
                        "that would answer this question or closely related ones, one "
                        "per line as 'entity | fact'. Keep exact names, dates and "
                        "numbers. If the excerpts contain nothing relevant, output "
                        "NONE.\n\n" + _sp_txt)
                    _sp_raw = writeback_llm.chat(
                        [{"role": "system", "content":
                          "You extract personal memory facts, one per line as "
                          "'entity | fact'. Output only fact lines or NONE."},
                         {"role": "user", "content": _sp_prompt}],
                        max_tokens=400, temperature=0.2)
                    _sp_facts = []
                    for _ln in (_sp_raw or "").splitlines():
                        if "|" not in _ln:
                            continue
                        _e, _, _f = _ln.partition("|")
                        _e, _f = _e.strip(" -*\t"), _f.strip()
                        if _e and _f and len(_e) < 80:
                            _sp_facts.append((_e, _f, "other"))
                    if _sp_facts:
                        semantic_store.add_many(_sp_facts[:8], episode_id=ti)
                        report.setdefault("secondpass_facts_added", 0)
                        report["secondpass_facts_added"] += len(_sp_facts[:8])
            except Exception as exc:
                console.print(f"[dim]second-pass extraction failed: {exc}[/dim]")

        # Step 2: record_use on every retrieved skill (FB_SKILL switch).
        _fb_skill = (os.environ.get("MEMCRYST_FB_SKILL", "1") == "1"
                     and os.environ.get("MEMCRYST_NO_FEEDBACK", "0") != "1")
        for sid in (r.used_skill_ids if _fb_skill else []):
            skill_store.record_use(sid, success=r.success)
            report["skill_uses_recorded"] += 1

        # outcome returns ONLY to the source episodes recorded at its creation
        # (skill.ancestry, the derivation record from mining) via an EMA on
        # entry.utility — which multiplies episodic retrieval scores directly.
        # "shuffled" keeps update count and magnitude but randomizes the
        # targets (deterministic per-skill seed, resume-safe). NOTE (mechanism
        # truth): facts have no creation-time derivation from skills — their
        # skill coupling is runtime parameterization — so the lineage channel
        # targets episodes; fact outcome flow stays on the FB_MEM channel.
        _lin = os.environ.get("MEMCRYST_FB_LINEAGE", "").strip().lower()
        if _lin in ("correct", "shuffled") and r.used_skill_ids:
            import random as _rnd
            _alpha = float(os.environ.get("MEMCRYST_LIN_ALPHA", "0.2"))
            _target_val = 1.0 if r.success else 0.0
            _all_ids = list(getattr(memory_store, "_entries", {}).keys())
            _lin_updates = 0
            for sid in r.used_skill_ids:
                _sk = skill_store.get(sid) if hasattr(skill_store, "get") else None
                _anc = list(getattr(_sk, "ancestry", []) or []) if _sk else []
                if not _anc:
                    continue
                if _lin == "shuffled":
                    _rng = _rnd.Random(hash((sid, "lin_shuf")) & 0xffffffff)
                    _anc = _rng.sample(_all_ids, min(len(_anc), len(_all_ids)))
                for _eid in _anc:
                    _e = memory_store.get(_eid)
                    if _e is None:
                        continue
                    _e.utility = (1 - _alpha) * _e.utility + _alpha * _target_val
                    _lin_updates += 1
            if _lin_updates:
                memory_store.save()
                report.setdefault("lineage_updates", 0)
                report["lineage_updates"] += _lin_updates

        if (_SKILL_FAILMEM and not r.success and r.used_skill_ids):
            try:
                sid = r.used_skill_ids[0]
                sk = skill_store.get(sid)
                if sk is not None:
                    traj_txt = _format_trajectory_for_prompt(r, last_n=12)
                    note = _extract_skill_failure_note(
                        r.task_description, traj_txt, sk, writeback_llm)
                    if note:
                        skill_store.add_failure_note(sid, note)
                        report.setdefault("skill_failure_notes", 0)
                        report["skill_failure_notes"] += 1
            except Exception as exc:
                console.print(f"[dim]skill failure-note extraction failed: {exc}[/dim]")

        # Step 2b: bidirectional feedback updates on memory side (N2/N3/N4).
        # These run on the memory entries the agent saw in the prompt this
        # task. r.used_memory_ids is populated in capability_plus_memory mode.
        outcome = 1.0 if r.success else 0.0
        if mark_verified_on_success and r.success:
            for mid in r.used_memory_ids:
                memory_store.mark_verified(mid)
        if mem_boost_on_success > 0 and r.success:
            for mid in r.used_memory_ids:
                memory_store.boost_weight(mid, amount=mem_boost_on_success)
        if memory_utility_alpha > 0:
            for mid in r.used_memory_ids:
                memory_store.update_utility(mid, outcome=outcome, alpha=memory_utility_alpha)

        if n6_distill and r.success and r.used_skill_ids:
            primary_skill = skill_store.get(r.used_skill_ids[0])
            if primary_skill is not None:
                n6_distill_success_entry(
                    memory_store, r.task_description, primary_skill,
                    r.used_memory_ids, r.steps,
                )

        if n9_pair_stats_path is not None and (r.used_skill_ids or r.used_memory_ids):
            stats = load_pair_stats(n9_pair_stats_path)
            record_pair_outcome(stats, r.used_skill_ids, r.used_memory_ids, r.success)
            save_pair_stats(n9_pair_stats_path, stats)

        # Step 3: axis attribution on failure (axis mode only).
        # In m3_mode=paper, paper Eq 9 uses uniform gamma over all ancestry,
        # so we skip the LLM axis attribution call entirely.
        if m3_mode == "axis" and not r.success and r.used_skill_ids:
            primary_sid = r.used_skill_ids[0]
            primary_skill = skill_store.get(primary_sid)
            if primary_skill is not None:
                axis = attribute_failure_axis(r, primary_skill, writeback_llm)
                if axis is not None:
                    skill_store.record_axis_failure(primary_sid, axis)
                    report["axis_failures_recorded"] += 1
                else:
                    report["attribution_failures"] += 1

    return report


def run_sleep_phase(
    memory_store: MemoryStore,
    skill_store: SkillStore,
    synth_llm: LLMClient,
    audit_llm: LLMClient,
    m3_mode: str = "axis",
    refine_enabled: bool = True,
    min_cluster_size: int = 3,
    audit_min_pass: int = 10,
    max_candidates_per_run: int = 10,
    m3_failure_threshold: float = 0.3,
    m3_max_severity: float = 0.9,
    m3_gc_threshold: float = 0.0,
    m3_min_uses: int = 1,
    m3_target: str = "memory",
    m3_protect_verified: bool = False,
    n7_revive_archived: bool = False,
    n8_mem_refine_fail: bool = False,
    n11_force_seed_utility: float = 0.0,
    n12_priority_cluster: bool = False,
    n13_joint_refine: bool = False,
    n13_min_utility: float = 0.5,
    n13_min_uses: int = 2,
    semantic_store=None,
) -> dict:
    """One full sleep step: restore archived → M3 → re-crystallize.

    m3_mode:
      - "axis" (default, our extension): axis-amplified flagging using LLM
        attribution from wake.
      - "paper" (paper Eq 9): uniform gamma over all ancestry.
      - "off": skip M3 entirely. No memory weight changes.

    Returns a report dict.
    """
    restored = restore_archived_memory(memory_store)
    memory_retriever = MemoryRetriever(memory_store)

    axis_aware = (m3_mode == "axis")
    m3_active = (m3_mode != "off")

    if m3_active:
        pre_feedback = FeedbackLoop(
            skill_store, memory_store,
            failure_threshold=m3_failure_threshold,
            min_uses=m3_min_uses,
            max_severity=m3_max_severity,
            axis_aware=axis_aware,
            gc_threshold=m3_gc_threshold,
            target=m3_target,
            protect_verified=m3_protect_verified,
            revive_archived_on_fail=n7_revive_archived,
            priority_cluster_on_fail=n12_priority_cluster,
        )
        flag_events = pre_feedback.run()

        if n8_mem_refine_fail:
            for ev in flag_events:
                skill = skill_store.get(ev.skill_id)
                if skill is None:
                    continue
                n8_append_failure_context(
                    memory_store, skill, skill.ancestry, skill.failure_rate(),
                )
    else:
        flag_events = []

    if n13_joint_refine:
        n13_joint_refine_low_utility(
            memory_store, min_utility=n13_min_utility, min_uses=n13_min_uses,
        )

    feedback_for_pipeline = (
        FeedbackLoop(
            skill_store, memory_store,
            failure_threshold=m3_failure_threshold,
            min_uses=m3_min_uses,
            max_severity=m3_max_severity,
            axis_aware=axis_aware,
            gc_threshold=m3_gc_threshold,
            target=m3_target,
            protect_verified=m3_protect_verified,
        )
        if m3_active
        else FeedbackLoop(skill_store, memory_store, failure_threshold=1.1, axis_aware=False)
    )
    pipeline = SleepPipeline(
        memory_store=memory_store,
        memory_retriever=memory_retriever,
        skill_store=skill_store,
        miner=PatternMiner(
            memory_retriever, min_cluster_size=min_cluster_size,
            min_mean_reward=0.4,
            force_seed_utility_threshold=n11_force_seed_utility,
            priority_seed_enabled=n12_priority_cluster,
        ),
        synthesizer=SkillSynthesizer(synth_llm),
        auditor=SkillAuditor(audit_llm, min_pass=audit_min_pass),
        feedback=feedback_for_pipeline,
        max_candidates_per_run=max_candidates_per_run,
        archive_on_accept=True,
        refine_enabled=refine_enabled,
        semantic_store=semantic_store,
    )
    sleep_report = pipeline.run()

    return {
        "restored_archived": restored,
        "m3_flag_events": len(flag_events),
        "m3_dominant_axes": [e.dominant_axis for e in flag_events if e.dominant_axis],
        "memory_flagged": sum(1 for e in memory_store if e.status == EntryStatus.FLAGGED),
        "memory_active": sum(1 for e in memory_store if e.status == EntryStatus.ACTIVE),
        "memory_archived": sum(1 for e in memory_store if e.status == EntryStatus.ARCHIVED),
        "sleep_candidates": len(sleep_report.candidates),
        "sleep_accepted": len(sleep_report.accepted_skills),
        "sleep_refined": len(sleep_report.refined_skills),
        "sleep_new": len(sleep_report.accepted_skills) - len(sleep_report.refined_skills),
    }


def _maybe_condensed(memory_store, agent_llm, output_dir, tag: str, mode: str = None):
    """Build a fresh condensed-memory retriever from the CURRENT memory bank when
    the given mode needs it (memory_condense / capability_plus_condense). Rebuilt
    each call so the condensed baseline grows with the stream. Returns None
    otherwise. mode defaults to _EVAL_MODE; pass _WAKE_MODE for the wake phase."""
    if (mode or _EVAL_MODE) not in ("memory_condense", "capability_plus_condense"):
        return None
    from memcryst.memory_bank.condense import build_condensed_memory, CondensedMemoryRetriever
    cdir = output_dir / "condensed" / tag
    cstore = build_condensed_memory(
        memory_store=memory_store, llm=agent_llm, output_dir=cdir,
        min_cluster_size=3, min_mean_reward=0.4,
    )
    cr = CondensedMemoryRetriever(cstore)
    cr.ensure_embeddings()
    return cr


def run_eval_phase(
    env_split: str,
    agent_llm: LLMClient,
    skill_store: SkillStore,
    skill_retriever: SkillRetriever,
    memory_retriever: MemoryRetriever,
    task_indices: list[int],
    max_steps: int,
    top_k_skills: int = 2,
    top_k_memory: int = 3,
    condensed_retriever=None,
    semantic_retriever=None,
) -> dict:
    """Eval _EVAL_MODE on held-out tasks. Fresh env to start at game 0
    and skip forward. condensed_retriever is required for memory_condense /
    capability_plus_condense eval modes.
    """
    report = {"tasks_run": 0, "successes": 0, "by_family": {}, "per_task": []}
    if not task_indices:        # --eval-tasks 0: skip held-out eval entirely
        return report
    env = make_env(split=env_split, max_steps=max_steps)
    # Skip forward to the held-out range. Note ALFWorld reset advances
    # internally regardless of task_idx.
    skip = task_indices[0]
    for _ in range(skip):
        env.reset()

    for ti in task_indices:
        agent = ReactAgent(
            env=env,
            llm=agent_llm,
            skill_retriever=skill_retriever,
            memory_retriever=memory_retriever,
            condensed_retriever=condensed_retriever,
            semantic_retriever=semantic_retriever,
            mode=_EVAL_MODE,
            top_k_skills=top_k_skills,
            top_k_memory=top_k_memory,
            max_steps=max_steps,
        )
        _t0 = time.time()
        r = agent.run(task_idx=ti)
        _wall = time.time() - _t0
        report["tasks_run"] += 1
        if r.success:
            report["successes"] += 1
        fam = family_from_task_description(r.task_description)
        bf = report["by_family"].setdefault(fam, {"n": 0, "succ": 0})
        bf["n"] += 1
        if r.success:
            bf["succ"] += 1
        report["per_task"].append(
            {
                "task_idx": ti,
                "task_description": r.task_description,
                "family": fam,
                "success": r.success,
                "steps": r.steps,
                "llm_calls": r.llm_calls,
                "wall_time_s": round(_wall, 2),
            }
        )
    return report


_GAP_POOL: dict = {}


def gap_gate_phase(
    env_split: str,
    agent_llm: LLMClient,
    skill_store: SkillStore,
    memory_store: MemoryStore,
    skill_retriever: SkillRetriever,
    memory_retriever: MemoryRetriever,
    val_indices: list[int],
    max_steps: int,
    top_k_skills: int = 2,
    top_k_memory: int = 3,
    downweight: float = 0.2,
) -> dict:
    """Competence-gap-gated bidirectional crystallization feedback.

    Per-family Δ = succ(dual) − succ(vanilla) on a held-out validation window
    (NOT the eval set). Families with Δ<=0 (no competence gap) -> their skills go
    dormant (excluded from retrieval) AND their source memory ancestry is
    downweighted, so the next cycle stops retrieving/crystallizing skills where
    the agent is already capable. A family is revived if its gap reappears later.
    This is the clean SKILL0-style helpfulness signal (not failure-blame), wired
    into the wake-sleep loop = the bidirectional crystallization feedback.
    """
    from collections import defaultdict

    from memcryst.agent import react_agent as _ra

    # Anti-lock-in: the probe must measure the UNGATED system. A family gated
    # last cycle would otherwise probe at Δ=0 (nothing injected) and stay gated
    # forever. Re-activate all skills and clear the injection gate for the probe.
    for s in skill_store:
        if getattr(s, "dormant", False):
            s.dormant = False
            skill_store.save_one(s)
    _ra.set_gated_families(set())

    stats = defaultdict(lambda: {"dual": [0, 0], "van": [0, 0]})
    for mode, key in (("capability_plus_memory", "dual"), ("vanilla", "van")):
        env = make_env(split=env_split, max_steps=max_steps)
        # Fresh environment instance for the probe.
        for ti in val_indices:
            agent = ReactAgent(
                env=env,
                llm=agent_llm,
                skill_retriever=(skill_retriever if mode != "vanilla" else None),
                memory_retriever=(memory_retriever if mode != "vanilla" else None),
                mode=mode,
                top_k_skills=top_k_skills,
                top_k_memory=top_k_memory,
                max_steps=max_steps,
            )
            r = agent.run(task_idx=ti)
            fam = family_from_task_description(r.task_description)
            stats[fam][key][0] += 1 if r.success else 0
            stats[fam][key][1] += 1

    # Gating rule. Default: gate families where dual does not beat vanilla (Δ<=0,
    # noisy on 12 tasks). MEMCRYST_COMPETENCE_GATE=<thr>: principled competence
    # gate from the competence-gap law (r=-0.80) — gate families whose BASE
    # (vanilla) competence is already high (>=thr), since prior can only mislead a
    # family the agent already nails (examine/move), and keep prior for weak
    # families (heat/cool) where the procedure adds value. Independent of dual noise.
    comp_thr = os.environ.get("MEMCRYST_COMPETENCE_GATE", "").strip()
    comp_thr = float(comp_thr) if comp_thr else None
    pooled = os.environ.get("MEMCRYST_GAP_POOLED", "0").strip() not in ("", "0")
    pool_thr = float(os.environ.get("MEMCRYST_GAP_POOL_THR", "0.03"))
    pool_minn = int(os.environ.get("MEMCRYST_GAP_POOL_MINN", "15"))
    deltas: dict[str, float] = {}
    gated: set[str] = set()
    for fam, s in stats.items():
        dr = s["dual"][0] / s["dual"][1] if s["dual"][1] else 0.0
        vr = s["van"][0] / s["van"][1] if s["van"][1] else 0.0
        deltas[fam] = round(dr - vr, 3)
        if pooled:
            P = _GAP_POOL.setdefault(fam, [0, 0, 0, 0])
            P[0] += s["dual"][0]; P[1] += s["dual"][1]
            P[2] += s["van"][0];  P[3] += s["van"][1]
            continue
        if comp_thr is not None:
            if vr >= comp_thr:
                gated.add(fam)
        elif dr - vr <= 0:
            gated.add(fam)
    if pooled:
        for fam, P in _GAP_POOL.items():
            if P[1] >= pool_minn and P[3] >= pool_minn:
                pd = P[0] / P[1] - P[2] / P[3]
                deltas[fam] = round(pd, 3)
                if pd <= -pool_thr:
                    gated.add(fam)

    n_dorm = n_undorm = 0
    gated_anc: set[str] = set()
    for skill in skill_store:
        sfam = skill.capability.verb_family if skill.capability else None
        if sfam in gated and not skill.dormant:
            skill.dormant = True
            skill_store.save_one(skill)
            n_dorm += 1
            gated_anc.update(skill.ancestry or [])
        elif sfam not in gated and skill.dormant:
            skill.dormant = False
            skill_store.save_one(skill)
            n_undorm += 1

    n_dn = 0
    for mid in gated_anc:
        e = memory_store.get(mid)
        if e is not None:
            e.weight = float(e.weight) * downweight
            n_dn += 1
    if n_dn:
        memory_store.save()

    # Activate the full per-family injection gate (skill AND memory withheld
    # for gated families) for this cycle's eval and the next cycle's wake.
    _ra.set_gated_families(gated)

    return {
        "deltas": deltas,
        "gated_families": sorted(gated),
        "skills_dormant": n_dorm,
        "skills_revived": n_undorm,
        "memory_downweighted": n_dn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MEMCRYST wake-sleep orchestrator")
    parser.add_argument("--input-dir", required=True, help="initial memory + skill bank")
    parser.add_argument("--output-dir", required=True, help="working dir for cycle artifacts")
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--tasks-per-cycle", type=int, default=30)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--eval-task-start", type=int, default=100, help="held-out range start")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--agent-model", default="gpt-4o-mini")
    parser.add_argument("--writeback-model", default="gpt-4o-mini")
    parser.add_argument("--synth-model", default="gpt-4o-mini")
    parser.add_argument("--audit-model", default="gpt-4o-mini")
    parser.add_argument("--split", default="eval_out_of_distribution",
                        help="WAKE/stream split (e.g. train). Gap-gate val also uses this.")
    parser.add_argument("--eval-split", default="",
                        help="HELD-OUT eval split for the per-cycle learning curve "
                             "(e.g. eval_out_of_distribution). Empty = same as --split "
                             "(legacy, leaky). Set disjoint from --split for a clean curve.")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="run the held-out eval every N cycles (plus always the last). "
                             "Streaming: 2 = eval at cycles 1,3,5,... to save compute.")
    parser.add_argument("--eval-split2", default="",
                        help="optional SECOND held-out eval split per checkpoint, e.g. "
                             "eval_in_distribution alongside eval_out_of_distribution.")
    parser.add_argument("--wake-pool", type=int, default=0,
                        help="in-domain streaming: wrap wake indices into [0,wake_pool) "
                             "so wake revisits this pool across cycles (multi-epoch) and "
                             "never enters the held-out eval range. 0 = linear (default).")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--m3-mode",
        choices=["axis", "paper", "off"],
        default="axis",
        help="axis: LLM axis attribution + axis-amplified flag (our extension). "
             "paper: uniform gamma over all ancestry (paper Eq 9). "
             "off: skip M3 entirely, no memory weight changes. "
             "paper/off modes also skip wake-time axis attribution LLM calls.",
    )
    parser.add_argument(
        "--refine", dest="refine", action="store_true",
        help="dedup-then-refine on same signature (default)",
    )
    parser.add_argument(
        "--no-refine", dest="refine", action="store_false",
        help="dedup-then-skip (Fix A only, no refine)",
    )
    parser.set_defaults(refine=True)
    parser.add_argument("--top-k-skills", type=int, default=2)
    parser.add_argument("--top-k-memory", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--audit-min-pass", type=int, default=10)
    parser.add_argument("--max-candidates-per-run", type=int, default=10)
    parser.add_argument(
        "--m3-failure-threshold", type=float, default=0.3,
        help="paper f*. Skill flagged if failure_rate > this. Lower = M3 fires more aggressively.",
    )
    parser.add_argument(
        "--m3-max-severity", type=float, default=0.9,
        help="cap on per-flag severity. Higher = stronger weight decay per flag event.",
    )
    parser.add_argument(
        "--m3-gc-threshold", type=float, default=0.0,
        help="paper w^arch. After flag, archive ancestry entries with weight below this. 0 disables GC.",
    )
    parser.add_argument(
        "--m3-min-uses", type=int, default=1,
        help="minimum skill uses before M3 considers it for flag. Higher = wait longer for statistical signal.",
    )
    parser.add_argument(
        "--m3-target", choices=["memory", "skill", "both", "none"], default="memory",
        help="memory (paper, flag ancestry), skill (N1, decay skill utility), both",
    )
    parser.add_argument(
        "--m3-protect-verified", action="store_true",
        help="N3: skip flagging memory entries whose verified_uses > 0",
    )
    parser.add_argument(
        "--m3-mem-boost", type=float, default=0.0,
        help="N2: after successful wake task, boost retrieved memory weight by this amount",
    )
    parser.add_argument(
        "--m3-memory-utility-alpha", type=float, default=0.0,
        help="N4: per-memory utility EMA rate. 0 disables. Typical 0.1.",
    )
    # --- Advanced bidirectional ablations (N6-N13). All default OFF. ---
    parser.add_argument("--n6-distill-on-success", action="store_true",
        help="N6: write a compiled-lesson memory entry after successful agent runs")
    parser.add_argument("--n7-revive-archived-on-fail", action="store_true",
        help="N7: revive ARCHIVED ancestry entries when skill exceeds failure threshold")
    parser.add_argument("--n8-mem-refine-on-fail", action="store_true",
        help="N8: append a refinement note to failing skill's ancestry memory entries (no weight change)")
    parser.add_argument("--n9-pair-tracking", action="store_true",
        help="N9: track per-(skill, memory) co-use outcomes; boost retrieval of high-pair-success memories")
    parser.add_argument("--n9-pair-boost", type=float, default=0.5,
        help="N9 retrieval bonus coefficient on best pair success rate. Effective when --n9-pair-tracking is on.")
    parser.add_argument("--n10-verified-rank-boost", type=float, default=0.0,
        help="N10: retrieval score += boost * log(1+verified_uses). 0 disables.")
    parser.add_argument("--n11-force-seed-utility", type=float, default=0.0,
        help="N11: M1 force-seeds clusters around memory entries with utility >= this threshold. 0 disables.")
    parser.add_argument("--n12-priority-cluster", action="store_true",
        help="N12: failing-skill ancestry gets priority_next_cluster=True; M1 force-seeds them next sleep")
    parser.add_argument("--n13-joint-refine", action="store_true",
        help="N13: at sleep time, append a self-doubt annotation to low-utility memory entries")
    parser.add_argument("--n13-min-utility", type=float, default=0.5)
    parser.add_argument("--n13-min-uses", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    progress_path = output_dir / "_progress.json"

    # Resume detection: if output_dir already has _progress.json, we resume
    # from the next cycle. Memory and skill stores are auto-persisted into
    # output_dir/memory and output_dir/skills, so they reload exactly where
    # the previous run stopped after the last completed cycle.
    #
    # If output_dir exists but _progress.json does NOT, that means a previous
    # run was killed mid-cycle (between two checkpoint writes). We clean and
    # start fresh in that case to avoid partial-cycle inconsistency.
    resume = False
    if progress_path.exists():
        resume = True
    elif output_dir.exists():
        shutil.rmtree(output_dir)

    if not resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(input_dir / "memory", output_dir / "memory")
        shutil.copytree(input_dir / "skills", output_dir / "skills")
        # semantic store is part of the bank: chain it exactly like memory/skills.
        # (Pre-seeding output_dir instead gets WIPED by the stale-dir cleanup above
        if (input_dir / "semantic").exists():
            shutil.copytree(input_dir / "semantic", output_dir / "semantic")

    console.print(Panel.fit("MEMCRYST wake-sleep continuous loop", style="bold cyan"))
    console.print(f"input: {input_dir}")
    console.print(f"output: {output_dir}")
    console.print(f"cycles: {args.num_cycles}, tasks/cycle: {args.tasks_per_cycle}, eval: {args.eval_tasks}\n")

    agent_llm = LLMClient(model=args.agent_model, temperature=0.0)
    writeback_llm = LLMClient(model=args.writeback_model, temperature=0.2)
    synth_llm = LLMClient(model=args.synth_model, temperature=0.3)
    audit_llm = LLMClient(model=args.audit_model, temperature=0.0)

    memory_store = MemoryStore(store_dir=output_dir / "memory")
    skill_store = SkillStore(store_dir=output_dir / "skills")
    console.print(f"loaded {len(memory_store)} memory entries, {len(skill_store)} skills\n")

    # Semantic memory (declarative grounding facts) — built only when a phase uses
    # capability_plus_semantic. Reuses the memory retriever's embedder so facts and
    # tasks share an embedding space. Procedural memory above stays the skill fuel.
    semantic_store = semantic_retriever = None
    if "capability_plus_semantic" in (_WAKE_MODE, _EVAL_MODE):
        _embed_src = MemoryRetriever(memory_store)
        semantic_store = SemanticMemoryStore(
            store_dir=output_dir / "semantic", embed_fn=_embed_src.embed)
        semantic_retriever = SemanticMemoryRetriever(semantic_store, embed_fn=_embed_src.embed)
        console.print(f"semantic memory: {len(semantic_store)} grounding facts loaded\n")

    all_reports = {"cycles": [], "initial_eval": None, "final_eval": None}
    resume_from_cycle = 0

    if resume:
        progress = json.loads(progress_path.read_text())
        all_reports["initial_eval"] = progress["initial_eval"]
        all_reports["cycles"] = progress["completed_cycles"]
        resume_from_cycle = len(progress["completed_cycles"])
        init = all_reports["initial_eval"]
        init_pct = 100 * init["successes"] / init["tasks_run"]
        console.print(
            f"[yellow bold]RESUME[/yellow bold] from cycle {resume_from_cycle} "
            f"({resume_from_cycle}/{args.num_cycles} cycles complete)\n"
            f"  initial eval: {init['successes']}/{init['tasks_run']} ({init_pct:.0f}%)\n"
        )
        if resume_from_cycle >= args.num_cycles:
            console.print("[green]All cycles already done. Writing final outputs.[/green]\n")

    # Wake env shared across cycles. env.reset(task_idx=ti) explicitly indexes
    # the task by id, so no fast-forward is needed when resuming.
    wake_env = make_env(split=args.split, max_steps=args.max_steps)
    eval_split = args.eval_split or args.split

    # Initial eval (cycle 0 baseline) — only if not resuming.
    if not resume:
        console.rule(f"initial eval (held-out split={eval_split})")
        pair_path = output_dir / "memory" / "pair_stats.json"
        initial_eval = run_eval_phase(
            env_split=eval_split,
            agent_llm=agent_llm,
            skill_store=skill_store,
            skill_retriever=SkillRetriever(skill_store, signature_llm=agent_llm),
            memory_retriever=build_memory_retriever(memory_store, args, pair_path),
            condensed_retriever=_maybe_condensed(memory_store, agent_llm, output_dir, "initial"),
            semantic_retriever=semantic_retriever,
            task_indices=list(range(args.eval_task_start, args.eval_task_start + args.eval_tasks)),
            max_steps=args.max_steps,
            top_k_skills=args.top_k_skills,
            top_k_memory=args.top_k_memory,
        )
        all_reports["initial_eval"] = initial_eval
        _ie_n = initial_eval['tasks_run']
        console.print(
            f"initial eval: {initial_eval['successes']}/{_ie_n} "
            f"({100*initial_eval['successes']/_ie_n:.0f}%)\n" if _ie_n
            else "initial eval: skipped (eval-tasks=0)\n"
        )
        # Checkpoint after initial eval so a kill before cycle 0 is also resumable.
        progress_path.write_text(json.dumps(
            {"completed_cycles": [], "initial_eval": initial_eval},
            indent=2,
        ))

    # Cycle loop. cycle_offset is derived from resume_from_cycle so that
    # wake_task_indices stays consistent with what would have run from scratch.
    # Phase-level checkpoints live in output_dir/_phase_<phase>_c<cycle>.json.
    # On entering a cycle we check if wake/sleep already completed (then we
    # skip those phases and reuse the saved report). This prevents the
    # memory-bloat from re-running wake after a mid-sleep kill.
    cycle_offset = resume_from_cycle * args.tasks_per_cycle
    for cycle in range(resume_from_cycle, args.num_cycles):
        wake_done_path = output_dir / f"_phase_wake_c{cycle}.json"
        sleep_done_path = output_dir / f"_phase_sleep_c{cycle}.json"
        # pair_path is used by eval phase below regardless of whether wake
        # ran from scratch or was resumed from checkpoint, so initialize here.
        pair_path = output_dir / "memory" / "pair_stats.json"

        if wake_done_path.exists():
            wake_report = json.loads(wake_done_path.read_text())
            console.rule(f"cycle {cycle} — wake phase [RESUMED from checkpoint]")
            console.print(
                f"  reusing wake report: {wake_report['successes']}/{wake_report['tasks_run']} success, "
                f"{wake_report['memory_writeback']} writebacks\n"
            )
        else:
            console.rule(f"cycle {cycle} — wake phase")
            if args.wake_pool > 0:
                # In-domain multi-epoch streaming: wrap wake indices into a fixed
                # pool [0, wake_pool) so wake revisits the same wake games across
                # cycles (multiple epochs) and NEVER marches into the held-out eval
                # range (eval_task_start should be >= wake_pool). Leak-free.
                wake_task_indices = [(cycle_offset + i) % args.wake_pool
                                     for i in range(args.tasks_per_cycle)]
            else:
                wake_task_indices = list(range(cycle_offset, cycle_offset + args.tasks_per_cycle))
            skill_retriever = SkillRetriever(skill_store, signature_llm=agent_llm)
            pair_path = output_dir / "memory" / "pair_stats.json"
            memory_retriever = build_memory_retriever(memory_store, args, pair_path)
            wake_report = run_wake_phase(
                env=wake_env,
                agent_llm=agent_llm,
                writeback_llm=writeback_llm,
                memory_store=memory_store,
                memory_retriever=memory_retriever,
                skill_store=skill_store,
                skill_retriever=skill_retriever,
                task_indices=wake_task_indices,
                max_steps=args.max_steps,
                m3_mode=args.m3_mode,
                top_k_skills=args.top_k_skills,
                top_k_memory=args.top_k_memory,
                mem_boost_on_success=args.m3_mem_boost,
                mark_verified_on_success=args.m3_protect_verified,
                memory_utility_alpha=args.m3_memory_utility_alpha,
                n6_distill=args.n6_distill_on_success,
                n9_pair_stats_path=(pair_path if args.n9_pair_tracking else None),
                # condense wake modes need the condensed retriever too (built per
                # cycle from current memory); None for non-condense modes.
                condensed_retriever=_maybe_condensed(
                    memory_store, agent_llm, output_dir, f"c{cycle}_wake", mode=_WAKE_MODE),
                semantic_retriever=semantic_retriever,
                semantic_store=semantic_store,
                # freeze-memory intervention (co-evolution loop test): after cycle K,
                # no NEW memory content is written; skills keep crystallizing from the
                # frozen experience pool. MEMCRYST_FREEZE_MEM_AFTER=K (0 = never).
                writeback_enabled=not (
                    int(os.environ.get("MEMCRYST_FREEZE_MEM_AFTER", "0") or 0)
                    and cycle >= int(os.environ.get("MEMCRYST_FREEZE_MEM_AFTER", "0"))),
            )
            console.print(
                f"\ncycle {cycle} wake: {wake_report['successes']}/{wake_report['tasks_run']} success, "
                f"{wake_report['memory_writeback']} writebacks, "
                f"{wake_report['axis_failures_recorded']} axis attributions\n"
            )
            # prune grounding facts that the accumulated skill outcomes attribute to
            # failure (used enough, success-rate-when-used too low). Skill failure
            # repairs memory — removal only, no rewrite.
            if (semantic_store is not None
                    and os.environ.get("MEMCRYST_SEMANTIC_PRUNE", "0").strip().lower()
                    not in ("", "0", "false", "no")):
                removed = semantic_store.prune_low_confidence(
                    min_lift=float(os.environ.get("MEMCRYST_PRUNE_MIN_LIFT", "-0.1")),
                    min_uses=int(os.environ.get("MEMCRYST_PRUNE_MIN_USES", "3")))
                if removed:
                    console.print(f"  [semantic prune] removed {len(removed)} low-confidence "
                                  f"facts attributed to skill failures "
                                  f"(e.g. {removed[0].entity}: {removed[0].fact[:40]})\n")
            # env-refuted location facts to where the entity was actually observed.
            # Keeps context full (weak backbones need the scaffolding) + fixes content.
            if (semantic_store is not None
                    and os.environ.get("MEMCRYST_FACT_CORRECT", "0").strip().lower()
                    not in ("", "0", "false", "no")):
                fixed = semantic_store.correct_refuted_facts(
                    min_refuted=int(os.environ.get("MEMCRYST_FACT_COR_MINREF", "3")),
                    min_rate=float(os.environ.get("MEMCRYST_FACT_COR_MINRATE", "0.6")),
                    multi_loc=os.environ.get("MEMCRYST_FACT_COR_MULTILOC", "0").strip().lower()
                    not in ("", "0", "false", "no"))
                if fixed:
                    console.print(f"  [semantic correct] rewrote {len(fixed)} refuted facts "
                                  f"to observed location (e.g. {fixed[0][0]}: "
                                  f"'{fixed[0][1][:30]}' -> '{fixed[0][2][:30]}')\n")
            # Snapshot wake completion so a sleep-phase kill does not re-run wake.
            wake_done_path.write_text(json.dumps(wake_report, indent=2))
        cycle_offset += args.tasks_per_cycle

        if sleep_done_path.exists():
            sleep_report = json.loads(sleep_done_path.read_text())
            console.rule(f"cycle {cycle} — sleep phase [RESUMED from checkpoint]")
            console.print(
                f"  reusing sleep report: {sleep_report['m3_flag_events']} flag events, "
                f"{sleep_report['sleep_new']} new, {sleep_report['sleep_refined']} refined\n"
            )
        elif (int(os.environ.get("MEMCRYST_FREEZE_SKILLS_AFTER", "0") or 0)
              and cycle >= int(os.environ.get("MEMCRYST_FREEZE_SKILLS_AFTER", "0"))):
            # freeze-skills intervention (co-evolution loop test): after cycle K the
            # skill bank stays as-is; memory keeps accumulating.
            console.rule(f"cycle {cycle} — sleep SKIPPED (skills frozen)")
            # The stub carries every key run_sleep_phase returns.
            sleep_report = {
                "restored_archived": 0, "m3_flag_events": 0,
                "m3_dominant_axes": ["frozen"],
                "memory_flagged": 0,
                "memory_active": sum(1 for _e in memory_store), "memory_archived": 0,
                "sleep_candidates": 0, "sleep_accepted": 0,
                "sleep_refined": 0, "sleep_new": 0,
            }
            sleep_done_path.write_text(json.dumps(sleep_report, indent=2))
        else:
            console.rule(f"cycle {cycle} — sleep phase (M3 + re-crystallize)")
            sleep_report = run_sleep_phase(
                memory_store, skill_store, synth_llm, audit_llm,
                m3_mode=args.m3_mode,
                refine_enabled=args.refine,
                min_cluster_size=args.min_cluster_size,
                audit_min_pass=args.audit_min_pass,
                max_candidates_per_run=args.max_candidates_per_run,
                m3_failure_threshold=args.m3_failure_threshold,
                m3_max_severity=args.m3_max_severity,
                m3_gc_threshold=args.m3_gc_threshold,
                m3_min_uses=args.m3_min_uses,
                m3_target=args.m3_target,
                m3_protect_verified=args.m3_protect_verified,
                n7_revive_archived=args.n7_revive_archived_on_fail,
                n8_mem_refine_fail=args.n8_mem_refine_on_fail,
                n11_force_seed_utility=args.n11_force_seed_utility,
                n12_priority_cluster=args.n12_priority_cluster,
                n13_joint_refine=args.n13_joint_refine,
                n13_min_utility=args.n13_min_utility,
                n13_min_uses=args.n13_min_uses,
                semantic_store=semantic_store,
            )
            console.print(
                f"sleep: {sleep_report['m3_flag_events']} flag events "
                f"(axes: {sleep_report['m3_dominant_axes']}), "
                f"{sleep_report['sleep_new']} new, "
                f"{sleep_report['sleep_refined']} refined\n"
            )
            sleep_done_path.write_text(json.dumps(sleep_report, indent=2))

        # competence-gap-gated bidirectional feedback (env-gated). Runs after
        # crystallization so dormant/downweight affects this cycle's eval AND the
        # next cycle's wake retrieval + crystallization.
        if os.environ.get("MEMCRYST_GAP_GATED", "0").strip().lower() not in ("", "0", "false", "no"):
            console.rule(f"cycle {cycle} — gap-gate (competence-gap bidirectional feedback)")
            gg_n = int(os.environ.get("MEMCRYST_GAP_VAL_N", "12"))
            gg_start = int(os.environ.get("MEMCRYST_GAP_VAL_START", "50"))
            _gg_idx = os.environ.get("MEMCRYST_GAP_VAL_IDX", "").strip()
            gg_val = ([int(x) for x in _gg_idx.split(",") if x.strip()]
                      if _gg_idx else list(range(gg_start, gg_start + gg_n)))
            gg = gap_gate_phase(
                env_split=args.split,
                agent_llm=agent_llm,
                skill_store=skill_store,
                memory_store=memory_store,
                skill_retriever=SkillRetriever(skill_store, signature_llm=agent_llm),
                memory_retriever=build_memory_retriever(memory_store, args, pair_path),
                val_indices=gg_val,
                max_steps=args.max_steps,
                top_k_skills=args.top_k_skills,
                top_k_memory=args.top_k_memory,
            )
            console.print(
                f"  gap-gate Δ={gg['deltas']} | gated={gg['gated_families']} | "
                f"dormant={gg['skills_dormant']} revived={gg['skills_revived']} "
                f"mem_downweighted={gg['memory_downweighted']}\n"
            )
            # persist the live gate so post-run evals can reuse the final decision
            (output_dir / "gated_families.json").write_text(json.dumps(gg["gated_families"]))

        # Streaming: only run the (expensive) held-out eval every --eval-every
        # cycles, plus always on the final cycle. The wake success above is the
        # free prequential/online point (recorded every cycle regardless).
        do_eval = ((cycle % args.eval_every) == 0) or (cycle == args.num_cycles - 1)
        cycle_eval = None
        if do_eval:
            console.rule(f"cycle {cycle} — held-out eval (mode={_EVAL_MODE}, split={eval_split})")
            skill_retriever = SkillRetriever(skill_store, signature_llm=agent_llm)
            memory_retriever_eval = build_memory_retriever(memory_store, args, pair_path)
            cycle_eval = run_eval_phase(
                env_split=eval_split,
                agent_llm=agent_llm,
                skill_store=skill_store,
                skill_retriever=skill_retriever,
                memory_retriever=memory_retriever_eval,
                condensed_retriever=_maybe_condensed(memory_store, agent_llm, output_dir, f"c{cycle}"),
                semantic_retriever=semantic_retriever,
                task_indices=list(range(args.eval_task_start, args.eval_task_start + args.eval_tasks)),
                max_steps=args.max_steps,
                top_k_skills=args.top_k_skills,
                top_k_memory=args.top_k_memory,
            )
            pct = 100 * cycle_eval["successes"] / cycle_eval["tasks_run"]
            console.print(
                f"cycle {cycle} held-out eval: {cycle_eval['successes']}/{cycle_eval['tasks_run']} ({pct:.0f}%)\n"
            )

        # Optional SECOND held-out eval on a different split (e.g. report both
        # eval_in_distribution and eval_out_of_distribution per checkpoint).
        cycle_eval2 = None
        eval_split2 = args.eval_split2.strip() if args.eval_split2 else ""
        if do_eval and eval_split2 and eval_split2 != eval_split:
            console.rule(f"cycle {cycle} — held-out eval2 (mode={_EVAL_MODE}, split={eval_split2})")
            cycle_eval2 = run_eval_phase(
                env_split=eval_split2,
                agent_llm=agent_llm,
                skill_store=skill_store,
                skill_retriever=SkillRetriever(skill_store, signature_llm=agent_llm),
                memory_retriever=build_memory_retriever(memory_store, args, pair_path),
                condensed_retriever=_maybe_condensed(memory_store, agent_llm, output_dir, f"c{cycle}_s2"),
                semantic_retriever=semantic_retriever,
                task_indices=list(range(args.eval_task_start, args.eval_task_start + args.eval_tasks)),
                max_steps=args.max_steps,
                top_k_skills=args.top_k_skills,
                top_k_memory=args.top_k_memory,
            )
            p2 = 100 * cycle_eval2["successes"] / cycle_eval2["tasks_run"]
            console.print(
                f"cycle {cycle} held-out eval2 ({eval_split2}): {cycle_eval2['successes']}/{cycle_eval2['tasks_run']} ({p2:.0f}%)\n"
            )

        all_reports["cycles"].append(
            {
                "cycle": cycle,
                "wake": wake_report,
                "sleep": sleep_report,
                "eval": cycle_eval,
                "eval2": cycle_eval2,
            }
        )

        # Checkpoint after every completed cycle. A kill after this line and
        # before the next cycle's wake completes will resume cleanly from
        # cycle+1. Mid-cycle kills land on phase checkpoints (_phase_*_c{cycle}.json)
        # so wake / sleep do not re-run unnecessarily.
        progress_path.write_text(json.dumps(
            {
                "completed_cycles": all_reports["cycles"],
                "initial_eval": all_reports["initial_eval"],
            },
            indent=2,
        ))
        # Cycle fully done — remove its phase checkpoints to keep the dir tidy.
        for p in (wake_done_path, sleep_done_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # Final report.
    console.rule("evolution curve")
    table = Table()
    table.add_column("phase")
    table.add_column("success rate")
    table.add_column("memory size")
    table.add_column("skill count")
    # Use all_reports["initial_eval"] so we don't depend on the local
    # variable initial_eval, which is unset in resume mode.
    init_ev = all_reports["initial_eval"]
    init_pct = 100 * init_ev["successes"] / init_ev["tasks_run"]
    table.add_row("initial (v1)", f"{init_ev['successes']}/{init_ev['tasks_run']} ({init_pct:.0f}%)", "-", "-")
    for c in all_reports["cycles"]:
        evp = c["eval"]
        # held-out eval is run only every --eval-every cycles; show wake (online)
        # success on skipped cycles so the table still has a per-cycle number.
        if evp is not None:
            eval_cell = f"{evp['successes']}/{evp['tasks_run']} ({100*evp['successes']/evp['tasks_run']:.0f}%)"
        else:
            wk = c.get("wake") or {}
            ws, wt = wk.get("successes"), wk.get("tasks_run")
            eval_cell = f"(skip; wake {ws}/{wt})" if wt else "(skipped)"
        mem_active = c["sleep"]["memory_active"]
        new_n = c["sleep"].get("sleep_new", c["sleep"]["sleep_accepted"])
        refined_n = c["sleep"].get("sleep_refined", 0)
        table.add_row(
            f"cycle {c['cycle']}",
            eval_cell,
            str(mem_active),
            f"+{new_n} new, {refined_n} refined",
        )
    console.print(table)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(all_reports, indent=2))
        console.print(f"\nresults dumped to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
