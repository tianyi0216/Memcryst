"""Semantic memory: declarative facts, as opposed to episodic traces.

  SKILL  = procedural knowledge (HOW): the operator sequence for a task family.
  MEMORY = declarative knowledge (THAT): facts about the environment that the
           procedure needs, which vary per instance and can be abstracted across
           episodes.

The store, retriever and extractor are domain-agnostic: a fact is an
``(entity, fact-sentence, support)`` triple and the only domain signal is the
content the LLM extracts. Facts carry outcome statistics (used / helped) that
are updated when a task that was served the fact succeeds or fails.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Merge threshold: a new (entity, fact) whose embedding is >= this cosine to an
# existing fact for the SAME entity is treated as the same fact (support++),
# rather than appended — keeps the store crisp and lets statistics accumulate.
import os as _os
MERGE_TAU = float(_os.environ.get('MEMCRYST_SEMANTIC_MERGE_TAU', '0.86'))


FACT_TYPES = ("affordance", "location", "other")

# ALFWorld receptacle types (instance ids stripped). Used by the skill-framed fact
# verification diagnostic to match a location fact's predicted receptacle-type
# against where the entity is actually observed in the environment.
RECEPTACLE_TYPES = (
    "countertop", "cabinet", "drawer", "fridge", "microwave", "sinkbasin",
    "stoveburner", "garbagecan", "shelf", "bed", "desk", "sidetable", "diningtable",
    "coffeetable", "sofa", "armchair", "ottoman", "dresser", "safe", "toilet",
    "bathtubbasin", "handtowelholder", "towelholder", "toiletpaperhanger",
    "coffeemachine", "stove", "sink", "table", "cart", "shelves",
)


def receptacle_types_in(text: str) -> set:
    """Return the set of ALFWorld receptacle TYPES mentioned in a text string
    (a fact sentence, an observation, or a `go to <recep N>` location)."""
    t = (text or "").lower()
    return {r for r in RECEPTACLE_TYPES if r in t}


# Coarse "where would you find an object" categories — the granularity the agent
# actually reasons at (a location fact helps by narrowing to the right KIND of place,
# not the exact receptacle instance). Used by category-level verification.
RECEPTACLE_CATEGORY = {
    "countertop": "surface", "table": "surface", "diningtable": "surface",
    "coffeetable": "surface", "sidetable": "surface", "desk": "surface",
    "dresser": "surface", "shelf": "surface", "shelves": "surface", "sofa": "surface",
    "armchair": "surface", "bed": "surface", "ottoman": "surface", "cart": "surface",
    "stoveburner": "surface", "stove": "surface",
    "cabinet": "container", "drawer": "container", "fridge": "container",
    "microwave": "container", "safe": "container",
    "coffeemachine": "appliance", "sinkbasin": "appliance", "sink": "appliance",
    "toilet": "bathroom", "bathtubbasin": "bathroom", "handtowelholder": "bathroom",
    "towelholder": "bathroom", "toiletpaperhanger": "bathroom", "garbagecan": "bathroom",
}


def receptacle_categories_in(text: str) -> set:
    """Coarse-category version of receptacle_types_in (surface/container/appliance/...)."""
    return {RECEPTACLE_CATEGORY[r] for r in receptacle_types_in(text)
            if r in RECEPTACLE_CATEGORY}


# Instead of a hand-written {receptacle: category} table (ALFWorld-specific), let an
# LLM assign each location a broad category. Cached per-type (~25 calls total). The
# taxonomy word-list is the only knob; swap it per domain (QA: entity/answer types).
_LLM_CAT_CACHE: dict = {}
_LLM_CAT_OPTIONS = ("surface", "container", "appliance", "fixture")
_LLM_CAT_PROMPT = (
    "Where would an object be found — classify this location into ONE broad category.\n"
    "Location: '{loc}'\n"
    "Options: surface (open surfaces/furniture you set things on), "
    "container (closed storage you must open: drawer, cabinet, fridge, microwave), "
    "appliance (functional stations: sink, coffee machine, stove), "
    "fixture (fixed wall/bathroom items).\n"
    "Answer with exactly one word from the options. Category:"
)


def llm_receptacle_category(rtype: str, llm_fn) -> Optional[str]:
    """LLM category for one receptacle type, cached. llm_fn: prompt->str."""
    if rtype in _LLM_CAT_CACHE:
        return _LLM_CAT_CACHE[rtype]
    cat = None
    try:
        out = (llm_fn(_LLM_CAT_PROMPT.format(loc=rtype)) or "").strip().lower()
        cat = next((o for o in _LLM_CAT_OPTIONS if o in out), None)
    except Exception:
        cat = None
    _LLM_CAT_CACHE[rtype] = cat
    return cat


def receptacle_categories_llm(text: str, llm_fn) -> set:
    """LLM-classified categories of the receptacle types mentioned in text."""
    return {c for c in (llm_receptacle_category(t, llm_fn)
                        for t in receptacle_types_in(text)) if c}


@dataclass
class SemanticFact:
    entity: str                      # the thing this fact is about (object/topic/widget)
    fact: str                        # one short transferable declarative sentence
    support: int = 1                 # how many episodes observed it (write-time count)
    ftype: str = "other"             # affordance | location | other (routing/gating)
    env: str = ""
    id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    embedding: Optional[list] = None  # cached embedding of "entity: fact"
    # ③ calibrated confidence: usefulness when actually retrieved (not just write count)
    used: int = 0                    # times retrieved into a task's context
    helped: int = 0                  # of those, times the task SUCCEEDED
    expected: float = 0.0            # running sum of the family base-rate at each use
                                     # (lets lift() separate "wrong fact" from "hard task")
    # paper-original WEIGHT-MODIFY channel (the variant the paper claims fails on
    # episodic memory): skill failure decays this multiplicative retrieval weight.
    # Default 1.0 = no-op unless MEMCRYST_SEMANTIC_WDECAY is set.
    weight: float = 1.0
    # ② provenance: links for the bidirectional skill<->memory ancestry graph
    from_episodes: list = field(default_factory=list)   # episode/task ids that observed it
    supports_skills: list = field(default_factory=list)  # skill ids it was co-retrieved with
    # skill-framed environmental verification (DIAGNOSTIC, MEMCRYST_FACT_VERIFY): when
    # the agent acts on this fact and the environment is observed, did the fact's claim
    # hold? Clean per-fact truth signal, INDEPENDENT of task success (unlike confidence).
    verified: int = 0                # episodes where the env confirmed the fact's claim
    refuted: int = 0                 # episodes where the env contradicted it
    # DIAGNOSTIC candidate signals (Jun29) — find one that tracks USEFULNESS, not just
    # truth (type-level verified/refuted decoupled from task-lift, r=-0.04).
    cat_verified: int = 0            # category-level (surface/container/...) confirm
    cat_refuted: int = 0
    find_cost_sum: float = 0.0       # search cost (go-tos to find entity) when this
    find_cost_n: int = 0             #   fact's entity was the TASK TARGET (target-cond)
    seen_types: dict = field(default_factory=dict)  # receptacle-type -> times entity
                                     #   observed there (the grounded correction target)
    # log-structured supersession (MEMCRYST_FACT_COR_MODE=supersede): a corrected
    # SUCCESSOR is appended and the predecessor shadowed from the read path —
    # content immutable, predecessor kept as anchor, rollback = un-shadow.
    prev_id: str = ""                # id of the fact this one supersedes ("" = original)
    shadowed: bool = False           # True = hidden from retrieval (superseded)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if self.ftype not in FACT_TYPES:
            self.ftype = "other"

    def confidence(self) -> float:
        """Laplace-smoothed success-rate-when-used. ~0.5 before any feedback, so
        unproven facts rank near neutral; proven-useful facts float up, proven-
        harmful facts sink. Used for retrieval ranking (③)."""
        return (self.helped + 1.0) / (self.used + 2.0)

    def lift(self) -> float:
        """Base-rate-RELATIVE usefulness: average (actual success − family base-rate)
        per use. lift > 0 -> the fact pushes tasks ABOVE the family baseline (genuinely
        useful); lift < 0 -> it drags tasks BELOW baseline (genuinely harmful). This
        separates a WRONG fact (lift<0) from a CORRECT fact on a HARD family (lift~0,
        low absolute confidence but at baseline). Returns 0.0 before any feedback."""
        if self.used == 0:
            return 0.0
        return (self.helped - self.expected) / self.used

    def refute_rate(self) -> float:
        tot = self.verified + self.refuted
        return self.refuted / tot if tot else 0.0

    def is_refuted_dominant(self, min_refuted: int = 3, min_rate: float = 0.6,
                            max_verified: int = 10 ** 9, use_category: bool = False) -> bool:
        """The environment has contradicted this fact's claim enough times (and
        mostly) that it should be quarantined from retrieval. Read-only: the fact
        stays in the store and re-enters if later verifications flip the balance.
        max_verified: only flag facts verified at most this many times.
        use_category: judge at COARSE category granularity (surface/container/...)
        instead of exact receptacle-type. Category refute-rate tracks task-usefulness
        (r=-0.40) far better than type-level (r=-0.12): it removes wrong-CATEGORY
        hallucinations ('bread in fridge') while sparing right-category-wrong-instance
        priors ('keychain on furniture-surface' — type-refuted but useful)."""
        ref = self.cat_refuted if use_category else self.refuted
        ver = self.cat_verified if use_category else self.verified
        tot = ref + ver
        rate = (ref / tot) if tot else 0.0
        return ref >= min_refuted and rate >= min_rate and ver <= max_verified

    def key_text(self) -> str:
        return f"{self.entity}: {self.fact}"

    def to_dict(self) -> dict:
        return {
            "entity": self.entity, "fact": self.fact, "support": self.support,
            "ftype": self.ftype, "env": self.env, "id": self.id, "created_at": self.created_at,
            "updated_at": self.updated_at, "embedding": self.embedding,
            "used": self.used, "helped": self.helped, "expected": self.expected,
            "weight": self.weight,
            "from_episodes": self.from_episodes, "supports_skills": self.supports_skills,
            "verified": self.verified, "refuted": self.refuted,
            "cat_verified": self.cat_verified, "cat_refuted": self.cat_refuted,
            "find_cost_sum": self.find_cost_sum, "find_cost_n": self.find_cost_n,
            "seen_types": self.seen_types,
            "prev_id": self.prev_id, "shadowed": self.shadowed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticFact":
        return cls(
            entity=d.get("entity", ""), fact=d.get("fact", ""),
            support=int(d.get("support", 1)), ftype=d.get("ftype", "other"),
            env=d.get("env", ""),
            id=d.get("id", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            embedding=d.get("embedding"),
            used=int(d.get("used", 0)), helped=int(d.get("helped", 0)),
            expected=float(d.get("expected", 0.0)), weight=float(d.get("weight", 1.0)),
            from_episodes=d.get("from_episodes", []) or [],
            supports_skills=d.get("supports_skills", []) or [],
            verified=int(d.get("verified", 0)), refuted=int(d.get("refuted", 0)),
            cat_verified=int(d.get("cat_verified", 0)), cat_refuted=int(d.get("cat_refuted", 0)),
            find_cost_sum=float(d.get("find_cost_sum", 0.0)), find_cost_n=int(d.get("find_cost_n", 0)),
            prev_id=d.get("prev_id", ""), shadowed=bool(d.get("shadowed", False)),
            seen_types=d.get("seen_types", {}) or {},
        )


class SemanticMemoryStore:
    """JSONL-backed store of entity-keyed declarative facts. Domain-agnostic."""

    def __init__(self, store_dir: Path, embed_fn: Optional[Callable[[str], np.ndarray]] = None,
                 autosave: bool = True) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "semantic_db.jsonl"
        self.embed_fn = embed_fn
        self.autosave = autosave
        self._facts: list[SemanticFact] = []
        self.load()

    # ----- persistence -----
    def load(self) -> None:
        self._facts.clear()
        if not self.db_path.exists():
            return
        for line in self.db_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._facts.append(SemanticFact.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                logger.warning("skipping malformed semantic fact line")

    def save(self) -> None:
        tmp = self.db_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for fact in self._facts:
                f.write(json.dumps(fact.to_dict()) + "\n")
        tmp.replace(self.db_path)

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self):
        return iter(self._facts)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embed_fn is None:
            return None
        v = np.asarray(self.embed_fn(text), dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ----- write: add with same-entity dedup/merge -----
    def add_fact(self, entity: str, fact: str, ftype: str = "other",
                 episode_id=None) -> SemanticFact:
        entity = (entity or "").strip().lower()
        fact = (fact or "").strip()
        if not entity or not fact:
            return None
        from memcryst.envscope import current_env, effective_env, scoped
        emb = self._embed(f"{entity}: {fact}")
        if emb is not None:
            for f in self._facts:
                if f.entity != entity or f.embedding is None:
                    continue
                if scoped() and effective_env(f.env) != current_env(content=True):
                    continue   # never merge observations across environments
                fv = np.asarray(f.embedding, dtype=float)
                if float(emb @ fv) >= MERGE_TAU:   # same fact seen again -> reinforce
                    f.support += 1
                    f.updated_at = time.time()
                    if episode_id is not None and episode_id not in f.from_episodes:
                        f.from_episodes.append(episode_id)   # ② provenance
                    if self.autosave:
                        self.save()
                    return f
        sf = SemanticFact(entity=entity, fact=fact, ftype=ftype,
                          env=current_env(content=True),
                          embedding=(emb.tolist() if emb is not None else None),
                          from_episodes=([episode_id] if episode_id is not None else []))
        self._facts.append(sf)
        if self.autosave:
            self.save()
        return sf

    def correct_refuted_facts(self, min_refuted: int = 3, min_rate: float = 0.6,
                              use_category: bool = True, multi_loc: bool = False) -> list:
        """GROUNDED in-place correction (not removal): for each location fact the
        environment has repeatedly refuted, rewrite its claimed location to where the
        entity was ACTUALLY observed most often (seen_types argmax). Keeps the fact
        count constant (no top-k churn / no scaffolding loss for weak backbones) while
        fixing the content. Re-embeds and resets the verification counters (the claim
        changed). This is an environment-GROUNDED rewrite, not an LLM hallucination.
        Returns [(entity, old_fact, new_fact)] for logging.

        MEMCRYST_FACT_COR_MODE=supersede (log-structured variant): instead of
        mutating the fact, append a corrected SUCCESSOR (prev_id lineage) and
        shadow the predecessor from the read path. The predecessor stays in the
        store as the audited anchor; if the successor itself becomes refuted-
        dominant, the read path falls back (un-shadow predecessor, shadow
        successor). Content is never regenerated in place."""
        mode = os.environ.get("MEMCRYST_FACT_COR_MODE", "rewrite").strip().lower()
        corrections = []
        _flipped = set()
        if mode == "supersede":
            # fallback pass FIRST: a bad correction gets caught by the same
            # refutation signal and the read path rolls back to the anchor.
            _by_id = {f.id: f for f in self._facts}
            for f in self._facts:
                if not f.prev_id or f.shadowed:
                    continue
                if not f.is_refuted_dominant(min_refuted, min_rate, use_category=use_category):
                    continue
                pred = _by_id.get(f.prev_id)
                if pred is None or not pred.shadowed:
                    continue
                f.shadowed, pred.shadowed = True, False
                _flipped.update((f.id, pred.id))
                corrections.append((f.entity, f.fact, f"[FALLBACK to] {pred.fact}"))
        for f in self._facts:
            if f.ftype != "location":
                continue
            if getattr(f, "shadowed", False) or f.id in _flipped:
                continue
            # skip facts whose "entity" is itself a receptacle (e.g. "countertop") —
            # rewriting those yields tautologies ("countertop found on a countertop").
            if f.entity in RECEPTACLE_TYPES:
                continue
            if not f.is_refuted_dominant(min_refuted, min_rate, use_category=use_category):
                continue
            if not f.seen_types:
                continue
            # only consider receptacle-type observations (drop non-receptacle noise)
            _cand = {t: c for t, c in f.seen_types.items() if t in RECEPTACLE_TYPES}
            if not _cand:
                continue
            ranked = sorted(_cand.items(), key=lambda kv: -kv[1])
            best = ranked[0][0]
            # already points at the right place? skip.
            if best in receptacle_types_in(f.fact):
                continue
            old = f.fact
            if multi_loc and len(ranked) > 1:
                # multi-location rewrite: merge the observed distribution into ONE
                # richer prior ("on a X or Y") instead of swapping to a single mode
                # that the next room randomization would re-refute. Addresses the
                # fact-fragmentation root cause (multi-mode entities).
                new_text = f"{f.entity} is often found on a {best} or {ranked[1][0]}"
            else:
                new_text = f"{f.entity} is often found on a {best}"
            if mode == "supersede":
                # already tried this exact successor before? leave the chain alone.
                if any(g.prev_id == f.id and g.fact == new_text for g in self._facts):
                    continue
                succ = SemanticFact(
                    entity=f.entity, fact=new_text, support=f.support,
                    ftype=f.ftype, env=f.env, prev_id=f.id,
                    from_episodes=list(f.from_episodes))
                _emb = self._embed(f"{succ.entity}: {succ.fact}")
                succ.embedding = _emb.tolist() if _emb is not None else None
                self._facts.append(succ)
                f.shadowed = True          # anchor kept, hidden from the read path
                f.updated_at = time.time()
                corrections.append((f.entity, old, new_text))
                continue
            f.fact = new_text
            _emb = self._embed(f"{f.entity}: {f.fact}")
            f.embedding = _emb.tolist() if _emb is not None else None
            # claim changed -> reset the (now stale) verification/usefulness counters
            f.verified = f.refuted = f.cat_verified = f.cat_refuted = 0
            f.seen_types = {}
            f.updated_at = time.time()
            corrections.append((f.entity, old, f.fact))
        if corrections and self.autosave:
            self.save()
        return corrections

    def add_many(self, triples, episode_id=None) -> int:
        """Accept (entity, fact) or (entity, fact, ftype) tuples."""
        prev_autosave = self.autosave
        self.autosave = False
        n = 0
        for t in triples:
            entity, fact = t[0], t[1]
            ftype = t[2] if len(t) > 2 else "other"
            if self.add_fact(entity, fact, ftype, episode_id=episode_id) is not None:
                n += 1
        self.autosave = prev_autosave
        if self.autosave:
            self.save()
        return n

    def get(self, fact_id: str) -> Optional[SemanticFact]:
        for f in self._facts:
            if f.id == fact_id:
                return f
        return None

    def prune_low_confidence(self, min_lift: float = -0.1, min_uses: int = 3) -> list:
        """★ Bidirectional attribution feedback (skill-failure -> repair memory),
        MEMCRYST's original unfilled idea. A fact retrieved enough times (>=min_uses)
        whose LIFT stays negative (lift<min_lift, i.e. tasks using it succeed BELOW
        the family base-rate) is a grounding fact that genuinely DRAGS the using
        skills down — so skill failures are attributed (via the ancestry edges) back
        to this fact and it is PRUNED. Base-rate-RELATIVE so a correct fact on a hard
        family (low absolute success but at baseline, lift~0) is NOT pruned; only
        genuinely harmful facts (lift<0) are. Removal only (no rewrite). Returns the
        removed facts (with the supports_skills they were repairing)."""
        removed = [f for f in self._facts
                   if f.used >= min_uses and f.lift() < min_lift]
        if removed:
            keep = {f.id for f in self._facts} - {f.id for f in removed}
            self._facts = [f for f in self._facts if f.id in keep]
            if self.autosave:
                self.save()
        return removed

    def record_outcome(self, fact_ids, success: bool, skill_ids=None,
                       weight_decay: float = 1.0, family_base: float = 0.0) -> None:
        """③ + ②④ feedback: after a task that retrieved these facts, update each
        fact's used/helped (calibrated confidence), the base-rate-relative lift signal
        (expected += family_base, the family's running success rate), and link the
        co-retrieved skills (supports_skills, the ancestry edge). weight_decay<1.0
        ALSO multiplies the retrieval weight on FAILURE (paper-original weight-MODIFY
        channel). family_base feeds lift() so pruning separates wrong-fact from hard-task."""
        skill_ids = skill_ids or []
        for fid in set(fact_ids or []):
            f = self.get(fid)
            if f is None:
                continue
            f.used += 1
            f.expected += family_base
            if success:
                f.helped += 1
            elif weight_decay < 1.0:
                f.weight *= weight_decay   # skill failure decays ancestry fact weight
            for sid in skill_ids:
                if sid not in f.supports_skills:
                    f.supports_skills.append(sid)
            f.updated_at = time.time()
        if self.autosave:
            self.save()


# Family -> the appliance its manipulation needs. Used by entity-profile retrieval
# to GUARANTEE the appliance's affordance facts surface even when the task text
# names only the object (fixes the "task says potato, affordance keyed on fridge"
# retrieval gap). Domain knowledge but tiny + ALFWorld-specific only here; the
# store/fact schema stay generic.
FAMILY_APPLIANCE = {"cool": "fridge", "heat": "microwave", "clean": "sinkbasin",
                    "examine": "desklamp"}


class SemanticMemoryRetriever:
    """Retrieve grounding facts for a task by semantic similarity of (entity: fact)
    to the task description. Returns a formatted block grouped by entity, with
    per-fact support as a lightweight confidence signal."""

    def __init__(self, store: SemanticMemoryStore,
                 embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> None:
        self.store = store
        self.embed_fn = embed_fn or store.embed_fn

    def retrieve_profiles(self, task_description: str, family: str = None,
                          top_k: int = 6, allowed_types=None,
                          min_support_location: int = 1) -> list[SemanticFact]:
        """① ENTITY-PROFILE retrieval (TriMem Tier-3). Instead of flat cosine top-k,
        assemble per-entity profiles for the entities RELEVANT to this task: every
        store entity whose name appears in the task text, PLUS the family's appliance.
        Guarantees the target object's location facts + the appliance's affordance
        facts are present together (a structured profile), fixing the flat-top-k gap.
        Confidence-ranked: affordance constraints first, then by confidence/support."""
        from memcryst.envscope import env_ok
        td = (task_description or "").lower()
        rel = set()
        for f in self.store:
            if not env_ok(getattr(f, "env", "")):
                continue
            if f.entity and f.entity in td:
                rel.add(f.entity)
        if family and family in FAMILY_APPLIANCE:
            rel.add(FAMILY_APPLIANCE[family])
        hits = []
        for f in self.store:
            if f.entity not in rel:
                continue
            if getattr(f, "shadowed", False):
                continue
            if not env_ok(getattr(f, "env", "")):
                continue
            if f.support < int(os.environ.get("MEMCRYST_FACT_MIN_SUPPORT", "1") or 1):
                continue
            if allowed_types is not None and f.ftype not in allowed_types:
                continue
            if f.ftype == "location" and f.support < min_support_location:
                continue
            hits.append(f)
        # affordance (hard constraints) first, then by confidence×weight then support
        hits.sort(key=lambda x: (x.ftype != "affordance",
                                 -x.confidence() * x.weight, -x.support))
        return hits[:top_k]

    def retrieve(self, task_description: str, top_k: int = 4,
                 allowed_types=None, min_support_location: int = 1,
                 boost_skills=None, skill_boost: float = 1.0,
                 quarantine=False, quarantine_penalty: float = 0.0,
                 quarantine_maxv: int = 10 ** 9,
                 quarantine_category: bool = False) -> list[SemanticFact]:
        """allowed_types: restrict to these fact types (e.g. {"affordance"}); None=all.
        min_support_location: drop location facts below this support (P2 pruning;
        affordance facts are NOT pruned — a single observed constraint is reliable).
        boost_skills (A — skill-conditioned retrieval): when the active skills for
        this task are passed, facts whose `supports_skills` ancestry edge intersects
        them get ranked up by `skill_boost` (>1.0). Read-only: lights up the skill→
        memory direction of the ancestry edge at retrieval time, no weight modify.
        quarantine: skip facts the environment has repeatedly refuted (skill-framed
        verification signal) — read-only, the fact stays in the store and re-enters
        if later verifications flip it. This is the CLEAN truth signal (env-grounded),
        not the task-success confidence that prune used."""
        facts = list(self.store)
        if not facts or self.embed_fn is None:
            return []
        q = np.asarray(self.embed_fn(task_description), dtype=float)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        _boost = set(boost_skills) if boost_skills else None
        from memcryst.envscope import env_ok
        # Support gate: a fact must recur across >= N independent episodes to be served.
        _minsup = int(os.environ.get("MEMCRYST_FACT_MIN_SUPPORT", "1") or 1)
        scored = []
        for f in facts:
            if f.embedding is None:
                continue
            if getattr(f, "shadowed", False):
                continue
            if not env_ok(getattr(f, "env", "")):
                continue
            if f.support < _minsup:
                continue
            if allowed_types is not None and f.ftype not in allowed_types:
                continue
            if f.ftype == "location" and f.support < min_support_location:
                continue
            _quar_penalty = 1.0
            if quarantine and f.is_refuted_dominant(max_verified=quarantine_maxv,
                                                    use_category=quarantine_category):
                if quarantine_penalty <= 0.0:
                    continue                       # HARD quarantine: skip entirely
                _quar_penalty = quarantine_penalty  # SOFT: down-rank, stay available
            fv = np.asarray(f.embedding, dtype=float)
            # Lexical overlap count (names/dates are lexical anchors; pure cosine
            # loses them in a 20k-fact pool — "Caroline" question must hit
            # "caroline: ..." facts). Consumed by HYBRID_LEX (additive bonus)
            # or HYBRID_RRF (rank fusion).
            _lexn = 0
            if os.environ.get("MEMCRYST_HYBRID_LEX", "0") not in ("", "0") \
                    or os.environ.get("MEMCRYST_HYBRID_RRF", "0") not in ("", "0"):
                _qt = set(task_description.lower().split())
                _ft = set((f.entity + " " + f.fact).lower().split())
                _lexn = len(_qt & _ft)
            # similarity × support × calibrated-confidence (③) × weight (paper-original
            # weight-modify channel, =1.0 no-op unless WDECAY on). confidence is ~0.5
            # before feedback (neutral); proven-useful facts rank up, harmful ones down.
            sim = (float(q @ fv) * (1.0 + 0.05 * min(f.support, 6))
                   * (0.5 + f.confidence()) * f.weight * _quar_penalty)
            # A: skill-conditioned boost — fact empirically co-retrieved with an
            # active skill is grounding this very procedure, so surface it.
            if _boost is not None and skill_boost != 1.0 and _boost & set(f.supports_skills):
                sim *= skill_boost
            scored.append((sim, _lexn, f))
        if os.environ.get("MEMCRYST_HYBRID_RRF", "0") not in ("", "0"):
            # MEMCRYST_HYBRID_RRF=1: reciprocal-rank fusion of the dense and
            # lexical channels — each ranked separately, fused 1/(k+rank).
            # Rank-based is dimensionless: no 0.12 magic weight, robust to
            # score-scale drift. k=60 is the Milvus/large-pool convention;
            # for our ~5k-fact banks with top_k~12 a smaller k (e.g. 20)
            # lets a lex-rank-1 hit rescue a dense-rank-30 true positive.
            _k = float(os.environ.get("MEMCRYST_RRF_K", "60") or 60)
            _dr = {f.id: i for i, (_, _, f) in enumerate(
                sorted(scored, key=lambda x: -x[0]))}
            _lr = {f.id: i for i, (_, _, f) in enumerate(
                sorted(scored, key=lambda x: -x[1]))}
            fused = [(1.0 / (_k + _dr[f.id]) + 1.0 / (_k + _lr[f.id]), f)
                     for _, _, f in scored]
            fused.sort(key=lambda x: -x[0])
            return [f for _, f in fused[:top_k]]
        if os.environ.get("MEMCRYST_HYBRID_LEX", "0") not in ("", "0"):
            scored = [(s + 0.12 * min(n, 4), n, f) for s, n, f in scored]
        scored.sort(key=lambda x: -x[0])
        return [f for _, _, f in scored[:top_k]]

    def format_block(self, facts: list[SemanticFact]) -> str:
        if not facts:
            return ""
        # group by entity so multiple location/attribute facts read as a prior
        by_entity: dict[str, list[SemanticFact]] = {}
        for f in facts:
            by_entity.setdefault(f.entity, []).append(f)
        lines = []
        for entity, fs in by_entity.items():
            fs.sort(key=lambda x: -x.support)
            parts = "; ".join(f"{x.fact} ({x.support}×)" for x in fs)
            lines.append(f"- {entity}: {parts}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Write-time extraction: trajectory -> declarative grounding facts (LLM).      #
# Generic prompt — works for ALFWorld, WebShop, web/API agents alike.          #
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT_QA = """You are building a PERSONAL-FACTS memory for an agent that will later answer questions about specific people and events in the conversations it reads.

Below is a conversation session the agent just read.

Task: {task}
Outcome: {outcome}
Session:
{trajectory}

Extract up to 25 memory facts about the NAMED speakers — BOTH kinds:
1. persistent facts (jobs, relationships, preferences, plans, possessions)
2. notable events and specific details from THIS session (what happened, dates,
   items, gifts, food, places, activities, feelings about events)

DATE ACCURACY (critical): the session header date is when the CONVERSATION happened, not when events happened. If a speaker says "yesterday"/"last week", RESOLVE it against the session date and record the EVENT date; include the date in the fact for every event.

Rules — the OPPOSITE of generalization:
- KEEP names, dates, places, numbers VERBATIM. "Caroline adopted a puppy in May 2023" is perfect; "a person can adopt a pet" is USELESS.
- entity = the person's NAME (or the specific thing discussed).

One fact per line, format exactly:  entity | fact
No JSON, no numbering, nothing else."""

_EXTRACT_PROMPT_SUCCESS = """You are building a reusable WORLD-MODEL for an agent that will face MANY future tasks in this same environment.

Below is one task the agent just SOLVED and (a summary of) its trajectory.

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Extract 0-3 TRANSFERABLE FACTS about the ENVIRONMENT that would help a DIFFERENT future task — regularities, NOT the step-by-step procedure of this task. Tag each fact with its TYPE:
- "location": where an object/item TYPE tends to be found (spatial prior). e.g. "alarmclock often on a desk, shelf, or sidetable".
- "affordance": a constraint/property of how something WORKS that must be respected. e.g. "microwave must be opened before an object can be heated"; "fridge contains food items".

Rules:
- DECLARATIVE world-knowledge ("knowing THAT"), never a procedure ("knowing HOW"). Bad: "go to the cabinet then take the mug."
- ONE ENTITY per fact — a GENERIC type, never an instance id ("mug", not "mug 2"; "desk", not "desk 1").
- ONE short general sentence, no numbers/ids/coordinates from this episode.
- Only TRUE-IN-GENERAL, USEFUL facts. Nothing transferable -> empty list.

Output ONLY a JSON list, e.g.:
[{{"entity": "alarmclock", "fact": "often located on a desk, shelf, or sidetable", "type": "location"}}, {{"entity": "microwave", "fact": "must be opened before an object can be heated", "type": "affordance"}}]
"""

_EXTRACT_PROMPT_FAILURE = """An agent FAILED the task below. Failures are the best source of AFFORDANCE constraints — rules about how the environment works that, when violated, cause failure.

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Extract 0-2 TRANSFERABLE AFFORDANCE FACTS — constraints/properties that a future task must respect. e.g. "microwave must be opened before heating"; "an object must be picked up before it can be cleaned"; "a closed container must be opened before taking from it".

Rules:
- ONLY affordance/constraint facts (how things WORK), NOT locations (a failed run's location observations are unreliable) and NOT procedures.
- ONE ENTITY per fact, GENERIC type, ONE short general sentence, no instance ids/numbers.
- State it as a general TRUE rule, not "the agent should have...". If the failure reveals no general constraint, output an empty list.

Output ONLY a JSON list, e.g.:
[{{"entity": "microwave", "fact": "must be opened before an object can be heated", "type": "affordance"}}]
"""


# demand-driven capture (MEMCRYST_CAPTURE_DEMAND=1): entities from questions the
# agent FAILED to answer, injected into the QA extraction prompt so writeback
# prioritizes capturing exactly what the query stream keeps asking for.
# Selection acting on the WRITE path: skills/queries teach memory what to look for.
_CAPTURE_DEMAND: list = []


def add_capture_demand(entities) -> None:
    """Record entities from a failed question (most-recent-first, deduped, cap 40)."""
    global _CAPTURE_DEMAND
    for e in entities:
        if e in _CAPTURE_DEMAND:
            _CAPTURE_DEMAND.remove(e)
        _CAPTURE_DEMAND.insert(0, e)
    del _CAPTURE_DEMAND[40:]


def extract_grounding_facts(task_description: str, outcome: str, trajectory_text: str,
                            llm, success: bool = True) -> list[tuple]:
    """LLM-distill a trajectory into (entity, fact, type) declarative grounding facts.
    success episodes -> location + affordance facts; failure episodes -> ONLY
    affordance constraints (the lesson of the failure). Returns [] on any failure."""
    _qa_style = os.environ.get("MEMCRYST_FACT_STYLE", "").strip().lower() == "qa"
    if _qa_style:
        tmpl = _EXTRACT_PROMPT_QA        # instance IS the content: keep names/dates
    else:
        tmpl = _EXTRACT_PROMPT_SUCCESS if success else _EXTRACT_PROMPT_FAILURE
    prompt = tmpl.format(task=task_description, outcome=outcome, trajectory=trajectory_text)
    if _qa_style and _CAPTURE_DEMAND and os.environ.get("MEMCRYST_CAPTURE_DEMAND", "0") == "1":
        prompt += ("\n\nPRIORITY ENTITIES: earlier questions about these could not be "
                   "answered from memory. If this session mentions any of them, extract "
                   "those facts FIRST, keeping exact names, dates and numbers: "
                   + ", ".join(_CAPTURE_DEMAND))
    try:
        _sys = ("You extract personal memory facts, one per line as 'entity | fact'. Output only fact lines."
                if _qa_style else
                "You extract transferable world-facts as a JSON list. Output JSON only.")
        raw = llm.chat(
            [{"role": "system", "content": _sys},
             {"role": "user", "content": prompt}],
            max_tokens=900 if _qa_style else 300, temperature=0.2,
        )
    except Exception as exc:
        logger.warning("extract_grounding_facts LLM call failed: %s", exc)
        return []
    if _qa_style:
        # line format "entity | fact" (lite-harness port): immune to the
        # truncated-JSON failure mode; a cut-off final line just drops one fact
        out = []
        for line in raw.splitlines():
            if "|" not in line:
                continue
            e, _, f = line.partition("|")
            e, f = e.strip(" -*\t"), f.strip()
            if e and f and len(e) < 80:
                out.append((e, f, "other"))
        return out[:25]
    # tolerant JSON parse: find the first [...] block
    try:
        start = raw.index("[")
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    data = json.loads(raw[start:i + 1])
                    break
        else:
            return []
    except (ValueError, json.JSONDecodeError):
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            e = str(item.get("entity", "")).strip()
            f = str(item.get("fact", "")).strip()
            t = str(item.get("type", "other")).strip().lower()
            if t not in FACT_TYPES:
                t = "affordance" if not success else "other"
            if e and f:
                out.append((e, f, t))
    # failures only ever contribute affordance facts
    if not success:
        out = [(e, f, "affordance") for e, f, _ in out]
    return out[:12] if _qa_style else out[:3]
