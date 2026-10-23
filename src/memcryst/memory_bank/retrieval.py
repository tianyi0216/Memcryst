"""Memory retrieval. Embedding plus weight ranked cosine similarity.

We use sentence-transformers all-MiniLM-L6-v2 by default because all major
memory baselines we benchmark against use it. The embedding model is loaded
lazily so import time stays cheap.

Retrieval score combines cosine similarity with the entry weight. The formula
is the standard one.

    score(entry, query) = weight(entry) * cosine_sim(emb(entry), emb(query))
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .persistence import MemoryStore
from .schema import EntryStatus, MemoryEntry

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}   # one embedder per model name, process-wide


DEFAULT_EMBEDDING_MODEL = os.environ.get("MEMCRYST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class MemoryRetriever:
    """Embedding based retriever with weight ranked cosine similarity."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        verified_boost: float = 0.0,
        pair_stats: Optional[dict] = None,
        pair_boost: float = 0.0,
        pair_partner_skill_ids: Optional[list[str]] = None,
    ) -> None:
        self.store = store
        self.embedding_model_name = embedding_model
        self._model = None
        # N10 verified_boost: add gamma * log(1 + verified_uses) to the score
        # so memories that have helped the agent succeed rank higher.
        self.verified_boost = verified_boost
        # N9 pair_stats: dict keyed by (skill_id, memory_id) → (uses, succ).
        # When retrieval is invoked for a query that has retrieved skills
        # known via pair_partner_skill_ids, we add a bonus for memories
        # whose pair success rate with any of those skills is high.
        self.pair_stats = pair_stats or {}
        self.pair_boost = pair_boost
        self.pair_partner_skill_ids = pair_partner_skill_ids or []

    # ----- embedding -----

    def _get_model(self):
        if self._model is None:
            # module-level cache: gap-gate rebuilds retrievers EVERY cycle; a
            # per-instance model leaked one ST copy per cycle onto the GPU and
            # OOM'd aflow_gate2 twice at cycle 15-19. One model per name, shared.
            m = _MODEL_CACHE.get(self.embedding_model_name)
            if m is None:
                from sentence_transformers import SentenceTransformer
                m = SentenceTransformer(self.embedding_model_name)
                _MODEL_CACHE[self.embedding_model_name] = m
            self._model = m
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([text], show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        mat = model.encode(texts, show_progress_bar=False)
        return np.asarray(mat, dtype=np.float32)

    def ensure_embeddings(self) -> int:
        """Backfill embeddings for every active entry that lacks one.

        Returns the count of entries that got a new embedding. We toggle the
        store autosave off during the loop so the whole JSONL is rewritten
        once at the end instead of per entry. Saves O(N^2) disk churn on
        cold start.
        """
        missing = [
            e for e in self.store
            if e.embedding is None and e.status != EntryStatus.ARCHIVED
        ]
        if not missing:
            return 0
        texts = [e.text_for_embedding() for e in missing]
        vecs = self.embed_many(texts)
        prev_autosave = self.store.autosave
        self.store.autosave = False
        try:
            for entry, vec in zip(missing, vecs):
                self.store.set_embedding(entry.id, vec.tolist())
        finally:
            self.store.autosave = prev_autosave
        self.store.save()
        return len(missing)

    # ----- retrieval -----

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        include_flagged: bool = False,
        include_archived: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return the top k entries ranked by weight times cosine similarity."""
        self.ensure_embeddings()
        q_vec = self.embed(query)

        scored: list[tuple[MemoryEntry, float]] = []
        from memcryst.envscope import env_ok
        for entry in self.store:
            if not env_ok(getattr(entry, "env", "")):
                continue
            if entry.status == EntryStatus.ARCHIVED and not include_archived:
                continue
            if entry.status == EntryStatus.FLAGGED and not include_flagged:
                continue
            if entry.embedding is None:
                continue
            e_vec = np.asarray(entry.embedding, dtype=np.float32)
            sim = self._cosine(q_vec, e_vec)
            # MEMCRYST_HYBRID_LEX=1: lexical anchor bonus (names/dates), same
            # rationale as the semantic store — excerpt entries live here.
            import os as _os
            if _os.environ.get("MEMCRYST_HYBRID_LEX", "0") not in ("", "0"):
                _qt = set(query.lower().split())
                _et = set((entry.question + " " + entry.improvement).lower().split())
                sim += 0.12 * min(len(_qt & _et), 4)   # ADDITIVE
            # Include per-memory utility EMA in the score (N4 dual-utility).
            # utility defaults to 1.0 so this is a no-op when N4 is disabled.
            score = entry.weight * entry.utility * sim
            # N10 verified rank boost: log-scaled bonus for entries that
            # have helped the agent succeed.
            if self.verified_boost > 0 and entry.verified_uses > 0:
                score += self.verified_boost * np.log(1.0 + entry.verified_uses)
            # N9 pair-tracking bonus: if any retrieved skill has a high
            # historical success rate with this memory entry, boost score.
            if self.pair_boost > 0 and self.pair_partner_skill_ids:
                best_pair_score = 0.0
                for sid in self.pair_partner_skill_ids:
                    stats = self.pair_stats.get((sid, entry.id))
                    if stats and stats[0] >= 2:  # min 2 co-uses
                        sr = stats[1] / stats[0]
                        if sr > best_pair_score:
                            best_pair_score = sr
                score += self.pair_boost * best_pair_score
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def embeddings_matrix(self, active_only: bool = True) -> tuple[list[MemoryEntry], np.ndarray]:
        """Return aligned lists of entries and their embedding rows.

        Used by the pattern miner to feed embeddings into HDBSCAN.
        """
        self.ensure_embeddings()
        rows = []
        entries: list[MemoryEntry] = []
        for entry in self.store:
            if active_only and entry.status != EntryStatus.ACTIVE:
                continue
            if entry.embedding is None:
                continue
            entries.append(entry)
            rows.append(entry.embedding)
        if not rows:
            return [], np.zeros((0, 384), dtype=np.float32)
        mat = np.asarray(rows, dtype=np.float32)
        return entries, mat
