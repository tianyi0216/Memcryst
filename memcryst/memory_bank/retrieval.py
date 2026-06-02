"""Memory retrieval. Embedding plus weight ranked cosine similarity.

We use sentence-transformers all-MiniLM-L6-v2 by default because all major
memory baselines we benchmark against use it. The embedding model is loaded
lazily so import time stays cheap.

Retrieval score combines cosine similarity with the entry weight. The formula
is the same Live-Evo uses.

    score(entry, query) = weight(entry) * cosine_sim(emb(entry), emb(query))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .persistence import MemoryStore
from .schema import EntryStatus, MemoryEntry

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MemoryRetriever:
    """Embedding based retriever with weight ranked cosine similarity."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.store = store
        self.embedding_model_name = embedding_model
        self._model = None

    # ----- embedding -----

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
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
        for entry in self.store:
            if entry.status == EntryStatus.ARCHIVED and not include_archived:
                continue
            if entry.status == EntryStatus.FLAGGED and not include_flagged:
                continue
            if entry.embedding is None:
                continue
            e_vec = np.asarray(entry.embedding, dtype=np.float32)
            sim = self._cosine(q_vec, e_vec)
            score = entry.weight * sim
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
