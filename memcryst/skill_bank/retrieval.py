"""Skill retrieval.

Two retrieval paths:

1. Capability-indexed retrieval (preferred). Extract a closed-set capability
   signature from the task NL, hash-lookup the matching skills, optionally
   rerank by description embedding. O(K) where K is the bucket size.
2. Legacy similarity retrieval. Cosine on the description embedding over all
   skills, O(N). Kept as a fallback for skills without a complete capability
   signature and as a baseline for benchmark comparison.

We embed the name plus description rather than the full body so retrieval
matches the trigger condition rather than the implementation detail.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core import LLMClient
from .capability_index import TaskSignatureExtractor
from .persistence import SkillStore
from .schema import Skill

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SkillRetriever:
    def __init__(
        self,
        store: SkillStore,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        signature_llm: Optional[LLMClient] = None,
    ) -> None:
        self.store = store
        self.embedding_model_name = embedding_model
        self._model = None
        self.signature_extractor = TaskSignatureExtractor(llm=signature_llm)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        return np.asarray(model.encode([text], show_progress_bar=False)[0], dtype=np.float32)

    def ensure_embeddings(self) -> int:
        missing = [s for s in self.store if s.embedding is None]
        if not missing:
            return 0
        model = self._get_model()
        texts = [s.text_for_embedding() for s in missing]
        vecs = model.encode(texts, show_progress_bar=False)
        for skill, vec in zip(missing, vecs):
            skill.embedding = np.asarray(vec, dtype=np.float32).tolist()
            self.store.save_one(skill)
        return len(missing)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Skill, float]]:
        """Legacy O(N) similarity retrieval. Kept as a fallback and baseline."""
        if len(self.store) == 0:
            return []
        self.ensure_embeddings()
        q_vec = self.embed(query)
        scored: list[tuple[Skill, float]] = []
        for skill in self.store:
            if skill.embedding is None:
                continue
            s_vec = np.asarray(skill.embedding, dtype=np.float32)
            sim = self._cosine(q_vec, s_vec)
            score = max(0.05, skill.utility) * sim
            scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_by_capability(
        self,
        query: str,
        top_k: int = 3,
        rerank_embeddings: bool = True,
        relax_affordance_if_empty: bool = True,
        fallback_similarity: bool = False,
    ) -> list[tuple[Skill, float, dict]]:
        """Capability-indexed retrieval. O(K) where K is the bucket size.

        Strategy:
          1. Extract task signature from the query NL.
          2. Hash-lookup the matching bucket. If empty, relax affordance axis
             and try again, then verb+affordance.
          3. If fallback_similarity is True AND nothing matched even after
             relaxation, fall through to legacy similarity retrieval. Default
             is False because retrieving an unrelated skill via fallback
             distracts the agent more than no skill at all.
          4. If rerank_embeddings, score the bucket by description embedding
             similarity to the query. Otherwise sort by skill utility.

        Returns list of (skill, score, meta) where meta describes which
        retrieval path was used so callers can log it. Empty list if no
        capability bucket matched and fallback_similarity is False.
        """
        if len(self.store) == 0:
            return []

        cap = self.signature_extractor.extract(query)
        sig = cap.signature()
        used_path = "exact"
        ids = self.store.capability_index.lookup(sig)

        if not ids and relax_affordance_if_empty:
            ids = self.store.capability_index.lookup_relaxed(sig, relax_axes={"affordance"})
            used_path = "relaxed_affordance" if ids else used_path

        if not ids:
            ids = self.store.capability_index.lookup_relaxed(
                sig, relax_axes={"affordance", "verb_family"}
            )
            used_path = "relaxed_verb_affordance" if ids else used_path

        if not ids:
            if fallback_similarity:
                fb = self.retrieve(query, top_k=top_k)
                return [
                    (skill, score, {"path": "fallback_similarity", "signature": sig})
                    for skill, score in fb
                ]
            # No skill matched. Return empty list so the agent runs without
            # a skill block this step, which is strictly better than being
            # distracted by an unrelated skill via cosine similarity.
            return []

        # Rerank the bucket. With small K this is cheap.
        candidates = [self.store.get(sid) for sid in ids]
        candidates = [s for s in candidates if s is not None]

        if rerank_embeddings:
            self.ensure_embeddings()
            q_vec = self.embed(query)
            scored: list[tuple[Skill, float]] = []
            for skill in candidates:
                if skill.embedding is None:
                    score = max(0.05, skill.utility)
                else:
                    s_vec = np.asarray(skill.embedding, dtype=np.float32)
                    sim = self._cosine(q_vec, s_vec)
                    score = max(0.05, skill.utility) * sim
                scored.append((skill, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                (skill, score, {"path": used_path, "signature": sig})
                for skill, score in scored[:top_k]
            ]

        ranked = sorted(candidates, key=lambda s: s.utility, reverse=True)
        return [
            (skill, skill.utility, {"path": used_path, "signature": sig})
            for skill in ranked[:top_k]
        ]
