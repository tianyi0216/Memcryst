"""memory_condense baseline.

AWM / Live-Evo style memory compression. Takes a raw memory bank, clusters
similar entries (using the same M1 PatternMiner as MEMCRYST so the clustering
step is comparable), and asks an LLM to summarize each cluster into one NL
paragraph. The output is stored as plain summary records, NOT as structured
skills. There is no capability tag, no procedure section, no 9+3 audit, no
ancestry feedback loop.

The purpose is to answer the question: does the value of MEMCRYST come
from the crystallization-into-structured-skill step, or just from clustering
plus summarization? memory_condense gives you cluster + summarize without
the structure or feedback.

Retrieval is plain cosine over the summary text. Same retrieval cost as
memory_raw, but the items retrieved are pre-summarized rather than raw
memory entries.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import LLMClient
from ..memory_bank import MemoryEntry, MemoryRetriever, MemoryStore
from ..memory_bank.retrieval import DEFAULT_EMBEDDING_MODEL
from ..sleep.m1_miner import ClusterCandidate, PatternMiner

logger = logging.getLogger(__name__)


@dataclass
class CondensedEntry:
    """One condensed memory record. NL summary of a cluster of raw entries."""

    id: str
    summary: str
    source_entry_ids: list[str] = field(default_factory=list)
    cluster_size: int = 0
    cluster_id: int = -1
    embedding: Optional[list[float]] = None
    created_at: float = field(default_factory=time.time)

    def text_for_embedding(self) -> str:
        return self.summary


# Prompt used to summarize a cluster of raw memory entries into a single
# NL paragraph. The prompt is intentionally minimal: no requirement for
# YAML, no skill structure, no procedure section. Just "summarize the
# common lesson across these experiences into one paragraph."
_CONDENSE_PROMPT = """Summarize the following past experiences into ONE concise paragraph that captures the common lesson and the gist of what worked. The summary will be retrieved later when the agent faces a similar task, so include the kind of task and the key successful steps in plain natural language.

Do NOT use any structured format. Do NOT use bullet points. Do NOT include section headers like Procedure or Anti patterns. Just one paragraph of natural language.

The output should be under 200 words.

Experiences in this cluster:

{episodes}

Output the summary paragraph only, no preamble."""


def _format_cluster_for_summary(cluster: ClusterCandidate) -> str:
    pieces = []
    for i, entry in enumerate(cluster.members):
        lines = [f"Experience {i + 1}:"]
        lines.append(f"  question: {entry.question}")
        if entry.improvement:
            lines.append(f"  improvement: {entry.improvement.strip()}")
        if entry.failure_reason:
            lines.append(f"  failure_reason: {entry.failure_reason.strip()}")
        pieces.append("\n".join(lines))
    return "\n\n".join(pieces)


def _summarize_cluster(cluster: ClusterCandidate, llm: LLMClient) -> str:
    episodes = _format_cluster_for_summary(cluster)
    prompt = _CONDENSE_PROMPT.format(episodes=episodes)
    messages = [
        {
            "role": "system",
            "content": "You produce concise natural-language summaries of clustered past experiences. No structure, no headers, no bullets.",
        },
        {"role": "user", "content": prompt},
    ]
    raw = llm.chat(messages, max_tokens=500, temperature=0.3)
    return raw.strip()


class CondensedMemoryStore:
    """JSONL backed store of condensed memory entries."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else Path.home() / ".memcryst" / "condense"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "condensed.jsonl"
        self._entries: dict[str, CondensedEntry] = {}
        self.load()

    def load(self) -> None:
        self._entries.clear()
        if not self.db_path.exists():
            return
        with self.db_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                e = CondensedEntry(**data)
                self._entries[e.id] = e

    def save(self) -> None:
        with self.db_path.open("w") as f:
            for e in self._entries.values():
                f.write(json.dumps(asdict(e)) + "\n")

    def add(self, entry: CondensedEntry) -> None:
        self._entries[entry.id] = entry
        self.save()

    def all(self) -> list[CondensedEntry]:
        return list(self._entries.values())

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {
            "total": len(self),
            "with_embedding": sum(1 for e in self if e.embedding is not None),
            "store_dir": str(self.store_dir),
        }


class CondensedMemoryRetriever:
    """Cosine retrieval over condensed memory summaries."""

    def __init__(
        self,
        store: CondensedMemoryStore,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.store = store
        self.embedding_model_name = embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        vec = self._get_model().encode([text], show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)

    def ensure_embeddings(self) -> int:
        missing = [e for e in self.store if e.embedding is None]
        if not missing:
            return 0
        texts = [e.text_for_embedding() for e in missing]
        vecs = self._get_model().encode(texts, show_progress_bar=False)
        for e, v in zip(missing, vecs):
            e.embedding = np.asarray(v, dtype=np.float32).tolist()
        self.store.save()
        return len(missing)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[CondensedEntry, float]]:
        self.ensure_embeddings()
        if len(self.store) == 0:
            return []
        q_vec = self.embed(query)
        scored = []
        for e in self.store:
            if e.embedding is None:
                continue
            e_vec = np.asarray(e.embedding, dtype=np.float32)
            scored.append((e, self._cosine(q_vec, e_vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def build_condensed_memory(
    memory_store: MemoryStore,
    llm: LLMClient,
    output_dir: Path,
    min_cluster_size: int = 3,
    min_mean_reward: float = 0.4,
    include_archived: bool = True,
) -> CondensedMemoryStore:
    """End-to-end: cluster the memory bank with M1, summarize each cluster
    with the LLM, store as a CondensedMemoryStore at output_dir.

    No skill structure is added. No audit gate. Just cluster + summarize.

    include_archived: if True (default), restore ARCHIVED entries to ACTIVE
    before clustering. This is essential when the memory bank has already
    been crystallized once (sleep archives the source entries). Without
    this, M1 only sees the post-crystallization residue and produces
    incoherent clusters. The restoration is in-memory only; the source
    memory store on disk is not modified.
    """
    import hashlib

    from ..memory_bank.schema import EntryStatus

    output_dir.mkdir(parents=True, exist_ok=True)
    condensed_store = CondensedMemoryStore(store_dir=output_dir)

    if include_archived:
        restored = 0
        for entry in memory_store:
            if entry.status == EntryStatus.ARCHIVED:
                entry.status = EntryStatus.ACTIVE
                restored += 1
        if restored:
            logger.info(
                "memory_condense: temporarily restored %d archived entries to ACTIVE for clustering",
                restored,
            )

    retriever = MemoryRetriever(memory_store)
    miner = PatternMiner(
        retriever,
        min_cluster_size=min_cluster_size,
        min_mean_reward=min_mean_reward,
    )
    clusters = miner.mine()
    logger.info("memory_condense: M1 produced %d clusters", len(clusters))

    for cluster in clusters:
        summary = _summarize_cluster(cluster, llm)
        seed = f"cluster-{cluster.cluster_id}-{summary[:60]}"
        cid = hashlib.md5(seed.encode()).hexdigest()[:8]
        entry = CondensedEntry(
            id=cid,
            summary=summary,
            source_entry_ids=[e.id for e in cluster.members],
            cluster_size=cluster.size,
            cluster_id=cluster.cluster_id,
        )
        condensed_store.add(entry)
        logger.info(
            "memory_condense: cluster %s size %d → condensed entry %s (%d chars)",
            cluster.cluster_id,
            cluster.size,
            cid,
            len(summary),
        )

    return condensed_store
