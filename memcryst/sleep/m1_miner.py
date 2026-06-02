"""M1 pattern miner. HDBSCAN over memory entry embeddings.

We cluster the active subset of the memory bank using density based methods.
Clusters with size and mean reward above configured thresholds are crystal
candidates. The miner returns candidates sorted by a difficulty score that
borrows from MemSkill.

    d(C) = mean_reward(C) * size(C) * density(C)

The density factor is the cluster persistence value returned by HDBSCAN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..memory_bank import MemoryEntry, MemoryRetriever

logger = logging.getLogger(__name__)


@dataclass
class ClusterCandidate:
    cluster_id: int
    members: list[MemoryEntry]
    mean_reward: float
    persistence: float
    category_mode: str = "general"
    centroid: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def difficulty_score(self) -> float:
        # Prefer larger clusters with higher mean reward and tighter density.
        return self.mean_reward * self.size * max(0.05, self.persistence)


def _reward_proxy(entry: MemoryEntry) -> float:
    """Map a memory entry to a reward proxy in [0, 1].

    Live-Evo entries do not carry a numeric reward. We use weight as a proxy
    because higher weight entries are the ones the agent has found useful.
    A weight greater than 1 is treated as fully reward positive.
    """
    return max(0.0, min(1.0, entry.weight / 1.5))


def _category_mode(entries: list[MemoryEntry]) -> str:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    if not counts:
        return "general"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# Verb to task-family map. Used as a post-HDBSCAN purity check to split
# clusters where sentence embedding pulled multiple task families together.
# Sentence-BERT embeds "clean the X and place at Y" and "heat the X and place
# at Y" close because the "and place at" suffix dominates, but the procedural
# shape is different. We split here.
_TASK_FAMILY_MAP: dict[str, str] = {
    "clean": "clean",
    "wash": "clean",
    "rinse": "clean",
    "scrub": "clean",
    "heat": "heat",
    "microwave": "heat",
    "warm": "heat",
    "cook": "heat",
    "boil": "heat",
    "cool": "cool",
    "chill": "cool",
    "freeze": "cool",
    "refrigerate": "cool",
    "examine": "examine",
    "inspect": "examine",
    "look": "examine",
    "view": "examine",
    "read": "examine",
    "place": "place",
    "put": "place",
    "move": "place",
    "bring": "place",
    "pick": "pickup",
    "take": "pickup",
    "grab": "pickup",
    "retrieve": "pickup",
    "find": "search",
    "locate": "search",
    "search": "search",
}


def _task_family(entry: MemoryEntry) -> str:
    """Canonical task family for verb-purity split.

    Delegates to constants.detect_task_verb_family which handles the
    "put a <modifier> X" pattern. Falls back to the local map if the
    constant-level detector returns None.
    """
    from ..skill_bank.constants import detect_task_verb_family

    fam = detect_task_verb_family(entry.question or "")
    if fam is not None:
        return fam
    text = (entry.question or "").strip().lower()
    if not text:
        return "unknown"
    first_word = text.split()[0].rstrip(".,!?;:")
    return _TASK_FAMILY_MAP.get(first_word, first_word)


def _split_by_family(
    members: list[MemoryEntry],
    min_subcluster_size: int,
) -> list[list[MemoryEntry]]:
    """Group cluster members by task family. Return one sub-list per family
    that meets the minimum size. Members in families below the threshold are
    dropped, not merged into a misc bucket, so the resulting sub-clusters
    stay procedurally coherent."""
    by_family: dict[str, list[MemoryEntry]] = {}
    for entry in members:
        fam = _task_family(entry)
        by_family.setdefault(fam, []).append(entry)
    if len(by_family) <= 1:
        return [members]
    qualified = [grp for grp in by_family.values() if len(grp) >= min_subcluster_size]
    if not qualified:
        # All sub-clusters are too small. Keep the original cluster intact so
        # the synthesizer still sees something. Better than dropping all.
        return [members]
    return qualified


class PatternMiner:
    """Cluster the memory bank into crystallization candidates."""

    def __init__(
        self,
        retriever: MemoryRetriever,
        min_cluster_size: int = 3,
        min_samples: Optional[int] = 1,
        min_mean_reward: float = 0.3,
        cluster_selection_epsilon: float = 0.0,
    ) -> None:
        # min_samples defaults to 1 so HDBSCAN behaves reasonably on small
        # banks. The default of None uses min_cluster_size which is too strict
        # for the PoC bank sizes we operate on. Pass min_samples explicitly
        # to override.
        self.retriever = retriever
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.min_mean_reward = min_mean_reward
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def mine(self) -> list[ClusterCandidate]:
        entries, matrix = self.retriever.embeddings_matrix(active_only=True)
        if len(entries) < self.min_cluster_size:
            logger.info(
                "not enough entries for clustering, have %d need %d",
                len(entries),
                self.min_cluster_size,
            )
            return []

        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError(
                "hdbscan is required for the pattern miner, pip install hdbscan"
            ) from exc

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(matrix)
        persistences = getattr(clusterer, "cluster_persistence_", None)

        # Group entries by cluster label, skipping the noise label of -1.
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label < 0:
                continue
            groups.setdefault(label, []).append(idx)

        candidates: list[ClusterCandidate] = []
        next_subcluster_id = 1000  # sub-cluster ids live above raw HDBSCAN labels
        for label, indices in groups.items():
            raw_members = [entries[i] for i in indices]
            raw_mean = float(np.mean([_reward_proxy(m) for m in raw_members]))
            if raw_mean < self.min_mean_reward:
                continue

            # Purity check. Split this HDBSCAN cluster by task family if it
            # mixes multiple procedurally distinct families.
            sub_member_lists = _split_by_family(raw_members, self.min_cluster_size)

            for sub_idx, sub_members in enumerate(sub_member_lists):
                sub_mean = float(np.mean([_reward_proxy(m) for m in sub_members]))
                if sub_mean < self.min_mean_reward:
                    continue
                sub_indices = [entries.index(m) for m in sub_members]
                sub_centroid = matrix[sub_indices].mean(axis=0)
                base_persistence = (
                    float(persistences[label])
                    if persistences is not None and label < len(persistences)
                    else 0.5
                )
                # Identify whether this is a split from a multi-family cluster.
                is_split = len(sub_member_lists) > 1
                if is_split:
                    cluster_id = next_subcluster_id
                    next_subcluster_id += 1
                    logger.info(
                        "purity split: HDBSCAN cluster %s family %s -> sub-cluster %d size %d",
                        label,
                        _task_family(sub_members[0]),
                        cluster_id,
                        len(sub_members),
                    )
                else:
                    cluster_id = label
                candidates.append(
                    ClusterCandidate(
                        cluster_id=cluster_id,
                        members=sub_members,
                        mean_reward=sub_mean,
                        persistence=base_persistence,
                        category_mode=_category_mode(sub_members),
                        centroid=sub_centroid,
                    )
                )

        candidates.sort(key=lambda c: c.difficulty_score, reverse=True)
        logger.info(
            "M1 mined %d candidate clusters from %d active entries",
            len(candidates),
            len(entries),
        )
        return candidates
