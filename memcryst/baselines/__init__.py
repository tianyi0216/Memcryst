"""Baselines for MEMCRYST comparison.

These are NOT MEMCRYST mechanisms. They are competing approaches we
compare against in the paper.

- condense: AWM / Live-Evo style. Cluster raw memory entries then LLM
  summarize each cluster into a single NL paragraph. NO skill structure,
  NO capability tags, NO procedure / anti-pattern sections, NO 9+3 audit,
  NO ancestry feedback. Just cluster + summarize + cosine retrieval.
"""

from .condense import (
    CondensedEntry,
    CondensedMemoryRetriever,
    CondensedMemoryStore,
    build_condensed_memory,
)

__all__ = [
    "CondensedEntry",
    "CondensedMemoryStore",
    "CondensedMemoryRetriever",
    "build_condensed_memory",
]
