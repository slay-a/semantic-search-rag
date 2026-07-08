"""Semantic retrieval: embed the query, search the store, return scored chunks.

Optionally applies HyDE query transformation — embedding a hypothetical answer
paragraph instead of (or alongside) the bare question. When enabled, the query
vector is the mean of the question embedding and the hypothetical-document
embedding, which keeps retrieval anchored to the literal question while gaining
HyDE's recall benefit.
"""

from __future__ import annotations

import numpy as np

from .config import Settings, settings
from .embeddings import Embedder
from .types import ScoredChunk
from .vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        cfg: Settings = settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = cfg

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        hyde_text: str | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k if top_k is not None else self._cfg.top_k
        query_vec = self._embedder.embed([question], input_type="query")[0]

        if hyde_text:
            hyde_vec = self._embedder.embed([hyde_text], input_type="query")[0]
            query_vec = query_vec + hyde_vec
            norm = np.linalg.norm(query_vec)
            if norm:
                query_vec = query_vec / norm

        return self._store.search(query_vec, top_k=top_k, min_score=self._cfg.min_score)
