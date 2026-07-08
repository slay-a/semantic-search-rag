"""Semantic retrieval: dense + optional sparse (hybrid), then optional rerank.

Pipeline per query:

1. **Dense** — embed the query and pull the top ``candidate_k`` chunks by cosine
   similarity. Optionally blended with a HyDE hypothetical-answer embedding.
2. **Sparse (hybrid)** — if enabled and the store supports it, also pull the top
   ``candidate_k`` chunks by BM25/full-text, and fuse the two rankings with
   Reciprocal Rank Fusion (see :mod:`fusion`).
3. **Rerank** — if a reranker is attached, re-score the fused candidates against
   the query and keep the best ``top_k``; otherwise just take the top ``top_k``.

Casting a wide net (``candidate_k``) and then reranking down to ``top_k`` is what
makes hybrid + rerank outperform plain dense search.
"""

from __future__ import annotations

import numpy as np

from .config import Settings, settings
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from .types import ScoredChunk
from .vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        cfg: Settings = settings,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = cfg
        self._reranker = reranker

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        hyde_text: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k if top_k is not None else self._cfg.top_k
        candidate_k = max(self._cfg.candidate_k, top_k)

        query_vec = self._embedder.embed([question], input_type="query")[0]
        if hyde_text:
            hyde_vec = self._embedder.embed([hyde_text], input_type="query")[0]
            query_vec = query_vec + hyde_vec
            norm = np.linalg.norm(query_vec)
            if norm:
                query_vec = query_vec / norm

        dense = self._store.search(
            query_vec,
            top_k=candidate_k,
            min_score=self._cfg.min_score,
            tenant_id=tenant_id,
        )

        candidates = dense
        if self._cfg.hybrid and hasattr(self._store, "keyword_search"):
            sparse = self._store.keyword_search(
                question, top_k=candidate_k, tenant_id=tenant_id
            )
            if sparse:
                candidates = reciprocal_rank_fusion([dense, sparse])

        if self._reranker is not None:
            return self._reranker.rerank(question, candidates, top_k=top_k)
        return candidates[:top_k]
