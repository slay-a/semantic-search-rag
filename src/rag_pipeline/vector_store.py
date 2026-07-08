"""A small, dependency-light vector store.

Vectors are unit-normalized on the way in, so cosine similarity is a single
matrix-vector product. This is exact (not approximate) nearest-neighbour search:
plenty fast for up to ~10^5 chunks on a laptop, and it keeps the repo runnable
with zero external services. The interface (``add`` / ``search`` / ``save`` /
``load``) is deliberately narrow so it can be swapped for a managed vector DB
(pgvector, Pinecone, Chroma, ...) without touching the rest of the pipeline.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .types import Chunk, ScoredChunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class VectorStore:
    def __init__(self, dimension: int | None = None) -> None:
        self._dim = dimension
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        # Lazily-built BM25 index for keyword_search (invalidated on add).
        self._bm25: _BM25Index | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        return self._dim

    def add(
        self, chunks: list[Chunk], vectors: np.ndarray, *, tenant_id: str | None = None
    ) -> None:
        if tenant_id is not None:
            # Stamp tenant on each chunk so search can filter by it (and so it
            # survives save/load in chunk metadata).
            for c in chunks:
                c.metadata.setdefault("tenant_id", tenant_id)
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[0] == 0:
            return
        if self._dim is None:
            self._dim = vectors.shape[1]
        elif vectors.shape[1] != self._dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != store dim {self._dim}"
            )

        vectors = vectors.astype(np.float32, copy=False)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )
        self._chunks.extend(chunks)
        self._bm25 = None  # corpus changed; rebuild lazily on next keyword search

    def _tenant_mask(self, tenant_id: str | None) -> np.ndarray | None:
        """Boolean mask of chunks visible to ``tenant_id`` (None = all)."""
        if tenant_id is None:
            return None
        return np.array(
            [c.metadata.get("tenant_id") == tenant_id for c in self._chunks]
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        *,
        tenant_id: str | None = None,
    ) -> list[ScoredChunk]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        query_vector = query_vector.reshape(-1).astype(np.float32)
        # Vectors are already normalized, so the dot product is cosine similarity.
        scores = self._vectors @ query_vector
        mask = self._tenant_mask(tenant_id)
        if mask is not None:
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)  # hide other tenants' chunks
        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those k by score.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
            if np.isfinite(scores[i])
        ]
        return [r for r in results if r.score >= min_score]

    def keyword_search(
        self, query: str, top_k: int = 5, *, tenant_id: str | None = None
    ) -> list[ScoredChunk]:
        """BM25 lexical search over the chunk texts (the sparse half of hybrid)."""
        if not self._chunks:
            return []
        if self._bm25 is None:
            self._bm25 = _BM25Index([c.text for c in self._chunks])
        # Over-fetch when filtering by tenant so the tenant still gets top_k hits.
        fetch = top_k if tenant_id is None else min(len(self._chunks), top_k * 5)
        ranked = self._bm25.search(query, fetch)
        out: list[ScoredChunk] = []
        for i, s in ranked:
            if tenant_id is not None and self._chunks[i].metadata.get("tenant_id") != tenant_id:
                continue
            out.append(ScoredChunk(chunk=self._chunks[i], score=s))
            if len(out) >= top_k:
                break
        return out

    # --- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        np.savez_compressed(
            path,
            vectors=self._vectors if self._vectors is not None else np.zeros((0, 0)),
            meta=np.array(json.dumps(meta)),
            dimension=np.array(self._dim if self._dim is not None else 0),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No vector store at {path}. Run ingestion first (rag-ingest)."
            )
        with np.load(path, allow_pickle=False) as data:
            dim = int(data["dimension"])
            store = cls(dimension=dim or None)
            meta = json.loads(str(data["meta"]))
            vectors = data["vectors"]
            store._chunks = [
                Chunk(
                    id=m["id"],
                    text=m["text"],
                    source=m["source"],
                    metadata=m.get("metadata", {}),
                )
                for m in meta
            ]
            store._vectors = vectors.astype(np.float32) if vectors.size else None
        return store


class _BM25Index:
    """Minimal Okapi BM25 over a fixed corpus of documents.

    Dependency-free (pure Python) so hybrid retrieval works offline and in tests.
    Built once per corpus; the parent store discards it when documents change.
    """

    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_tokenize(d) for d in docs]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._tf: list[Counter] = [Counter(t) for t in self._doc_tokens]

        df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        n = len(docs)
        # BM25 idf with the +0.5 smoothing from the standard formulation.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = _tokenize(query)
        scores = np.zeros(len(self._tf), dtype=np.float32)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self._k1 * (1 - self._b + self._b * self._doc_len[i] / self._avgdl)
                scores[i] += idf * (f * (self._k1 + 1)) / denom

        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        # Drop zero-score docs (query terms absent) — they aren't real matches.
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]
