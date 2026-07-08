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
from pathlib import Path

import numpy as np

from .types import Chunk, ScoredChunk


class VectorStore:
    def __init__(self, dimension: int | None = None) -> None:
        self._dim = dimension
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        return self._dim

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
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

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, min_score: float = 0.0
    ) -> list[ScoredChunk]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        query_vector = query_vector.reshape(-1).astype(np.float32)
        # Vectors are already normalized, so the dot product is cosine similarity.
        scores = self._vectors @ query_vector
        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those k by score.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
        ]
        return [r for r in results if r.score >= min_score]

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
