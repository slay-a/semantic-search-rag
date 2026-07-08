"""Embedding backends.

The pipeline embeds two kinds of text with the *same* model but *different*
``input_type`` hints:

* ``document`` — corpus chunks, embedded once at ingest time.
* ``query`` — the user's question, embedded per request.

Voyage's asymmetric embeddings use that hint to place a question near the
passages that answer it (rather than near other questions), which measurably
improves retrieval. OpenAI's ``text-embedding-3`` models are symmetric and
ignore the hint — the same vector space is used for both. The
:class:`HashEmbedder` is a dependency-free, offline
fallback used by the test suite and for quick demos without an API key — it is
deterministic but NOT semantically meaningful, so never use it in production.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

import numpy as np

from .config import Settings, settings

InputType = Literal["query", "document"]


class Embedder(Protocol):
    """Anything that turns a list of strings into unit-normalized vectors."""

    dimension: int

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VoyageEmbedder:
    """Production embedder backed by Voyage AI (Anthropic's recommended provider)."""

    def __init__(self, cfg: Settings = settings) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "voyageai is required for the Voyage embedder. "
                "Install it with `pip install voyageai`, or set "
                "RAG_EMBEDDING_PROVIDER=hash for an offline demo."
            ) from exc

        if not cfg.voyage_api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Add it to your environment/.env, "
                "or set RAG_EMBEDDING_PROVIDER=hash to run offline."
            )

        self._client = voyageai.Client(api_key=cfg.voyage_api_key)
        self._model = cfg.embedding_model
        self._batch_size = cfg.embedding_batch_size
        # Resolved lazily from the first API response (varies by model).
        self.dimension = 0

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = self._client.embed(batch, model=self._model, input_type=input_type)
            vectors.extend(result.embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1]
        return _normalize(matrix)


class OpenAIEmbedder:
    """Production embedder backed by OpenAI's ``text-embedding-3`` models (GPT family).

    Unlike Voyage, OpenAI embeddings are *symmetric* — there is no ``input_type``
    hint, so the same model embeds both documents and queries. We accept the
    ``input_type`` argument for interface compatibility but ignore it. OpenAI
    returns unit-normalized vectors already; we re-normalize defensively.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "openai is required for the OpenAI embedder. "
                "Install it with `pip install openai`, or set "
                "RAG_EMBEDDING_PROVIDER=hash for an offline demo."
            ) from exc

        if not cfg.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment/.env, "
                "or set RAG_EMBEDDING_PROVIDER=hash to run offline."
            )

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.embedding_model
        self._batch_size = cfg.embedding_batch_size
        # Resolved lazily from the first API response (varies by model).
        self.dimension = 0

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in result.data)

        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1]
        return _normalize(matrix)


class HashEmbedder:
    """Deterministic offline embedder — a hashed bag-of-words projection.

    Good enough to exercise the plumbing and write reproducible tests; it has no
    real semantic understanding. Selected via RAG_EMBEDDING_PROVIDER=hash.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dimension] += 1.0
        return vec

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts])
        return _normalize(matrix)


def build_embedder(cfg: Settings = settings) -> Embedder:
    """Factory that honours ``RAG_EMBEDDING_PROVIDER``."""
    provider = cfg.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(cfg)
    if provider == "voyage":
        return VoyageEmbedder(cfg)
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding provider: {cfg.embedding_provider!r}")
