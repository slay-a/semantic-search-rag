"""Central configuration for the RAG pipeline.

All values can be overridden with environment variables (see ``.env.example``).
Import :data:`settings` for a ready-to-use, cached instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not required
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline settings resolved from the environment."""

    # --- Generation (Claude) ---
    generation_model: str = field(
        default_factory=lambda: os.getenv("RAG_GENERATION_MODEL", "claude-opus-4-8")
    )
    # Effort controls thinking depth + token spend on Opus 4.8: low|medium|high|xhigh|max.
    effort: str = field(default_factory=lambda: os.getenv("RAG_EFFORT", "high"))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_MAX_TOKENS", 4096))

    # --- Embeddings (OpenAI / GPT) ---
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_PROVIDER", "openai")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    # OpenAI embeddings are symmetric (input_type is ignored); Voyage uses it for
    # asymmetric query/document embeddings. Batching keeps API round-trips down.
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("RAG_EMBEDDING_BATCH_SIZE", 64)
    )

    # --- Chunking ---
    chunk_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_TOKENS", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 64))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    # Drop retrieved chunks below this cosine similarity before they reach the prompt.
    min_score: float = field(
        default_factory=lambda: float(os.getenv("RAG_MIN_SCORE", "0.0"))
    )

    # --- Persistence ---
    store_path: str = field(
        default_factory=lambda: os.getenv("RAG_STORE_PATH", "store/index.npz")
    )

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def voyage_api_key(self) -> str | None:
        return os.getenv("VOYAGE_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
