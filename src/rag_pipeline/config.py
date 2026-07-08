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

    # --- Generation ---
    generation_provider: str = field(
        default_factory=lambda: os.getenv("RAG_GENERATION_PROVIDER", "openai")
    )
    generation_model: str = field(
        default_factory=lambda: os.getenv("RAG_GENERATION_MODEL", "gpt-4o-mini")
    )
    # Anthropic-only: effort controls thinking depth + token spend on Claude
    # (low|medium|high|xhigh|max). Ignored by the OpenAI backend.
    effort: str = field(default_factory=lambda: os.getenv("RAG_EFFORT", "high"))

    # --- Bot personality (per-deployment; the app can override per session) ---
    bot_name: str = field(default_factory=lambda: os.getenv("RAG_BOT_NAME", "Assistant"))
    # Extra persona/voice instructions prepended to the grounding system prompt.
    bot_persona: str = field(default_factory=lambda: os.getenv("RAG_BOT_PERSONA", ""))
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
    # Hybrid search: fuse dense (vector) + sparse (BM25/full-text) rankings.
    hybrid: bool = field(
        default_factory=lambda: os.getenv("RAG_HYBRID", "true").lower() == "true"
    )
    # Candidate pool pulled from each retriever before fusion / reranking.
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 30))

    # --- Reranking ---
    rerank: bool = field(
        default_factory=lambda: os.getenv("RAG_RERANK", "true").lower() == "true"
    )
    rerank_provider: str = field(
        default_factory=lambda: os.getenv("RAG_RERANK_PROVIDER", "llm")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RAG_RERANK_MODEL", "gpt-4o-mini")
    )

    # --- Persistence ---
    # Backend for the vector store: "numpy" (local .npz file) or "pgvector" (Postgres).
    store_backend: str = field(
        default_factory=lambda: os.getenv("RAG_STORE_BACKEND", "numpy")
    )
    store_path: str = field(
        default_factory=lambda: os.getenv("RAG_STORE_PATH", "store/index.npz")
    )
    # Postgres connection string for the pgvector backend, e.g.
    # postgresql://user:pass@host:5432/dbname
    database_url: str | None = field(
        default_factory=lambda: os.getenv("DATABASE_URL")
    )
    # Table name used by the pgvector backend.
    pg_table: str = field(default_factory=lambda: os.getenv("RAG_PG_TABLE", "rag_chunks"))
    # Log each query (question + whether it was answered) to a table for analytics.
    enable_analytics: bool = field(
        default_factory=lambda: os.getenv("RAG_ANALYTICS", "true").lower() == "true"
    )

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    # --- Supabase Auth (optional login). When both are set, the app requires
    # sign-in and scopes each user's data to their account. ---
    @property
    def supabase_url(self) -> str | None:
        return os.getenv("SUPABASE_URL")

    @property
    def supabase_anon_key(self) -> str | None:
        return os.getenv("SUPABASE_ANON_KEY")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def voyage_api_key(self) -> str | None:
        return os.getenv("VOYAGE_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
