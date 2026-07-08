"""Postgres + pgvector backend — the deployable vector store.

Implements the same ``add`` / ``search`` / ``keyword_search`` / ``save`` surface
as the in-memory :class:`~rag_pipeline.vector_store.VectorStore`, so the pipeline
is backend-agnostic. Selected with ``RAG_STORE_BACKEND=pgvector`` +
``DATABASE_URL``.

One table holds everything hybrid search needs:

* ``embedding vector(d)`` with an HNSW cosine index → dense search.
* ``tsv tsvector`` (generated from ``text``) with a GIN index → BM25-style
  full-text search via ``ts_rank`` / ``websearch_to_tsquery``.

Data is committed on ``add`` (Postgres *is* the persistence), so ``save`` is a
no-op kept only for interface parity with the file-based store.
"""

from __future__ import annotations

import json

from .config import Settings, settings
from .types import Chunk, ScoredChunk


def _vec_literal(vector) -> str:
    """Format a vector as a pgvector text literal, e.g. ``[0.1,0.2,0.3]``."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, cfg: Settings = settings) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "psycopg is required for the pgvector backend. "
                "Install it with `pip install 'psycopg[binary]'`."
            ) from exc
        if not cfg.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Point it at your Postgres instance, "
                "e.g. postgresql://user:pass@host:5432/dbname"
            )
        self._cfg = cfg
        self._table = cfg.pg_table
        self._conn = psycopg.connect(cfg.database_url, autocommit=True)
        self._dim: int | None = None
        self._ensure_extension()

    # --- schema ------------------------------------------------------------
    def _ensure_extension(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _ensure_table(self, dim: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id        text PRIMARY KEY,
                    tenant_id text,
                    text      text NOT NULL,
                    source    text NOT NULL,
                    metadata  jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({dim}),
                    tsv       tsvector GENERATED ALWAYS AS
                              (to_tsvector('english', text)) STORED
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_tsv_idx "
                f"ON {self._table} USING gin (tsv)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_tenant_idx "
                f"ON {self._table} (tenant_id)"
            )
        self._dim = dim

    def reset(self, tenant_id: str | None = None) -> None:
        """Drop rows before a non-append re-ingest (scoped to a tenant if given)."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{self._table}",))
            if cur.fetchone()[0] is None:
                return
            if tenant_id is None:
                cur.execute(f"TRUNCATE TABLE {self._table}")
            else:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE tenant_id = %s", (tenant_id,)
                )

    # --- writes ------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors, *, tenant_id: str | None = None) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[0] == 0:
            return
        self._ensure_table(vectors.shape[1])
        rows = [
            (
                c.id,
                tenant_id or c.metadata.get("tenant_id"),
                c.text,
                c.source,
                json.dumps(c.metadata),
                _vec_literal(v),
            )
            for c, v in zip(chunks, vectors)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._table} (id, tenant_id, text, source, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    text = EXCLUDED.text,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

    def save(self, path=None) -> None:  # noqa: ARG002 - parity with VectorStore
        """No-op: rows are committed on ``add``. Kept for interface parity."""

    # --- reads -------------------------------------------------------------
    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{self._table}",))
            if cur.fetchone()[0] is None:  # table not created yet
                return 0
            cur.execute(f"SELECT count(*) FROM {self._table}")
            return int(cur.fetchone()[0])

    def search(
        self,
        query_vector,
        top_k: int = 5,
        min_score: float = 0.0,
        *,
        tenant_id: str | None = None,
    ) -> list[ScoredChunk]:
        lit = _vec_literal(query_vector)
        where = "WHERE tenant_id = %s" if tenant_id is not None else ""
        params = ([tenant_id] if tenant_id is not None else []) + [lit, top_k]
        with self._conn.cursor() as cur:
            # 1 - cosine_distance = cosine similarity, to match the NumPy store.
            cur.execute(
                f"""
                SELECT id, text, source, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self._table}
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [lit, *params],
            )
            rows = cur.fetchall()
        results = [self._row_to_scored(r) for r in rows]
        return [r for r in results if r.score >= min_score]

    def keyword_search(
        self, query: str, top_k: int = 5, *, tenant_id: str | None = None
    ) -> list[ScoredChunk]:
        tenant_filter = "AND tenant_id = %s" if tenant_id is not None else ""
        params = [query, query]
        if tenant_id is not None:
            params.append(tenant_id)
        params.append(top_k)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, text, source, metadata,
                       ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
                FROM {self._table}
                WHERE tsv @@ websearch_to_tsquery('english', %s)
                {tenant_filter}
                ORDER BY score DESC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()
        return [self._row_to_scored(r) for r in rows]

    @staticmethod
    def _row_to_scored(row) -> ScoredChunk:
        cid, text, source, metadata, score = row
        meta = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
        return ScoredChunk(
            chunk=Chunk(id=cid, text=text, source=source, metadata=meta),
            score=float(score),
        )

    def close(self) -> None:
        self._conn.close()
