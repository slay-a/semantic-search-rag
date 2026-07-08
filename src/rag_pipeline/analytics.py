"""Best-effort query analytics.

Logs each question and whether it was answered (vs. the bot abstaining), keyed by
tenant. Abstentions are the valuable signal: they show which questions the
current documents *can't* answer — i.e. what content a customer should add next.

Storage follows the vector-store backend:

* **pgvector** → a ``rag_query_log`` table in the same Postgres/Supabase DB.
* **numpy** → a local ``store/query_log.jsonl`` file.

Logging never raises: analytics must not break a user's query.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings, settings


def _log_pg(cfg: Settings, tenant_id: str | None, question: str, answered: bool) -> None:
    from .pg_store import PgVectorStore  # lazy: only when pg backend is active

    store = PgVectorStore(cfg)
    with store._conn.cursor() as cur:  # noqa: SLF001 - reuse the open connection
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_query_log (
                id        bigserial PRIMARY KEY,
                tenant_id text,
                question  text NOT NULL,
                answered  boolean NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            "INSERT INTO rag_query_log (tenant_id, question, answered) VALUES (%s, %s, %s)",
            (tenant_id, question, answered),
        )
    store.close()


def _log_file(cfg: Settings, tenant_id: str | None, question: str, answered: bool) -> None:
    path = Path(cfg.store_path).parent / "query_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"tenant_id": tenant_id, "question": question, "answered": answered}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def log_query(
    question: str,
    answered: bool,
    *,
    tenant_id: str | None = None,
    cfg: Settings = settings,
) -> None:
    """Record one query. Silently does nothing if analytics is off or errors."""
    if not cfg.enable_analytics:
        return
    try:
        if cfg.store_backend == "pgvector":
            _log_pg(cfg, tenant_id, question, answered)
        else:
            _log_file(cfg, tenant_id, question, answered)
    except Exception:  # analytics is best-effort; never break the query path
        pass
