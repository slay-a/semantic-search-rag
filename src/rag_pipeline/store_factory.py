"""Pick the vector-store backend from config (``RAG_STORE_BACKEND``).

Keeps the pgvector import lazy so the numpy/offline path never needs psycopg.
"""

from __future__ import annotations

from .config import Settings, settings
from .vector_store import VectorStore


def open_for_write(
    cfg: Settings = settings, *, append: bool = False, tenant_id: str | None = None
):
    """Return a store ready to receive ``add`` calls during ingestion.

    When ``append`` is False and a ``tenant_id`` is given, only that tenant's
    existing rows are cleared (other tenants are left untouched).
    """
    if cfg.store_backend == "pgvector":
        from .pg_store import PgVectorStore

        store = PgVectorStore(cfg)
        if not append:
            store.reset(tenant_id)
        return store

    # numpy backend: keep other tenants by loading + appending when a tenant is
    # scoped (the file holds all tenants; retrieval filters by tenant_id).
    if append or tenant_id is not None:
        try:
            return VectorStore.load(cfg.store_path)
        except FileNotFoundError:
            pass
    return VectorStore()


def open_for_read(cfg: Settings = settings, path: str | None = None):
    """Return a populated store to query against."""
    if cfg.store_backend == "pgvector":
        from .pg_store import PgVectorStore

        return PgVectorStore(cfg)
    return VectorStore.load(path or cfg.store_path)
