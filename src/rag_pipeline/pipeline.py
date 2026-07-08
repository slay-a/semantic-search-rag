"""Top-level orchestration tying retrieval and generation together.

    RAGPipeline.from_store("store/index.npz").query("How does auth work?")

The pipeline is the single object an application interacts with. It owns the
embedder, retriever, generator, and reranker, and exposes both a one-shot
:meth:`query` and a token-streaming :meth:`stream_query`. Both accept a
``tenant_id`` (multi-tenant isolation) and ``history`` (multi-turn memory).
"""

from __future__ import annotations

from collections.abc import Iterator

from . import analytics
from .config import Settings, settings
from .embeddings import Embedder, build_embedder
from .generator import Generator, build_generator, extract_citations
from .prompts import is_abstention
from .reranker import Reranker, build_reranker
from .retriever import Retriever
from .store_factory import open_for_read
from .types import RAGAnswer


class RAGPipeline:
    def __init__(
        self,
        store,
        *,
        cfg: Settings = settings,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        reranker: Reranker | None = None,
        persona: str | None = None,
        allow_general: bool = False,
        use_hyde: bool = False,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder or build_embedder(cfg)
        self._reranker = reranker or build_reranker(cfg)
        self._retriever = Retriever(store, self._embedder, cfg, reranker=self._reranker)
        self._generator = generator or build_generator(cfg, persona, allow_general)
        self._use_hyde = use_hyde

    @classmethod
    def from_store(
        cls, path: str | None = None, *, cfg: Settings = settings, **kwargs
    ) -> "RAGPipeline":
        store = open_for_read(cfg, path)
        return cls(store, cfg=cfg, **kwargs)

    def _retrieve(self, question: str, top_k: int | None, tenant_id: str | None):
        hyde_text = None
        if self._use_hyde:
            hyde_text = self._generator.rewrite_query_hyde(question)
        return self._retriever.retrieve(
            question, top_k=top_k, hyde_text=hyde_text, tenant_id=tenant_id
        )

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        tenant_id: str | None = None,
        history: list[dict] | None = None,
    ) -> RAGAnswer:
        """Answer a question end to end and return the full trace."""
        retrieved = self._retrieve(question, top_k, tenant_id)
        answer_text, usage = self._generator.answer(question, retrieved, history=history)
        abstained = is_abstention(answer_text)
        analytics.log_query(
            question, answered=not abstained, tenant_id=tenant_id, cfg=self._cfg
        )
        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=extract_citations(answer_text, retrieved),
            retrieved=retrieved,
            usage=usage,
            abstained=abstained,
        )

    def stream_query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        tenant_id: str | None = None,
        history: list[dict] | None = None,
    ) -> tuple[Iterator[str], RAGAnswer]:
        """Stream the answer live.

        Returns ``(token_iterator, answer)``. Drain the iterator first (that is
        what runs generation); the returned :class:`RAGAnswer` is populated with
        the final text, citations, usage, and abstention flag once the iterator
        is exhausted.
        """
        retrieved = self._retrieve(question, top_k, tenant_id)
        answer = RAGAnswer(question=question, answer="", retrieved=retrieved)

        def _run() -> Iterator[str]:
            parts: list[str] = []
            for delta in self._generator.stream_answer(
                question, retrieved, history=history
            ):
                parts.append(delta)
                yield delta
            answer.answer = "".join(parts)
            answer.citations = extract_citations(answer.answer, retrieved)
            answer.usage = dict(self._generator.last_usage)
            answer.abstained = is_abstention(answer.answer)
            analytics.log_query(
                question, answered=not answer.abstained, tenant_id=tenant_id, cfg=self._cfg
            )

        return _run(), answer
