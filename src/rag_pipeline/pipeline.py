"""Top-level orchestration tying retrieval and generation together.

    RAGPipeline.from_store("store/index.npz").query("How does auth work?")

The pipeline is the single object an application interacts with. It owns the
embedder, retriever, and generator, and exposes both a one-shot :meth:`query`
and a token-streaming :meth:`stream_query`.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import Settings, settings
from .embeddings import Embedder, build_embedder
from .generator import Generator, extract_citations
from .retriever import Retriever
from .types import RAGAnswer
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        *,
        cfg: Settings = settings,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        use_hyde: bool = False,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder or build_embedder(cfg)
        self._retriever = Retriever(store, self._embedder, cfg)
        self._generator = generator or Generator(cfg)
        self._use_hyde = use_hyde

    @classmethod
    def from_store(
        cls, path: str | None = None, *, cfg: Settings = settings, **kwargs
    ) -> "RAGPipeline":
        store = VectorStore.load(path or cfg.store_path)
        return cls(store, cfg=cfg, **kwargs)

    def _retrieve(self, question: str, top_k: int | None):
        hyde_text = None
        if self._use_hyde:
            hyde_text = self._generator.rewrite_query_hyde(question)
        return self._retriever.retrieve(question, top_k=top_k, hyde_text=hyde_text)

    def query(self, question: str, *, top_k: int | None = None) -> RAGAnswer:
        """Answer a question end to end and return the full trace."""
        retrieved = self._retrieve(question, top_k)
        answer_text, usage = self._generator.answer(question, retrieved)
        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=extract_citations(answer_text, retrieved),
            retrieved=retrieved,
            usage=usage,
        )

    def stream_query(
        self, question: str, *, top_k: int | None = None
    ) -> tuple[Iterator[str], RAGAnswer]:
        """Stream the answer live.

        Returns ``(token_iterator, answer)``. Drain the iterator first (that is
        what runs generation); the returned :class:`RAGAnswer` is populated with
        the final text, citations, and usage once the iterator is exhausted.
        """
        retrieved = self._retrieve(question, top_k)
        answer = RAGAnswer(question=question, answer="", retrieved=retrieved)

        def _run() -> Iterator[str]:
            parts: list[str] = []
            for delta in self._generator.stream_answer(question, retrieved):
                parts.append(delta)
                yield delta
            answer.answer = "".join(parts)
            answer.citations = extract_citations(answer.answer, retrieved)
            msg = getattr(self._generator, "_last_message", None)
            if msg is not None:
                answer.usage = {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        msg.usage, "cache_read_input_tokens", 0
                    ),
                }

        return _run(), answer
