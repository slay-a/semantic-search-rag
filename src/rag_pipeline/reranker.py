"""Reranking: re-score retrieval candidates by reading query + passage together.

Hybrid retrieval casts a wide, cheap net (``candidate_k`` hits). A reranker then
does the expensive, accurate part: for each candidate it judges *this passage
against this exact query* and reorders, so the handful of chunks that actually
reach the prompt are the most relevant ones.

Backends (``RAG_RERANK_PROVIDER``):

* ``llm`` (default) — asks an OpenAI chat model to score each candidate 0-10.
  No heavy ML dependencies, so it deploys anywhere (unlike a PyTorch
  cross-encoder, which won't fit typical free-tier memory limits). Costs a small
  number of tokens per query.
* ``none`` — identity passthrough (keeps the fused order). Used offline/in tests.

The interface is a single :meth:`rerank` call, so a local cross-encoder backend
can be dropped in later without touching the pipeline.
"""

from __future__ import annotations

import json
from typing import Protocol

from .config import Settings, settings
from .types import ScoredChunk

# Keep the reranker prompt cheap: cap how many candidates we send and how much
# of each passage the model reads.
_MAX_CANDIDATES = 30
_SNIPPET_CHARS = 600


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]: ...


class IdentityReranker:
    """No-op reranker — returns the first ``top_k`` candidates unchanged."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        return candidates[:top_k]


class LLMReranker:
    """Rerank candidates by asking an OpenAI chat model to score each 0-10."""

    def __init__(self, cfg: Settings = settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "openai is required for the LLM reranker. "
                "Install it with `pip install openai`, or set RAG_RERANK=false."
            ) from exc
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment/.env.")
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.rerank_model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pool = candidates[:_MAX_CANDIDATES]

        listing = "\n".join(
            f"[{i}] {' '.join(sc.chunk.text.split())[:_SNIPPET_CHARS]}"
            for i, sc in enumerate(pool)
        )
        system = (
            "You are a search reranker. Score how well each passage answers the "
            "user's query on a scale from 0 (irrelevant) to 10 (directly answers "
            "it). Respond ONLY with a JSON object mapping each passage index (as a "
            'string) to its integer score, e.g. {"0": 8, "1": 2}.'
        )
        user = f"Query: {query}\n\nPassages:\n{listing}"

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            scores = json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            # Any failure (network, parse) falls back to the fused order.
            return pool[:top_k]

        def score_for(idx: int) -> float:
            raw = scores.get(str(idx))
            try:
                return float(raw)
            except (TypeError, ValueError):
                return -1.0

        reranked = sorted(
            range(len(pool)), key=lambda i: score_for(i), reverse=True
        )
        return [
            ScoredChunk(chunk=pool[i].chunk, score=score_for(i))
            for i in reranked[:top_k]
        ]


def build_reranker(cfg: Settings = settings) -> Reranker:
    """Factory honouring ``RAG_RERANK`` / ``RAG_RERANK_PROVIDER``."""
    if not cfg.rerank:
        return IdentityReranker()
    provider = cfg.rerank_provider.lower()
    if provider in ("none", "identity"):
        return IdentityReranker()
    if provider == "llm":
        return LLMReranker(cfg)
    raise ValueError(f"Unknown rerank provider: {cfg.rerank_provider!r}")
