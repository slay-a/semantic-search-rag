"""Answer generation with Claude (Anthropic Messages API).

Design notes:
- **Model:** defaults to ``claude-opus-4-8`` with **adaptive thinking** and a
  configurable **effort** level — the recommended surface for Opus 4.8.
- **Streaming:** responses stream so we never trip the SDK's long-request
  timeout and can surface tokens live in the CLI. ``get_final_message()`` still
  gives us the complete message + usage at the end.
- **Prompt caching:** the (byte-stable) system prompt carries a ``cache_control``
  breakpoint, so repeated queries in a session pay ~0.1x for the system prefix.
- **Citations:** after generation we parse the ``[n]`` markers out of the answer
  and resolve them back to the retrieved chunks.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .config import Settings, settings
from .prompts import (
    HYDE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_hyde_prompt,
    build_user_prompt,
)
from .types import Citation, ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _client(cfg: Settings):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "anthropic is required for generation. Install with `pip install anthropic`."
        ) from exc
    if not cfg.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your environment/.env.")
    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def extract_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Resolve inline ``[n]`` markers back to the sources they point at."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n in seen or not (1 <= n <= len(chunks)):
            continue
        seen.add(n)
        sc = chunks[n - 1]
        snippet = " ".join(sc.chunk.text.split())[:200]
        citations.append(
            Citation(marker=n, source=sc.chunk.source, score=sc.score, snippet=snippet)
        )
    return sorted(citations, key=lambda c: c.marker)


class Generator:
    """Wraps the Messages API for grounded answering and HyDE rewriting."""

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._anthropic = _client(cfg)

    def _system_blocks(self) -> list[dict]:
        # Single cache breakpoint on the stable system prompt.
        return [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def stream_answer(
        self, question: str, chunks: list[ScoredChunk]
    ) -> Iterator[str]:
        """Yield answer text deltas as they arrive from the model."""
        user_prompt = build_user_prompt(question, chunks)
        with self._anthropic.messages.stream(
            model=self._cfg.generation_model,
            max_tokens=self._cfg.max_tokens,
            system=self._system_blocks(),
            thinking={"type": "adaptive"},
            output_config={"effort": self._cfg.effort},
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            yield from stream.text_stream
            self._last_message = stream.get_final_message()

    def answer(self, question: str, chunks: list[ScoredChunk]) -> tuple[str, dict]:
        """Non-streaming convenience wrapper — returns (answer_text, usage)."""
        parts = list(self.stream_answer(question, chunks))
        msg = getattr(self, "_last_message", None)
        usage: dict = {}
        if msg is not None:
            usage = {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cache_read_input_tokens": getattr(
                    msg.usage, "cache_read_input_tokens", 0
                ),
            }
        return "".join(parts), usage

    def rewrite_query_hyde(self, question: str) -> str:
        """Generate a hypothetical answer paragraph to embed instead of the raw question.

        HyDE (Hypothetical Document Embeddings) narrows the gap between a short
        question and the longer passages that answer it. Uses a small effort
        budget since this is a cheap rewrite, not the final answer.
        """
        response = self._anthropic.messages.create(
            model=self._cfg.generation_model,
            max_tokens=512,
            system=HYDE_SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": build_hyde_prompt(question)}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text.strip() or question
