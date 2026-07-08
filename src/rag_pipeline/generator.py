"""Answer generation from retrieved context.

Two interchangeable backends implement the same small interface
(``stream_answer`` / ``answer`` / ``rewrite_query_hyde``); pick one with
``RAG_GENERATION_PROVIDER``:

* **OpenAI** (default) — Chat Completions API, streamed. GPT models don't expose
  Claude's thinking/effort knobs, so those settings are simply ignored here.
* **Anthropic (Claude)** — Messages API with adaptive thinking, a configurable
  effort level, and a prompt-cache breakpoint on the stable system prompt.

Both expose ``last_usage`` (a normalized token-usage dict) after a call, and both
emit inline ``[n]`` citation markers that :func:`extract_citations` resolves back
to the retrieved chunks.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol

from .config import Settings, settings
from .prompts import (
    HYDE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_hyde_prompt,
    build_user_prompt,
)
from .types import Citation, ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _system_text(persona: str) -> str:
    """Grounding system prompt, optionally prefixed with a per-bot persona."""
    persona = (persona or "").strip()
    return f"{persona}\n\n{SYSTEM_PROMPT}" if persona else SYSTEM_PROMPT


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


class Generator(Protocol):
    """The narrow surface the pipeline depends on."""

    last_usage: dict

    def stream_answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = ...
    ) -> Iterator[str]: ...
    def answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = ...
    ) -> tuple[str, dict]: ...
    def rewrite_query_hyde(self, question: str) -> str: ...


# --------------------------------------------------------------------------- #
# OpenAI (GPT) backend
# --------------------------------------------------------------------------- #
def _openai_client(cfg: Settings):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "openai is required for generation. Install with `pip install openai`."
        ) from exc
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment/.env.")
    return OpenAI(api_key=cfg.openai_api_key)


def _openai_usage(usage) -> dict:
    if usage is None:
        return {}
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "cache_read_input_tokens": cached,
    }


class OpenAIGenerator:
    """Grounded answering + HyDE rewriting via OpenAI Chat Completions."""

    def __init__(self, cfg: Settings = settings, persona: str = "") -> None:
        self._cfg = cfg
        self._client = _openai_client(cfg)
        self._system = _system_text(persona)
        self.last_usage: dict = {}

    def stream_answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = None
    ) -> Iterator[str]:
        self.last_usage = {}
        messages = [{"role": "system", "content": self._system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": build_user_prompt(question, chunks)})
        stream = self._client.chat.completions.create(
            model=self._cfg.generation_model,
            max_tokens=self._cfg.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            messages=messages,
        )
        for chunk in stream:
            if chunk.usage is not None:
                self.last_usage = _openai_usage(chunk.usage)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = None
    ) -> tuple[str, dict]:
        parts = list(self.stream_answer(question, chunks, history=history))
        return "".join(parts), self.last_usage

    def rewrite_query_hyde(self, question: str) -> str:
        response = self._client.chat.completions.create(
            model=self._cfg.generation_model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": build_hyde_prompt(question)},
            ],
        )
        return (response.choices[0].message.content or "").strip() or question


# --------------------------------------------------------------------------- #
# Anthropic (Claude) backend
# --------------------------------------------------------------------------- #
def _anthropic_client(cfg: Settings):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "anthropic is required for generation. Install with `pip install anthropic`."
        ) from exc
    if not cfg.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your environment/.env.")
    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def _anthropic_usage(msg) -> dict:
    if msg is None:
        return {}
    return {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0),
    }


class AnthropicGenerator:
    """Grounded answering + HyDE rewriting via the Claude Messages API."""

    def __init__(self, cfg: Settings = settings, persona: str = "") -> None:
        self._cfg = cfg
        self._anthropic = _anthropic_client(cfg)
        self._persona = (persona or "").strip()
        self.last_usage: dict = {}

    def _system_blocks(self) -> list[dict]:
        # Cache breakpoint on the stable grounding prompt; persona (which can vary
        # per bot) goes in a separate, uncached block after it.
        blocks = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]
        if self._persona:
            blocks.append({"type": "text", "text": self._persona})
        return blocks

    def stream_answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = None
    ) -> Iterator[str]:
        self.last_usage = {}
        messages = list(history or [])
        messages.append({"role": "user", "content": build_user_prompt(question, chunks)})
        with self._anthropic.messages.stream(
            model=self._cfg.generation_model,
            max_tokens=self._cfg.max_tokens,
            system=self._system_blocks(),
            thinking={"type": "adaptive"},
            output_config={"effort": self._cfg.effort},
            messages=messages,
        ) as stream:
            yield from stream.text_stream
            self.last_usage = _anthropic_usage(stream.get_final_message())

    def answer(
        self, question: str, chunks: list[ScoredChunk], *, history: list[dict] | None = None
    ) -> tuple[str, dict]:
        parts = list(self.stream_answer(question, chunks, history=history))
        return "".join(parts), self.last_usage

    def rewrite_query_hyde(self, question: str) -> str:
        response = self._anthropic.messages.create(
            model=self._cfg.generation_model,
            max_tokens=512,
            system=HYDE_SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": build_hyde_prompt(question)}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text.strip() or question


def build_generator(cfg: Settings = settings, persona: str | None = None) -> Generator:
    """Factory that honours ``RAG_GENERATION_PROVIDER`` (openai | anthropic)."""
    persona = cfg.bot_persona if persona is None else persona
    provider = cfg.generation_provider.lower()
    if provider == "openai":
        return OpenAIGenerator(cfg, persona=persona)
    if provider == "anthropic":
        return AnthropicGenerator(cfg, persona=persona)
    raise ValueError(f"Unknown generation provider: {cfg.generation_provider!r}")
