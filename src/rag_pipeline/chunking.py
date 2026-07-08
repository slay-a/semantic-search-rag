"""Turn raw documents into overlapping, retrieval-sized chunks.

Chunking strategy matters as much as the embedding model. We split on paragraph
boundaries first (to keep semantically coherent units together), then pack
paragraphs into windows of roughly ``chunk_tokens`` with ``chunk_overlap`` tokens
of carry-over so a fact that straddles a boundary is still fully present in at
least one chunk.

Token counts here are approximated as ``len(text) / 4`` (a solid rule of thumb
for English prose) to avoid a hard tokenizer dependency at ingest time. The
retrieval quality is insensitive to small errors in this estimate.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .types import Chunk

_CHARS_PER_TOKEN = 4
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARAGRAPH_RE.split(text)]
    return [p for p in parts if p]


def _hard_split(paragraph: str, max_chars: int) -> Iterable[str]:
    """Split an over-long paragraph on sentence boundaries, then hard-wrap."""
    if len(paragraph) <= max_chars:
        yield paragraph
        return
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    buf = ""
    for sentence in sentences:
        if len(sentence) > max_chars:  # a single monster sentence -> hard wrap
            if buf:
                yield buf
                buf = ""
            for i in range(0, len(sentence), max_chars):
                yield sentence[i : i + max_chars]
            continue
        if len(buf) + len(sentence) + 1 > max_chars:
            yield buf
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        yield buf


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:12]
    return f"{index:04d}-{digest}"


def chunk_document(
    text: str,
    source: str,
    *,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    """Split one document's text into overlapping :class:`Chunk` objects."""
    max_chars = chunk_tokens * _CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * _CHARS_PER_TOKEN
    base_metadata = base_metadata or {}

    # Break paragraphs down so none individually exceeds the window.
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        units.extend(_hard_split(paragraph, max_chars))

    chunks: list[Chunk] = []
    buf = ""
    index = 0
    for unit in units:
        candidate = f"{buf}\n\n{unit}".strip() if buf else unit
        if buf and len(candidate) > max_chars:
            chunks.append(
                Chunk(
                    id=_chunk_id(source, index, buf),
                    text=buf,
                    source=source,
                    metadata=dict(base_metadata),
                )
            )
            index += 1
            # Start the next window with a tail of the previous one (overlap).
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            buf = candidate

    if buf.strip():
        chunks.append(
            Chunk(
                id=_chunk_id(source, index, buf),
                text=buf,
                source=source,
                metadata=dict(base_metadata),
            )
        )
    return chunks
