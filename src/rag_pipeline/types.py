"""Shared data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance metadata."""

    id: str
    text: str
    source: str
    # Free-form metadata (page number, title, section, ingest timestamp, ...).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk returned by retrieval, with its similarity to the query."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A source the model cited, resolved back to the retrieved chunk."""

    marker: int  # the [n] index shown to the model
    source: str
    score: float
    snippet: str


@dataclass
class RAGAnswer:
    """The full result of a query: answer text, citations, and retrieval trace."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
