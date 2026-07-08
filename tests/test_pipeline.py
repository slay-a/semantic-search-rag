"""Tests that run fully offline — no API keys required.

They exercise chunking, the hashed offline embedder, the vector store
(including save/load round-trips), retrieval, and citation extraction. Claude
generation is covered with a fake generator so we can assert the orchestration
without a network call.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_pipeline.chunking import chunk_document, estimate_tokens
from rag_pipeline.embeddings import HashEmbedder
from rag_pipeline.generator import extract_citations
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.types import Chunk, ScoredChunk
from rag_pipeline.vector_store import VectorStore


def test_chunking_overlap_and_coverage():
    text = "\n\n".join(f"Paragraph {i} with some content about topic {i}." for i in range(50))
    chunks = chunk_document(text, source="doc.md", chunk_tokens=32, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(c.source == "doc.md" for c in chunks)
    # Every chunk should be non-empty and carry an id.
    assert all(c.text.strip() and c.id for c in chunks)


def test_estimate_tokens_monotonic():
    assert estimate_tokens("a b c") < estimate_tokens("a b c d e f g h i j k")


def test_hash_embedder_is_normalized_and_deterministic():
    emb = HashEmbedder(dimension=64)
    a = emb.embed(["retry on 429"], input_type="query")
    b = emb.embed(["retry on 429"], input_type="query")
    assert a.shape == (1, 64)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)


def _toy_store(embedder: HashEmbedder) -> VectorStore:
    chunks = [
        Chunk(id="1", text="The API returns 429 and a Retry-After header.", source="a"),
        Chunk(id="2", text="Payouts are settled every business day.", source="b"),
        Chunk(id="3", text="Webhooks are signed with HMAC-SHA256.", source="c"),
    ]
    vectors = embedder.embed([c.text for c in chunks], input_type="document")
    store = VectorStore()
    store.add(chunks, vectors)
    return store


def test_search_ranks_relevant_chunk_first():
    embedder = HashEmbedder(dimension=128)
    store = _toy_store(embedder)
    qv = embedder.embed(["what happens on 429 Retry-After"], input_type="query")[0]
    results = store.search(qv, top_k=3)
    assert results[0].chunk.id == "1"
    assert results[0].score >= results[-1].score


def test_store_save_load_roundtrip(tmp_path):
    embedder = HashEmbedder(dimension=64)
    store = _toy_store(embedder)
    path = tmp_path / "index.npz"
    store.save(path)

    loaded = VectorStore.load(path)
    assert len(loaded) == len(store)
    assert loaded.dimension == store.dimension
    qv = embedder.embed(["signed webhook"], input_type="query")[0]
    assert loaded.search(qv, top_k=1)[0].chunk.id == "3"


def test_extract_citations_resolves_markers():
    chunks = [
        ScoredChunk(Chunk(id="1", text="Alpha content", source="a"), 0.9),
        ScoredChunk(Chunk(id="2", text="Beta content", source="b"), 0.8),
    ]
    citations = extract_citations("Claim one [1]. Claim two [2][1]. Bogus [9].", chunks)
    markers = [c.marker for c in citations]
    assert markers == [1, 2]  # deduped, ordered, out-of-range dropped
    assert citations[0].source == "a"


class _FakeGenerator:
    """Stand-in for the Claude-backed Generator (no network)."""

    def answer(self, question, chunks):
        cite = "".join(f"[{i}]" for i in range(1, len(chunks) + 1))
        return f"Grounded answer {cite}", {"input_tokens": 10, "output_tokens": 5}

    def stream_answer(self, question, chunks):
        yield from ["Grounded ", "answer ", "[1]"]

    def rewrite_query_hyde(self, question):
        return question


def test_pipeline_query_end_to_end_offline():
    embedder = HashEmbedder(dimension=128)
    store = _toy_store(embedder)
    rag = RAGPipeline(store, embedder=embedder, generator=_FakeGenerator())
    answer = rag.query("what happens on 429", top_k=2)
    assert answer.answer.startswith("Grounded answer")
    assert answer.citations  # markers resolved back to sources
    assert len(answer.retrieved) == 2


def test_add_dimension_mismatch_raises():
    store = VectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add([Chunk(id="x", text="t", source="s")], np.zeros((1, 8), dtype=np.float32))
