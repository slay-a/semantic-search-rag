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
from rag_pipeline.fusion import reciprocal_rank_fusion
from rag_pipeline.generator import extract_citations
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.reranker import IdentityReranker
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
    """Stand-in for a real Generator (no network)."""

    last_usage: dict = {"input_tokens": 10, "output_tokens": 5}

    def answer(self, question, chunks, *, history=None):
        cite = "".join(f"[{i}]" for i in range(1, len(chunks) + 1))
        return f"Grounded answer {cite}", self.last_usage

    def stream_answer(self, question, chunks, *, history=None):
        yield from ["Grounded ", "answer ", "[1]"]

    def rewrite_query_hyde(self, question):
        return question


def test_pipeline_query_end_to_end_offline():
    embedder = HashEmbedder(dimension=128)
    store = _toy_store(embedder)
    from dataclasses import replace

    from rag_pipeline.config import settings as _settings

    rag = RAGPipeline(
        store,
        cfg=replace(_settings, enable_analytics=False),  # no analytics file writes in tests
        embedder=embedder,
        generator=_FakeGenerator(),
        reranker=IdentityReranker(),  # avoid the network-backed default reranker
    )
    answer = rag.query("what happens on 429", top_k=2)
    assert answer.answer.startswith("Grounded answer")
    assert answer.citations  # markers resolved back to sources
    assert len(answer.retrieved) == 2


def test_add_dimension_mismatch_raises():
    store = VectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add([Chunk(id="x", text="t", source="s")], np.zeros((1, 8), dtype=np.float32))


def test_bm25_keyword_search_finds_exact_term():
    embedder = HashEmbedder(dimension=64)
    store = _toy_store(embedder)
    # "HMAC" is a rare exact term dense embeddings often miss but BM25 nails.
    results = store.keyword_search("HMAC signed webhooks", top_k=3)
    assert results
    assert results[0].chunk.id == "3"


def test_rrf_fusion_rewards_agreement():
    a = Chunk(id="a", text="", source="")
    b = Chunk(id="b", text="", source="")
    c = Chunk(id="c", text="", source="")
    dense = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8)]
    sparse = [ScoredChunk(b, 5.0), ScoredChunk(c, 4.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    # b appears in both lists, so it should fuse to the top.
    assert fused[0].chunk.id == "b"
    assert {sc.chunk.id for sc in fused} == {"a", "b", "c"}


def test_bm25_index_survives_save_load(tmp_path):
    embedder = HashEmbedder(dimension=64)
    store = _toy_store(embedder)
    path = tmp_path / "idx.npz"
    store.save(path)
    reloaded = VectorStore.load(path)
    results = reloaded.keyword_search("Retry-After header", top_k=2)
    assert results and results[0].chunk.id == "1"


def test_tenant_isolation_in_search_and_keyword():
    embedder = HashEmbedder(dimension=64)
    store = VectorStore()
    acme = [Chunk(id="a1", text="Acme refunds take 5 business days.", source="acme")]
    globex = [Chunk(id="g1", text="Globex refunds are instant.", source="globex")]
    store.add(acme, embedder.embed([c.text for c in acme], input_type="document"),
              tenant_id="acme")
    store.add(globex, embedder.embed([c.text for c in globex], input_type="document"),
              tenant_id="globex")

    qv = embedder.embed(["refund policy"], input_type="query")[0]
    acme_hits = store.search(qv, top_k=5, tenant_id="acme")
    assert acme_hits and all(h.chunk.source == "acme" for h in acme_hits)

    kw = store.keyword_search("refunds", top_k=5, tenant_id="globex")
    assert kw and all(h.chunk.source == "globex" for h in kw)


def test_crawl_site_stays_on_domain_and_caps_pages():
    from rag_pipeline.ingest import crawl_site

    pages = {
        "https://ex.com/": "<a href='/a'>a</a><a href='https://other.com/z'>z</a> home",
        "https://ex.com/a": "<a href='/b'>b</a><a href='/c'>c</a> alpha",
        "https://ex.com/b": "beta",
        "https://ex.com/c": "gamma",
    }
    out = crawl_site("https://ex.com/", max_pages=3, fetcher=lambda u: pages[u])
    urls = [u for u, _ in out]
    assert len(out) == 3                          # max_pages respected
    assert "https://ex.com/" in urls              # started at the root
    assert all("other.com" not in u for u in urls)  # never left the domain
    assert all(text.strip() for _, text in out)   # extracted readable text


def test_csv_and_xlsx_loaders(tmp_path):
    from rag_pipeline.ingest import LOADERS, load_chunks

    assert ".csv" in LOADERS and ".xlsx" in LOADERS
    csv_path = tmp_path / "faq.csv"
    csv_path.write_text("question,answer\nreturns?,within 30 days\n", encoding="utf-8")
    chunks = load_chunks([csv_path], tenant_id="t1")
    assert chunks
    assert "within 30 days" in chunks[0].text
    assert chunks[0].metadata["tenant_id"] == "t1"
