# semantic-search-rag

A production-shaped **Retrieval-Augmented Generation (RAG)** pipeline for
semantic search over unstructured data. It combines **vector embeddings** for
retrieval with **advanced prompt engineering** for grounded, cited answers from
**Claude**.

- **Embeddings:** [Voyage AI](https://www.voyageai.com/) (`voyage-3` by default) — Anthropic's recommended embedding provider — with asymmetric `query`/`document` input types for better retrieval.
- **Generation:** Claude `claude-opus-4-8` via the Messages API, with adaptive thinking, streaming, prompt caching, and inline citations.
- **Vector store:** an exact cosine-similarity store in NumPy — zero external services, swappable for a managed DB.
- **Offline mode:** a deterministic hashed embedder lets you run ingest, retrieval, and the full test suite with **no API keys**.

---

## How it works

```
              ┌────────────── ingest (once) ──────────────┐
  files ─▶ load ─▶ chunk ─▶ embed(document) ─▶ VectorStore.save()
 (.txt/.md/.pdf)   │          (Voyage)              │
                   └── overlapping, paragraph-aware ┘

              ┌────────────── query (per request) ────────┐
  question ─▶ [HyDE rewrite?] ─▶ embed(query) ─▶ search top-k
                                                     │
                                                     ▼
             build grounded prompt (numbered <source> blocks
             + citation contract + abstain-if-unsupported)
                                                     │
                                                     ▼
                Claude (adaptive thinking, streamed) ─▶ answer
                                                     │
                                                     ▼
                     parse [n] markers ─▶ resolved Citations
```

The pieces map 1:1 to modules in [`src/rag_pipeline/`](src/rag_pipeline):

| Module | Responsibility |
|---|---|
| [`chunking.py`](src/rag_pipeline/chunking.py) | Paragraph-aware, overlapping chunking |
| [`embeddings.py`](src/rag_pipeline/embeddings.py) | Voyage embedder + offline hashed fallback |
| [`vector_store.py`](src/rag_pipeline/vector_store.py) | Cosine search, save/load |
| [`ingest.py`](src/rag_pipeline/ingest.py) | File loaders (txt/md/pdf) → chunks → store |
| [`retriever.py`](src/rag_pipeline/retriever.py) | Query embedding + optional HyDE |
| [`prompts.py`](src/rag_pipeline/prompts.py) | The grounding contract & prompt assembly |
| [`generator.py`](src/rag_pipeline/generator.py) | Claude Messages API, streaming, citations |
| [`pipeline.py`](src/rag_pipeline/pipeline.py) | Orchestration (`RAGPipeline`) |
| [`cli.py`](src/rag_pipeline/cli.py) | `rag-ingest` / `rag-query` |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
cp .env.example .env             # add ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

## Quickstart

```bash
# 1. Index a folder of documents into a vector store
rag-ingest data/sample_docs -o store/index.npz

# 2. Ask a grounded, cited question (streams the answer)
rag-query "What does the API do when it receives a 429?" -s store/index.npz

# Optional: HyDE query rewriting for harder/terser questions
rag-query "idempotency guarantees?" --hyde
```

Without the console scripts installed, use the module form:

```bash
python -m rag_pipeline.cli ingest data/sample_docs
python -m rag_pipeline.cli query "your question"
```

## Library usage

```python
from rag_pipeline import RAGPipeline, build_store

build_store(["data/sample_docs"]).save("store/index.npz")   # ingest once

rag = RAGPipeline.from_store("store/index.npz")
answer = rag.query("How is the ledger kept consistent with payments?")

print(answer.answer)
for c in answer.citations:
    print(f"[{c.marker}] {c.source}  score={c.score:.3f}")
```

---

## Run it with no API keys (offline)

The hashed embedder makes ingest, retrieval, and tests fully offline. It has no
real semantic understanding — it's for wiring, demos, and CI, not production.

```bash
RAG_EMBEDDING_PROVIDER=hash rag-ingest data/sample_docs
RAG_EMBEDDING_PROVIDER=hash python examples/quickstart.py --retrieval-only
RAG_EMBEDDING_PROVIDER=hash pytest -q
```

---

## The prompt-engineering contract

Faithfulness comes from the prompt design in [`prompts.py`](src/rag_pipeline/prompts.py):

1. **Stable, cached system prompt** — fixes role, grounding rules, and citation
   format; byte-identical across requests so it's prompt-cached.
2. **Numbered `<source>` blocks** — each retrieved chunk is XML-delimited with an
   `[n]` marker, source path, and score. Claude follows XML structure reliably,
   and the markers are the handle for citation resolution.
3. **Abstain-when-unsupported** — the model is told to answer *only* from the
   sources and to say "I don't have enough information…" rather than invent one.
4. **Post-hoc citation resolution** — inline `[n]` markers are parsed and mapped
   back to the exact chunks, so every answer is auditable to its sources.

## Configuration

All knobs are environment variables (see [`.env.example`](.env.example)); defaults
live in [`config.py`](src/rag_pipeline/config.py). Common ones:
`RAG_GENERATION_MODEL`, `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_PROVIDER`,
`RAG_TOP_K`, `RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`, `RAG_EFFORT`, `RAG_STORE_PATH`.

## Extending

- **Swap the vector DB:** implement `add` / `search` / `save` / `load` (the same
  narrow interface as `VectorStore`) over pgvector, Pinecone, Chroma, etc.
- **Add a file type:** register a loader in `LOADERS` in `ingest.py`.
- **Add a re-ranker:** insert a cross-encoder pass between `retriever.retrieve`
  and `generator.answer` in `pipeline.py`.

## Tests

```bash
pytest -q          # 8 offline tests: chunking, embedder, store, retrieval, citations
```

## License

MIT
