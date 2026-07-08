# semantic-search-rag

A production-shaped **Retrieval-Augmented Generation (RAG)** pipeline for
semantic search over unstructured data. It combines **vector embeddings** for
retrieval with **advanced prompt engineering** for grounded, cited answers.

- **Embeddings:** [OpenAI](https://platform.openai.com/) (`text-embedding-3-small` by default; `-large` supported). Voyage AI is available as an alternative provider.
- **Generation:** OpenAI (`gpt-4o-mini` by default) via Chat Completions, streamed with inline citations. Claude (Messages API, with adaptive thinking + prompt caching) is a drop-in alternative via `RAG_GENERATION_PROVIDER=anthropic`.
- **Hybrid search:** dense (vector) + sparse (BM25 / Postgres full-text) fused with Reciprocal Rank Fusion, then an LLM **re-ranker** trims the pool to the best chunks. Toggle with `RAG_HYBRID` / `RAG_RERANK`.
- **Multi-source ingest:** `.pdf`, `.txt`, `.md`, `.csv`, `.xlsx`, and **web URLs**.
- **Multi-tenant (workspaces):** every chunk carries a `tenant_id`; retrieval is filtered so each user/customer only sees their own knowledge base.
- **Vector store:** pluggable backend — a zero-dependency NumPy store for local/offline use, or **Postgres + pgvector / Supabase** for deployment (`RAG_STORE_BACKEND=pgvector`).
- **Chat UI:** a [Streamlit](https://streamlit.io/) app (`app.py`) with light/dark themes, conversation memory, a custom bot persona, citations, and a **human-handoff** prompt when the bot must abstain.
- **Analytics:** logs each question and whether it was answered vs. abstained — surfacing the content gaps a workspace should fill.
- **Deploy:** one-command `docker compose up` (app + pgvector), or point `DATABASE_URL` at Supabase.
- **Offline mode:** a deterministic hashed embedder lets you run ingest, retrieval, and the full test suite with **no API keys**.

---

## How it works

```
              ┌────────────── ingest (once) ──────────────┐
  files ─▶ load ─▶ chunk ─▶ embed(document) ─▶ VectorStore.save()
 (.txt/.md/.pdf)   │          (OpenAI)              │
                   └── overlapping, paragraph-aware ┘

              ┌────────────── query (per request) ────────┐
  question ─▶ [HyDE rewrite?] ─▶ embed(query)
                   │                    │
                   │            dense search (top candidate_k)
                   │            sparse/BM25 search (top candidate_k)
                   │                    │
                   └──▶ Reciprocal Rank Fusion ─▶ rerank ─▶ top-k
                                                     │
                                                     ▼
             build grounded prompt (numbered <source> blocks
             + citation contract + abstain-if-unsupported)
                                                     │
                                                     ▼
                    LLM (OpenAI / Claude, streamed) ─▶ answer
                                                     │
                                                     ▼
                     parse [n] markers ─▶ resolved Citations
```

The pieces map 1:1 to modules in [`src/rag_pipeline/`](src/rag_pipeline):

| Module | Responsibility |
|---|---|
| [`chunking.py`](src/rag_pipeline/chunking.py) | Paragraph-aware, overlapping chunking |
| [`embeddings.py`](src/rag_pipeline/embeddings.py) | OpenAI / Voyage embedders + offline hashed fallback |
| [`vector_store.py`](src/rag_pipeline/vector_store.py) | NumPy cosine search + BM25, save/load |
| [`pg_store.py`](src/rag_pipeline/pg_store.py) | Postgres + pgvector backend (dense + full-text) |
| [`store_factory.py`](src/rag_pipeline/store_factory.py) | Pick the backend from `RAG_STORE_BACKEND` |
| [`fusion.py`](src/rag_pipeline/fusion.py) | Reciprocal Rank Fusion for hybrid results |
| [`reranker.py`](src/rag_pipeline/reranker.py) | LLM re-ranker (pluggable) |
| [`ingest.py`](src/rag_pipeline/ingest.py) | File loaders (txt/md/pdf) → chunks → store |
| [`retriever.py`](src/rag_pipeline/retriever.py) | Dense + sparse hybrid, HyDE, rerank |
| [`prompts.py`](src/rag_pipeline/prompts.py) | The grounding contract & prompt assembly |
| [`generator.py`](src/rag_pipeline/generator.py) | OpenAI / Claude backends, streaming, citations |
| [`pipeline.py`](src/rag_pipeline/pipeline.py) | Orchestration (`RAGPipeline`) |
| [`cli.py`](src/rag_pipeline/cli.py) | `rag-ingest` / `rag-query` |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
cp .env.example .env             # add OPENAI_API_KEY (used for embeddings + generation)
```

## Web UI

```bash
streamlit run app.py
```

A chat app with **workspaces** (isolated knowledge bases), multi-source ingest
(PDF/CSV/Excel/URL), light/dark themes, conversation memory, a custom bot
persona, inline citations, and a human-handoff prompt when the bot can't answer
from the docs. Pick a workspace name, add sources, and chat.

## Deploy (Postgres + pgvector)

The NumPy store is a single local file — great for a demo, wrong for a deployed
service. For deployment, switch to the **pgvector** backend: it holds both the
vector index and the full-text index, so hybrid search needs no extra service.

**One command locally** (spins up Postgres+pgvector and the app):

```bash
cp .env.example .env             # set OPENAI_API_KEY
docker compose up --build        # → http://localhost:8501
```

**Supabase (free tier)** — the recommended managed option:

1. Create a project at [supabase.com](https://supabase.com).
2. **Database → Extensions →** enable `vector`.
3. **Project Settings → Database → Connection string (URI)** — copy it.
4. Point the app at it:

```bash
export RAG_STORE_BACKEND=pgvector
export DATABASE_URL='postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require'
streamlit run app.py             # table + indexes are created on first ingest
```

[Neon](https://neon.tech) works identically. The table (HNSW vector index + GIN
full-text index + a `tenant_id` column for workspace isolation) is created
automatically on first ingest — no manual schema step.

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

- **Models:** `RAG_GENERATION_PROVIDER`, `RAG_GENERATION_MODEL`, `RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`
- **Search:** `RAG_HYBRID`, `RAG_RERANK`, `RAG_RERANK_MODEL`, `RAG_TOP_K`, `RAG_CANDIDATE_K`, `RAG_MIN_SCORE`
- **Store:** `RAG_STORE_BACKEND` (`numpy`|`pgvector`), `DATABASE_URL`, `RAG_PG_TABLE`, `RAG_STORE_PATH`
- **Chunking:** `RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`

## Extending

- **Swap the vector DB:** implement the `add` / `search` / `keyword_search` /
  `save` interface (see `VectorStore` and `PgVectorStore`) over Pinecone, Qdrant,
  Chroma, etc., and wire it into `store_factory.py`.
- **Add a file type:** register a loader in `LOADERS` in `ingest.py`.
- **Swap the re-ranker:** implement the `Reranker` protocol in `reranker.py`
  (e.g. a local cross-encoder or Cohere Rerank) and add it to `build_reranker`.

## Tests

```bash
pytest -q          # offline tests: chunking, embedder, store, BM25, RRF fusion, retrieval, citations
```

## License

MIT
