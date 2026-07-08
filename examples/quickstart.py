"""End-to-end example: ingest the sample docs, then ask a grounded question.

Run with real providers:
    OPENAI_API_KEY=... python examples/quickstart.py

Run fully offline (no API keys, hashed embeddings, retrieval only):
    RAG_EMBEDDING_PROVIDER=hash python examples/quickstart.py --retrieval-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline import RAGPipeline, build_store
from rag_pipeline.retriever import Retriever
from rag_pipeline.embeddings import build_embedder

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation; just show what retrieval returns.",
    )
    parser.add_argument(
        "-q",
        "--question",
        default="What does the API do when it receives a 429 response?",
    )
    args = parser.parse_args()

    print(f"Ingesting {DOCS} ...")
    store = build_store([DOCS])
    print(f"Indexed {len(store)} chunks.\n")

    if args.retrieval_only:
        embedder = build_embedder()
        retriever = Retriever(store, embedder)
        for sc in retriever.retrieve(args.question):
            print(f"[{sc.score:.3f}] {sc.chunk.source}")
            print(f"        {sc.chunk.text[:120].strip()}...\n")
        return

    rag = RAGPipeline(store)
    print(f"Q: {args.question}\nA: ", end="", flush=True)
    tokens, answer = rag.stream_query(args.question)
    for delta in tokens:
        print(delta, end="", flush=True)
    print("\n\nCitations:")
    for c in answer.citations:
        print(f"  [{c.marker}] {c.source} (score {c.score:.3f})")


if __name__ == "__main__":
    main()
