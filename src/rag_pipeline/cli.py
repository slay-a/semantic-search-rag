"""Command-line entry points: ``rag-ingest`` and ``rag-query``.

Also runnable without an install:
    python -m rag_pipeline.cli ingest data/sample_docs
    python -m rag_pipeline.cli query "your question"
"""

from __future__ import annotations

import argparse
import sys

from .config import settings
from .ingest import build_store
from .pipeline import RAGPipeline
from .types import RAGAnswer
from .vector_store import VectorStore


def ingest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-ingest", description="Index files into the store.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("-o", "--out", default=settings.store_path, help="Output store path.")
    parser.add_argument(
        "--append", action="store_true", help="Add to an existing store instead of replacing it."
    )
    args = parser.parse_args(argv)

    existing = None
    if args.append:
        try:
            existing = VectorStore.load(args.out)
        except FileNotFoundError:
            existing = None

    print(f"Ingesting {args.paths} (provider={settings.embedding_provider}, "
          f"model={settings.embedding_model}) ...", file=sys.stderr)
    store = build_store(args.paths, store=existing)
    store.save(args.out)
    print(f"Indexed {len(store)} chunks -> {args.out}", file=sys.stderr)
    return 0


def _print_answer(answer: RAGAnswer) -> None:
    if answer.citations:
        print("\n\nSources:")
        for c in answer.citations:
            print(f"  [{c.marker}] {c.source}  (score {c.score:.3f})")
    if answer.usage:
        u = answer.usage
        print(
            f"\n[tokens in={u.get('input_tokens', 0)} "
            f"out={u.get('output_tokens', 0)} "
            f"cache_read={u.get('cache_read_input_tokens', 0)}]",
            file=sys.stderr,
        )


def query_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-query", description="Ask a grounded question.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("-s", "--store", default=settings.store_path, help="Store path.")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Chunks to retrieve.")
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE query rewriting.")
    parser.add_argument(
        "--no-stream", action="store_true", help="Wait for the full answer instead of streaming."
    )
    args = parser.parse_args(argv)

    rag = RAGPipeline.from_store(args.store, use_hyde=args.hyde)

    if args.no_stream:
        answer = rag.query(args.question, top_k=args.top_k)
        print(answer.answer)
    else:
        tokens, answer = rag.stream_query(args.question, top_k=args.top_k)
        for delta in tokens:
            sys.stdout.write(delta)
            sys.stdout.flush()

    _print_answer(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", add_help=False)
    sub.add_parser("query", add_help=False)
    args, rest = parser.parse_known_args(argv)

    if args.command == "ingest":
        return ingest_main(rest)
    if args.command == "query":
        return query_main(rest)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
