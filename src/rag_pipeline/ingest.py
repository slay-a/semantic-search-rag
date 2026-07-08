"""Load unstructured files from disk and index them into the vector store.

Supported formats: ``.txt``, ``.md``, and ``.pdf``. New loaders can be added by
extending :data:`LOADERS`. Everything funnels through :func:`chunk_document`, so
the rest of the pipeline never sees a file — only :class:`Chunk` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .chunking import chunk_document
from .config import Settings, settings
from .embeddings import Embedder, build_embedder
from .types import Chunk
from .vector_store import VectorStore

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("pypdf is required to ingest PDF files.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


LOADERS: dict[str, Callable[[Path], str]] = {
    **{suffix: _load_text for suffix in TEXT_SUFFIXES},
    ".pdf": _load_pdf,
}


def discover_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a flat, sorted list of loadable files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(
                f for f in sorted(p.rglob("*")) if f.suffix.lower() in LOADERS
            )
        elif p.is_file() and p.suffix.lower() in LOADERS:
            found.append(p)
    return found


def load_chunks(paths: Iterable[str | Path], cfg: Settings = settings) -> list[Chunk]:
    """Read every supported file under ``paths`` into chunks."""
    chunks: list[Chunk] = []
    for path in discover_files(paths):
        loader = LOADERS[path.suffix.lower()]
        text = loader(path).strip()
        if not text:
            continue
        chunks.extend(
            chunk_document(
                text,
                source=str(path),
                chunk_tokens=cfg.chunk_tokens,
                chunk_overlap=cfg.chunk_overlap,
                base_metadata={"filename": path.name, "suffix": path.suffix.lower()},
            )
        )
    return chunks


def build_store(
    paths: Iterable[str | Path],
    *,
    cfg: Settings = settings,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> VectorStore:
    """Load, embed, and index files into a (new or existing) vector store."""
    embedder = embedder or build_embedder(cfg)
    store = store or VectorStore()

    chunks = load_chunks(paths, cfg)
    if not chunks:
        raise ValueError(f"No ingestible content found under: {list(paths)}")

    vectors = embedder.embed([c.text for c in chunks], input_type="document")
    store.add(chunks, vectors)
    return store
