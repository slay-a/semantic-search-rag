"""Semantic-search RAG pipeline over unstructured data (OpenAI, or Claude).

Public surface:

    from rag_pipeline import RAGPipeline, build_store, VectorStore

    store = build_store(["data/sample_docs"])          # ingest + embed
    store.save("store/index.npz")

    rag = RAGPipeline.from_store("store/index.npz")     # load
    answer = rag.query("What retry policy does the API use?")
    print(answer.answer)
    for c in answer.citations:
        print(c.marker, c.source, round(c.score, 3))
"""

from .config import Settings, settings
from .ingest import build_store, load_chunks
from .pipeline import RAGPipeline
from .types import Chunk, Citation, RAGAnswer, ScoredChunk
from .vector_store import VectorStore

__all__ = [
    "RAGPipeline",
    "VectorStore",
    "Settings",
    "settings",
    "build_store",
    "load_chunks",
    "Chunk",
    "ScoredChunk",
    "Citation",
    "RAGAnswer",
]

__version__ = "0.1.0"
