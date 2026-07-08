"""Reciprocal Rank Fusion (RRF) for combining hybrid retrieval results.

Dense (vector) and sparse (BM25/full-text) searches return their hits on
*incomparable* score scales — a cosine similarity of 0.7 and a BM25 score of 12
can't be added directly. RRF sidesteps this by ignoring the raw scores and
fusing on *rank* alone: an item's fused score is the sum, over every ranked list
it appears in, of ``1 / (k + rank)``. Items that rank highly in *either* list
(or modestly in *both*) float to the top, which is exactly the hybrid behaviour
we want.

``k`` (default 60, the value from the original RRF paper) damps the influence of
very high ranks so the tail of each list still contributes a little.
"""

from __future__ import annotations

from .types import ScoredChunk

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], *, k: int = RRF_K
) -> list[ScoredChunk]:
    """Fuse several ranked lists of chunks into one, keyed by chunk id.

    Each input list must already be ordered best-first. The returned list is
    ordered by fused RRF score (descending); each returned :class:`ScoredChunk`
    carries its fused score (not the original dense/sparse score).
    """
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, ScoredChunk] = {}

    for ranked in rankings:
        for rank, sc in enumerate(ranked):
            cid = sc.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # Keep the first ScoredChunk we see for each id (its text/source).
            chunk_by_id.setdefault(cid, sc)

    ordered_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [
        ScoredChunk(chunk=chunk_by_id[cid].chunk, score=fused[cid])
        for cid in ordered_ids
    ]
