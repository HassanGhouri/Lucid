from typing import Any

from backend.app.retrieval.qdrant_store import search_chunks_dense, search_chunks_sparse
from backend.app.observability.langsmith_tracing import traceable


def rrf_fuse(
    result_lists: list[list[dict[str, Any]]],
    k: int,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked retrieval result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Ranked retrieval result lists.
        k: RRF smoothing constant.

    Returns:
        Fused and reranked result list with retrieval scores and preserved
        dense/sparse scores when available.
    """
    scores = {}
    objects = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            point_id = hit["id"]

            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)

            if point_id not in objects:
                objects[point_id] = dict(hit)
            else:
                objects[point_id].update(
                    {
                        key: value
                        for key, value in hit.items()
                        if key in {"dense_score", "sparse_score"}
                    }
                )

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    return [
        {
            **objects[point_id],
            "retrieval_score": float(scores[point_id]),
            "rrf_score": float(scores[point_id]),
        }
        for point_id in ranked_ids
    ]


@traceable(name="Hybrid Retrieval", run_type="retriever")
def hybrid_search(
    client: Any,
    dense_query: Any,
    sparse_query: Any,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    fusion_top_k: int,
    tag_name: str | None = None,
    doc_name: str | None = None,
    source_type: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run dense and sparse retrieval, then combine results using RRF.

    Args:
        client: Active Qdrant client.
        dense_query: Dense embedded query.
        sparse_query: Sparse embedded query.
        dense_top_k: Number of dense results to retrieve.
        sparse_top_k: Number of sparse results to retrieve.
        rrf_k: Reciprocal Rank Fusion smoothing constant.
        fusion_top_k: Number of fused chunks to return.
        tag_name: Optional tag filter.
        doc_name: Optional document filter.
        source_type: Optional source-type filter.
        session_id: Optional anonymous browser session ID for private uploads.

    Returns:
        Fused retrieval results with retrieval_score, dense_score, and/or
        sparse_score where available.
    """
    dense_hits = search_chunks_dense(
        client,
        dense_query,
        tag_name,
        doc_name,
        source_type,
        session_id,
        limit=dense_top_k,
    )

    sparse_hits = search_chunks_sparse(
        client,
        sparse_query,
        tag_name,
        doc_name,
        source_type,
        session_id,
        limit=sparse_top_k,
    )

    fused = rrf_fuse(
        [dense_hits, sparse_hits],
        k=rrf_k,
    )

    return fused[:fusion_top_k]
