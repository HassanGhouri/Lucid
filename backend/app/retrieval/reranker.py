from sentence_transformers import CrossEncoder
from typing import Any

from backend.app.observability.langsmith_tracing import traceable
from backend.app.retrieval.embeddings import detect_device


def build_reranker(cross_encoder_model_name: str) -> CrossEncoder:
    """
    Load a cross-encoder reranker model on the detected device.

    Args:
        cross_encoder_model_name: Hugging Face/SentenceTransformers cross-encoder model name.

    Returns:
        Loaded CrossEncoder placed on CUDA when available, CPU otherwise.
    """
    return CrossEncoder(
        cross_encoder_model_name,
        device=detect_device(),
    )


@traceable(name="Reranking", run_type="chain")
def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query: User question.
        chunks: Retrieved chunk dictionaries from vector search.
        reranker: Loaded CrossEncoder reranker.
        top_k: Number of reranked chunks to return.

    Returns:
        Top reranked chunks with rerank scores.
    """
    if not chunks:
        return []

    valid_chunks = [chunk for chunk in chunks if chunk.get("text")]

    if not valid_chunks:
        return []

    pairs = [[query, chunk["text"]] for chunk in valid_chunks]
    scores = reranker.predict(pairs)

    reranked_chunks = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in zip(valid_chunks, scores)
    ]

    reranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked_chunks[:top_k]
