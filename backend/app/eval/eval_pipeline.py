from dataclasses import dataclass
from typing import Any, Literal

from backend.app.config import settings
from backend.app.generation.llm import generate_answer, rewrite_query
from backend.app.retrieval.embeddings import (
    build_dense_embedding_model,
    build_sparse_embedding_model,
    embed_query_dense,
    embed_query_sparse,
)
from backend.app.retrieval.hybrid import hybrid_search
from backend.app.retrieval.qdrant_store import (
    create_collection_if_not_exists,
    get_qdrant_client,
    search_chunks_dense,
)
from backend.app.retrieval.reranker import build_reranker, rerank_chunks


EvalMode = Literal[
    "dense_only",
    "hybrid",
    "hybrid_plus_rerank",
    "hybrid_plus_rerank_plus_rewrite",
]


MODE_LABELS = {
    "dense_only": "Dense Only",
    "hybrid": "Hybrid",
    "hybrid_plus_rerank": "Hybrid + Rerank",
    "hybrid_plus_rerank_plus_rewrite": "Hybrid + Rerank + Rewrite",
}


@dataclass
class EvalRuntime:
    """
    Long-lived runtime dependencies used during evaluation.
    """

    qdrant_client: Any
    dense_embedding_model: Any
    sparse_embedding_model: Any
    reranker: Any


def build_eval_runtime() -> EvalRuntime:
    """
    Load Qdrant, embedding models, sparse embedding model, and reranker once
    for an evaluation run.

    Returns:
        EvalRuntime containing all reusable eval dependencies.
    """
    qdrant_client = get_qdrant_client()
    create_collection_if_not_exists(qdrant_client, settings.dense_vector_size)

    dense_embedding_model = build_dense_embedding_model(
        settings.dense_embedding_model_checkpoint
    )

    sparse_embedding_model = build_sparse_embedding_model(
        settings.sparse_embedding_model_checkpoint
    )

    reranker = build_reranker(settings.cross_encoder_model_checkpoint)

    return EvalRuntime(
        qdrant_client=qdrant_client,
        dense_embedding_model=dense_embedding_model,
        sparse_embedding_model=sparse_embedding_model,
        reranker=reranker,
    )


def safe_float(value: Any) -> float | None:
    """
    Convert a value to float when possible.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_scores(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add simple normalized retrieval/rerank scores to eval chunks.

    This mirrors the app's display-oriented score normalization without
    importing the FastAPI app module.
    """
    if not chunks:
        return []

    retrieval_scores = [
        safe_float(chunk.get("retrieval_score"))
        for chunk in chunks
        if safe_float(chunk.get("retrieval_score")) is not None
    ]

    top_retrieval = max(retrieval_scores, default=None)

    rerank_scores = [
        safe_float(chunk.get("rerank_score"))
        for chunk in chunks
        if safe_float(chunk.get("rerank_score")) is not None
    ]

    min_rerank = min(rerank_scores) if rerank_scores else None
    max_rerank = max(rerank_scores) if rerank_scores else None

    normalized = []

    for chunk in chunks:
        retrieval_score = safe_float(chunk.get("retrieval_score"))
        rerank_score = safe_float(chunk.get("rerank_score"))

        retrieval_norm = None
        if retrieval_score is not None and top_retrieval not in (None, 0):
            retrieval_norm = max(0.0, min(1.0, retrieval_score / top_retrieval))

        rerank_norm = None
        if (
            rerank_score is not None
            and min_rerank is not None
            and max_rerank is not None
            and max_rerank != min_rerank
        ):
            rerank_norm = (rerank_score - min_rerank) / (max_rerank - min_rerank)
        elif rerank_score is not None:
            rerank_norm = 1.0

        normalized.append(
            {
                **chunk,
                "retrieval_score_normalized": retrieval_norm,
                "rerank_score_normalized": rerank_norm,
            }
        )

    return normalized


def build_basic_confidence(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a lightweight eval confidence payload from retrieval and rerank scores.

    Like ``normalize_scores`` above, this mirrors the production confidence
    helper in ``backend/app/main.py`` (``build_answer_confidence``) without
    importing the FastAPI app module. The eval version omits the judge-score
    term because the grounding judge runs in a separate eval step
    (see ``run_eval.py``).
    """
    retrieval_values = [
        safe_float(chunk.get("retrieval_score_normalized"))
        for chunk in chunks
        if safe_float(chunk.get("retrieval_score_normalized")) is not None
    ]

    rerank_values = [
        safe_float(chunk.get("rerank_score_normalized"))
        for chunk in chunks
        if safe_float(chunk.get("rerank_score_normalized")) is not None
    ]

    retrieval_score = (
        sum(retrieval_values) / len(retrieval_values)
        if retrieval_values
        else None
    )

    rerank_score = (
        sum(rerank_values) / len(rerank_values)
        if rerank_values
        else None
    )

    available = [
        value for value in [retrieval_score, rerank_score]
        if value is not None
    ]

    score = sum(available) / len(available) if available else 0.0

    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    else:
        label = "Low"

    return {
        "label": label,
        "score": score,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
    }


def format_eval_citation(chunk: dict[str, Any]) -> str:
    """
    Format a retrieved chunk citation for eval outputs.
    """
    doc_name = chunk.get("doc_name", "Document")
    page_start = chunk.get("page_start") or chunk.get("page")
    page_end = chunk.get("page_end") or page_start

    if page_start and page_end and page_start != page_end:
        return f"{doc_name} pp. {page_start}-{page_end}"

    if page_start:
        return f"{doc_name} p. {page_start}"

    return doc_name


def retrieve_contexts_for_mode(
    question: str,
    mode: EvalMode,
    runtime: EvalRuntime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Retrieve contexts for one eval question under a selected retrieval mode.

    Args:
        question: Evaluation question.
        mode: One of the four Phase 2 eval modes.
        runtime: Loaded eval runtime dependencies.

    Returns:
        Tuple of retrieved chunks and query metadata.
    """
    query_metadata = {
        "original_query": question,
        "rewritten_query": None,
        "rewrite_used": False,
        "rewrite_error": None,
    }

    retrieval_query = question

    if mode == "hybrid_plus_rerank_plus_rewrite":
        query_info = rewrite_query(question)
        retrieval_query = query_info.get("rewritten_query", question)

        query_metadata.update(
            {
                "rewritten_query": retrieval_query,
                "rewrite_used": query_info.get("rewrite_used", False),
                "rewrite_error": query_info.get("rewrite_error"),
            }
        )

    dense_query = embed_query_dense(
        retrieval_query,
        runtime.dense_embedding_model,
    )

    if mode == "dense_only":
        chunks = search_chunks_dense(
            client=runtime.qdrant_client,
            query_embedding=dense_query,
            limit=settings.rerank_top_k,
        )

        chunks = [
            {
                **chunk,
                "retrieval_score": float(chunk.get("dense_score", 0.0)),
            }
            for chunk in chunks
        ]

        return normalize_scores(chunks), query_metadata

    sparse_query = embed_query_sparse(
        retrieval_query,
        runtime.sparse_embedding_model,
    )

    retrieved_chunks = hybrid_search(
        client=runtime.qdrant_client,
        dense_query=dense_query,
        sparse_query=sparse_query,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        fusion_top_k=settings.fusion_top_k,
    )

    if mode == "hybrid":
        chunks = retrieved_chunks[: settings.rerank_top_k]
        return normalize_scores(chunks), query_metadata

    reranked_chunks = rerank_chunks(
        query=question,
        chunks=retrieved_chunks,
        reranker=runtime.reranker,
        top_k=settings.rerank_top_k,
    )

    return normalize_scores(reranked_chunks), query_metadata


def run_eval_example(
    example: dict[str, Any],
    mode: EvalMode,
    runtime: EvalRuntime,
) -> dict[str, Any]:
    """
    Run one dataset example through one Lucid eval mode.

    Args:
        example: One item from eval_dataset.json.
        mode: Evaluation mode.
        runtime: Loaded eval runtime.

    Returns:
        Detailed result with question, answer, contexts, citations, scores,
        and query metadata.
    """
    question = example["question"]

    chunks, query_metadata = retrieve_contexts_for_mode(
        question=question,
        mode=mode,
        runtime=runtime,
    )

    answer = generate_answer(
        question=question,
        chunks=chunks,
        max_output_tokens=settings.max_output_tokens,
    )

    citations = [format_eval_citation(chunk) for chunk in chunks]

    return {
        "id": example["id"],
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "document": example["document"],
        "topic": example["topic"],
        "difficulty": example["difficulty"],
        "question_type": example["question_type"],
        "question": question,
        "ground_truth": example["ground_truth"],
        "expected_source_hint": example.get("expected_source_hint"),
        "answer": answer,
        "contexts": [chunk.get("text", "") for chunk in chunks],
        "citations": citations,
        "hits": chunks,
        "confidence": build_basic_confidence(chunks),
        **query_metadata,
    }