from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import logging
from functools import lru_cache

from backend.app.observability.langsmith_tracing import traceable

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_device() -> str:
    """
    Pick the best torch device for embedding and reranker models.

    Preference order: CUDA -> CPU. MPS (Apple Silicon) is intentionally
    skipped — CrossEncoder rerank on MPS is catastrophically slow under
    concurrency, so CPU is strictly better on Mac dev machines. The deploy
    target is Linux CPU; CUDA branch exists so the system gracefully uses
    a GPU if one is present (resume signal), not because it's needed.

    Cached so the selection (and its log line) is computed once per
    process even though both build_dense_embedding_model and build_reranker
    call this.
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Selected device: cuda")
            return "cuda"
    except Exception as exc:
        logger.debug("CUDA probe failed, falling back to CPU: %s", exc)
    logger.info("Selected device: cpu")
    return "cpu"


def build_sparse_embedding_model(sparse_embedding_model_checkpoint: str) -> SparseTextEmbedding:
    """
    Load a sparse embedding model (BM25-style).

    Args:
        sparse_embedding_model_checkpoint: Name of the sparse model, such as
            "Qdrant/bm25".

    Returns:
        Loaded sparse embedding model.
    """
    return SparseTextEmbedding(model_name=sparse_embedding_model_checkpoint)


def build_dense_embedding_model(dense_embedding_model_checkpoint: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer embedding model on the detected device.

    Args:
        dense_embedding_model_checkpoint: Hugging Face/SentenceTransformers
            model name or local path.

    Returns:
        Loaded SentenceTransformer model placed on CUDA when available, CPU otherwise.
    """
    return SentenceTransformer(
        dense_embedding_model_checkpoint,
        device=detect_device(),
    )


def embed_texts_dense(
    texts: Sequence[str],
    embedding_model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    """
    Embed texts using a SentenceTransformer model.

    Args:
        texts: Text strings to embed.
        embedding_model: Loaded SentenceTransformer model.
        batch_size: Number of texts to encode per batch.

    Returns:
        Float32 NumPy array of normalized embeddings.
    """
    embeddings = embedding_model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


@traceable(name="Dense Query Embedding", run_type="embedding")
def embed_query_dense(
    query: str,
    embedding_model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed a single search query using a SentenceTransformer model.

    This is a convenience wrapper for one query string and returns a single
    normalized embedding vector. Useful during retrieval when converting a
    user question into a vector for semantic search in Qdrant.

    Args:
        query: User search query or question to embed.
        embedding_model: Loaded SentenceTransformer model.

    Returns:
        Float32 NumPy array representing the query embedding.
    """
    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )
    return embedding.astype("float32")


def embed_texts_sparse(
    texts: Sequence[str],
    sparse_model: SparseTextEmbedding,
):
    """
    Generate sparse embeddings for a list of texts.

    Args:
        texts: Text strings to embed.
        sparse_model: Loaded sparse embedding model.

    Returns:
        List of sparse vectors (indices + values).
    """
    return list(sparse_model.embed(texts))


@traceable(name="Sparse Query Embedding", run_type="embedding")
def embed_query_sparse(
    query: str,
    sparse_model: SparseTextEmbedding,
):
    """
    Generate sparse embedding for a single query.

    Args:
        query: User search query or question to embed.
        sparse_model: Loaded sparse embedding model.

    Returns:
        Sparse vector for the query.
    """
    return embed_texts_sparse([query], sparse_model)[0]
