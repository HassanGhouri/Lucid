import uuid
from typing import Any, Sequence
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.app.config import settings
from backend.app.retrieval.filters import (
    PRIVATE_VISIBILITY,
    build_chunk_filter,
)

COLLECTION_NAME = settings.qdrant_collection_name


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client connected to the configured Qdrant server.

    Returns:
        A QdrantClient instance connected to the Qdrant URL from settings. When
        QDRANT_API_KEY is configured, it is passed through for Qdrant Cloud.
    """
    client_kwargs = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        client_kwargs["api_key"] = settings.qdrant_api_key

    return QdrantClient(**client_kwargs)

def create_collection_if_not_exists(client: QdrantClient, dense_vector_size: int) -> None:
    """
    Create the Lucid Qdrant collection if it does not already exist.

    The collection stores one vector per document chunk using cosine similarity,
    which is appropriate for normalized SentenceTransformer embeddings.

    Args:
        client: Active Qdrant client.
        dense_vector_size: Size of dense vectors.

    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                settings.dense_vector_name: VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.sparse_vector_name: SparseVectorParams()
            },
        )

def _stable_point_id(chunk: dict[str, Any]) -> str:
    """
    Create a deterministic UUID for a chunk.

    Deterministic IDs make re-ingestion safer because the same document chunk
    can overwrite its previous Qdrant point instead of creating duplicates.

    Args:
        chunk: Chunk dictionary containing an id or enough metadata to build one.

    Returns:
        UUID string suitable for use as a Qdrant point ID.
    """
    raw_id = chunk.get("id") or (
        f"{chunk.get('owner_id') or 'public'}::"
        f"{chunk.get('doc_name', 'unknown')}::"
        f"{chunk.get('page_start', chunk.get('page', 'unknown'))}::"
        f"{chunk.get('chunk_id', 'unknown')}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

def upsert_chunks(
    client: QdrantClient,
    chunks: Sequence[dict[str, Any]],
    dense_embeddings: np.ndarray,
    sparse_embeddings: Sequence[Any]
) -> None:
    """
    Insert or update embedded document chunks in Qdrant.

    Each chunk is stored as a Qdrant point containing an embedding vector and
    payload metadata. Lucid uses the payload for citations and filtered
    retrieval by tag, document, and source type.

    Args:
        client: Active Qdrant client.
        chunks: Sequence of chunk dictionaries containing text and metadata.
        dense_embeddings: NumPy array of dense embedding vectors aligned with the chunks.
        sparse_embeddings: Sparse embedding vectors aligned with the chunks.

    Raises:
        ValueError: If chunks and embeddings are not aligned positionally.
    """
    if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_embeddings):
        raise ValueError(
            "Chunks, dense embeddings, and sparse embeddings must have equal lengths."
        )

    points = []

    for chunk, dense_embedding, sparse_embedding in zip(
        chunks,
        dense_embeddings,
        sparse_embeddings,
    ):
        point = PointStruct(
            id=_stable_point_id(chunk),
            vector={
                settings.dense_vector_name: dense_embedding.tolist(),
                settings.sparse_vector_name: {
                    "indices": sparse_embedding.indices,
                    "values": sparse_embedding.values,
                },
            },
            payload={
                "text": chunk["text"],

                "document": chunk.get("document"),
                "topic": chunk.get("topic"),

                "doc_name": chunk["doc_name"],
                "page": chunk.get("page"),
                "page_start": chunk.get("page_start", chunk.get("page")),
                "page_end": chunk.get("page_end", chunk.get("page")),
                "chunk_id": chunk["chunk_id"],
                "tag_name": chunk.get("tag_name", "Untagged"),
                "tag_color": chunk.get("tag_color", "#64748B"),
                "source_type": chunk.get("source_type", "unknown"),
                "visibility": chunk.get("visibility", "public"),
                "owner_id": chunk.get("owner_id"),
                "uploaded_at": chunk.get("uploaded_at"),
                "uploaded_at_epoch": chunk.get("uploaded_at_epoch"),
                "pdf_filename": chunk.get("pdf_filename", chunk.get("doc_name")),
            },
        )
        points.append(point)

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )


def search_chunks_dense(
    client: QdrantClient,
    query_embedding: np.ndarray | list[float],
    tag_name: str | None = None,
    doc_name: str | None = None,
    source_type: str | None = None,
    session_id: str | None = None,
    limit: int = settings.dense_top_k,
) -> list[dict[str, Any]]:
    """
    Search Qdrant for chunks semantically similar to a query embedding.

    Optional metadata filters restrict search to selected tags, documents, or
    source types before returning the top matching chunks.

    Args:
        client: Active Qdrant client.
        query_embedding: Embedded user query.
        tag_name: Optional tag filter, such as "CSC420" or "Midterm".
        doc_name: Optional document filter, such as "week3_cnn.pdf".
        source_type: Optional source type filter, such as "lecture" or "exam".
        session_id: Optional anonymous browser session ID for private uploads.
        limit: Number of chunks to retrieve from Qdrant before reranking.

    Returns:
        List of chunk dictionaries containing text, metadata, and Qdrant scores.
    """
    query_vector = (
        query_embedding.tolist()
        if isinstance(query_embedding, np.ndarray)
        else query_embedding
    )

    scored_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=settings.dense_vector_name,
        query_filter=build_chunk_filter(
            tag_name=tag_name,
            doc_name=doc_name,
            source_type=source_type,
            session_id=session_id,
        ),
        with_payload=True,
        limit=limit,
    ).points

    chunks = []
    for point in scored_points:
        payload = point.payload or {}
        chunks.append(
            {
                **payload,
                "id": str(point.id),
                "dense_score": float(point.score),
            }
        )

    return chunks

def search_chunks_sparse(
    client: QdrantClient,
    sparse_query: Any,
    tag_name: str | None = None,
    doc_name: str | None = None,
    source_type: str | None = None,
    session_id: str | None = None,
    limit: int = settings.sparse_top_k,
) -> list[dict[str, Any]]:
    """
    Search Qdrant using a sparse BM25-style query vector.

    Args:
        client: Active Qdrant client.
        sparse_query: Sparse query vector produced by FastEmbed.
        tag_name: Optional tag filter.
        doc_name: Optional document filter.
        source_type: Optional source type filter.
        session_id: Optional anonymous browser session ID for private uploads.
        limit: Number of sparse results to return.

    Returns:
        List of chunk dictionaries containing text, metadata, and sparse scores.
    """
    scored_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        ),
        using=settings.sparse_vector_name,
        query_filter=build_chunk_filter(
            tag_name=tag_name,
            doc_name=doc_name,
            source_type=source_type,
            session_id=session_id,
        ),
        with_payload=True,
        limit=limit,
    ).points

    chunks = []
    for point in scored_points:
        payload = point.payload or {}
        chunks.append(
            {
                **payload,
                "id": str(point.id),
                "sparse_score": float(point.score),
            }
        )

    return chunks


def recreate_collection(client: QdrantClient, dense_vector_size: int) -> None:
    """
    Delete and recreate the Qdrant collection.

    WARNING: This deletes all currently ingested chunks.

    Args:
        client: Active Qdrant client.
        dense_vector_size: Size of dense vectors in the recreated collection.
    """
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    create_collection_if_not_exists(client, dense_vector_size)

def delete_document_chunks(client: QdrantClient, doc_name: str) -> None:
    """
    Delete all chunks belonging to a specific document from Qdrant.

    Args:
        client: Active Qdrant client.
        doc_name: Name of the document whose chunks should be deleted.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=build_chunk_filter(doc_name=doc_name),
    )


def _private_owner_filter(owner_id: str) -> Filter:
    """Return a filter matching private chunks owned by one anonymous session."""
    return Filter(
        must=[
            FieldCondition(
                key="visibility",
                match=MatchValue(value=PRIVATE_VISIBILITY),
            ),
            FieldCondition(
                key="owner_id",
                match=MatchValue(value=owner_id),
            ),
        ]
    )


def _private_owner_document_filter(owner_id: str, doc_name: str) -> Filter:
    """Return a filter matching one private document owned by a session."""
    return Filter(
        must=[
            FieldCondition(
                key="visibility",
                match=MatchValue(value=PRIVATE_VISIBILITY),
            ),
            FieldCondition(
                key="owner_id",
                match=MatchValue(value=owner_id),
            ),
            FieldCondition(
                key="doc_name",
                match=MatchValue(value=doc_name),
            ),
        ]
    )


def _expired_private_filter(cutoff_epoch: float) -> Filter:
    """Return a filter matching private chunks older than the cutoff epoch."""
    return Filter(
        must=[
            FieldCondition(
                key="visibility",
                match=MatchValue(value=PRIVATE_VISIBILITY),
            ),
            FieldCondition(
                key="uploaded_at_epoch",
                range=Range(lt=cutoff_epoch),
            ),
        ]
    )


def delete_owner_chunks(client: QdrantClient, owner_id: str) -> None:
    """
    Delete all private chunks owned by one anonymous session.

    Args:
        client: Active Qdrant client.
        owner_id: Anonymous browser session ID whose private chunks to delete.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=_private_owner_filter(owner_id),
    )


def delete_owner_document_chunks(
    client: QdrantClient,
    owner_id: str,
    doc_name: str,
) -> None:
    """
    Delete one private document owned by an anonymous session.

    Args:
        client: Active Qdrant client.
        owner_id: Anonymous browser session ID.
        doc_name: User-facing document name to delete.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=_private_owner_document_filter(owner_id, doc_name),
    )


def delete_expired_private_chunks(client: QdrantClient, cutoff_epoch: float) -> None:
    """
    Delete private chunks older than the supplied Unix timestamp.

    Args:
        client: Active Qdrant client.
        cutoff_epoch: Unix timestamp; private chunks older than this are deleted.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=_expired_private_filter(cutoff_epoch),
    )


def collection_has_points(client: QdrantClient) -> bool:
    """
    Return whether the Lucid collection currently contains any chunks.

    Args:
        client: Active Qdrant client.

    Returns:
        True if at least one point exists in the collection, otherwise False.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    return bool(collection_info.points_count and collection_info.points_count > 0)


def list_payload_options(
    client: QdrantClient,
    session_id: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """
    List available tags, documents, and source types from visible Qdrant payloads.

    This supports frontend dropdowns without needing a separate metadata
    database. For larger production apps, these values could later be moved to a
    relational database, but Qdrant payload scrolling is enough for Lucid.
    Public chunks are visible to everyone; private chunks are visible only to
    the matching anonymous session.

    Args:
        client: Active Qdrant client.
        session_id: Optional anonymous browser session ID for private uploads.

    Returns:
        Dictionary with sorted tag, document, and source type options.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return {"tags": [], "documents": [], "source_types": []}

    tags: dict[str, str] = {}
    documents: set[str] = set()
    source_types: set[str] = set()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=build_chunk_filter(session_id=session_id),
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            tag_name = payload.get("tag_name")
            if tag_name:
                tags[tag_name] = payload.get("tag_color", "#64748B")

            doc_name = payload.get("doc_name")
            if doc_name:
                documents.add(doc_name)

            source_type = payload.get("source_type")
            if source_type:
                source_types.add(source_type)

        if next_offset is None:
            break

    return {
        "tags": [
            {"name": name, "color": tags[name]}
            for name in sorted(tags.keys(), key=str.lower)
        ],
        "documents": sorted(documents, key=str.lower),
        "source_types": sorted(source_types, key=str.lower),
    }


def scroll_chunks_for_filter(
    client: QdrantClient,
    doc_name: str,
    tag_name: str | None = None,
    session_id: str | None = None,
    limit: int = 512,
) -> list[dict[str, Any]]:
    """
    Scroll all chunks in the collection matching a doc_name (and optional tag).

    Args:
        client: Active Qdrant client.
        doc_name: Required document filter.
        tag_name: Optional tag filter.
        session_id: Optional anonymous browser session ID for private uploads.
        limit: Maximum number of points to return.

    Returns:
        List of payload dicts (each containing text, doc_name, page_start,
        page_end, tag_name, tag_color, source_type, etc.).
    """
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=build_chunk_filter(
            doc_name=doc_name,
            tag_name=tag_name,
            session_id=session_id,
        ),
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )

    return [dict(p.payload or {}) for p in points]


def list_private_pdf_filenames(
    client: QdrantClient,
    owner_id: str | None = None,
    doc_name: str | None = None,
    cutoff_epoch: float | None = None,
) -> set[str]:
    """
    List stored private PDF filenames matching owner, document, or age filters.

    Args:
        client: Active Qdrant client.
        owner_id: Optional anonymous session ID.
        doc_name: Optional user-facing document name.
        cutoff_epoch: Optional Unix timestamp; private chunks older than this
            timestamp are matched.

    Returns:
        Set of stored PDF filenames referenced by matching private chunks.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return set()

    conditions = [
        FieldCondition(
            key="visibility",
            match=MatchValue(value=PRIVATE_VISIBILITY),
        )
    ]

    if owner_id:
        conditions.append(
            FieldCondition(
                key="owner_id",
                match=MatchValue(value=owner_id),
            )
        )

    if doc_name:
        conditions.append(
            FieldCondition(
                key="doc_name",
                match=MatchValue(value=doc_name),
            )
        )

    if cutoff_epoch is not None:
        conditions.append(
            FieldCondition(
                key="uploaded_at_epoch",
                range=Range(lt=cutoff_epoch),
            )
        )

    filenames: set[str] = set()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=conditions),
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            pdf_filename = payload.get("pdf_filename")
            if pdf_filename:
                filenames.add(str(pdf_filename))

        if next_offset is None:
            break

    return filenames


def find_authorized_pdf_filename(
    client: QdrantClient,
    doc_name: str,
    session_id: str | None = None,
) -> str | None:
    """
    Return the stored PDF filename when the document is visible to the session.

    Args:
        client: Active Qdrant client.
        doc_name: User-facing document name.
        session_id: Optional anonymous browser session ID for private uploads.

    Returns:
        Stored PDF filename, or None when no visible chunks reference the doc.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return None

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=build_chunk_filter(
            doc_name=doc_name,
            session_id=session_id,
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    payload = points[0].payload or {}
    return str(payload.get("pdf_filename") or payload.get("doc_name") or doc_name)
