from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from typing import Any
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
import json
from fastapi.responses import StreamingResponse, FileResponse
import random
from starlette.requests import Request
from starlette.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.config import settings
from backend.app.generation.llm import (
    generate_answer,
    generate_flashcards,
    judge_grounding,
    rewrite_query,
    stream_answer,
)
from backend.app.ingestion.pipeline import ingest_pdf
from backend.app.retrieval.embeddings import (
    build_dense_embedding_model,
    build_sparse_embedding_model,
    embed_texts_dense,
    embed_texts_sparse,
    embed_query_dense,
    embed_query_sparse
)
from backend.app.retrieval.hybrid import hybrid_search
from backend.app.retrieval.qdrant_store import (
    collection_has_points,
    create_collection_if_not_exists,
    delete_expired_private_chunks,
    delete_owner_chunks,
    delete_owner_document_chunks,
    find_authorized_pdf_filename,
    get_qdrant_client,
    list_private_pdf_filenames,
    list_payload_options,
    scroll_chunks_for_filter,
    upsert_chunks,
)
from backend.app.retrieval.reranker import build_reranker, rerank_chunks
from backend.app.generation.prompts import build_rag_prompt
from backend.app.observability.langsmith_tracing import (
    configure_langsmith_from_settings,
    is_tracing_enabled,
    get_trace_url,
    safe_add_metadata,
    safe_add_outputs,
    sanitize_chunks,
    trace_span,
    traceable,
    truncate_text,
)
from backend.app.observability.metrics import (
    Timer,
    record_request,
    set_cold_start_ms,
    snapshot as metrics_snapshot,
)


logger = logging.getLogger(__name__)

qdrant_client = None
dense_embedding_model = None
sparse_embedding_model = None
reranker = None
tokenizer = None
last_private_upload_cleanup_epoch = 0.0

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")

limiter = Limiter(key_func=get_remote_address)

class AskRequest(BaseModel):
    """Request body for a Lucid question."""

    question: str
    tag_name: str | None = None
    doc_name: str | None = None
    source_type: str | None = None
    history: list[dict] | None = None
    rewrite_enabled: bool = True
    session_id: str | None = None

class FlashcardRequest(BaseModel):
    """Request body for generating flashcards from a document."""

    doc_name: str
    tag_name: str | None = None
    num_cards: int = 10
    session_id: str | None = None


class ClearSessionUploadsRequest(BaseModel):
    """Request body for clearing private uploads for an anonymous session."""

    session_id: str


def _normalize_filter_value(value: str | None) -> str | None:
    """
    Convert frontend "All"/empty values into None for backend filtering.

    Args:
        value: Raw filter value from the frontend.

    Returns:
        None when no filter should be applied, otherwise the stripped value.
    """
    if value is None:
        return None

    value = value.strip()
    if value == "" or value == "All":
        return None

    return value


def _normalize_session_id(session_id: str | None) -> str | None:
    """
    Validate an anonymous session ID from the frontend.

    Args:
        session_id: Raw session ID from query params, form data, or JSON.

    Returns:
        Cleaned session ID, or None when no session was supplied.

    Raises:
        HTTPException: If the supplied session ID is malformed.
    """
    if session_id is None:
        return None

    cleaned = session_id.strip()
    if not cleaned:
        return None

    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Invalid session_id.")

    return cleaned


def _build_private_pdf_filename(original_filename: str, session_id: str) -> str:
    """
    Build a collision-resistant stored filename for a private uploaded PDF.

    Args:
        original_filename: User-facing PDF filename.
        session_id: Anonymous session that owns the upload.

    Returns:
        Safe filename used for storage on disk.
    """
    safe_name = Path(original_filename or "uploaded.pdf").name
    stem = Path(safe_name).stem or "uploaded"
    suffix = Path(safe_name).suffix or ".pdf"
    return f"{stem}-{session_id[:8]}-{uuid.uuid4().hex[:12]}{suffix}"


def _delete_uploaded_pdf_files(pdf_filenames: set[str]) -> int:
    """
    Delete stored private PDFs from the upload directory.

    Args:
        pdf_filenames: Stored PDF filenames referenced by private Qdrant chunks.

    Returns:
        Number of files deleted.
    """
    upload_dir = Path("data/uploads").resolve()
    deleted = 0

    for filename in pdf_filenames:
        safe_filename = Path(filename).name
        if not safe_filename or safe_filename.startswith("."):
            continue

        target = (upload_dir / safe_filename).resolve()
        try:
            target.relative_to(upload_dir)
        except ValueError:
            continue

        if target.is_file():
            target.unlink()
            deleted += 1

    return deleted


def _run_private_upload_cleanup_if_due() -> dict[str, int | bool]:
    """
    Opportunistically delete expired private upload chunks and PDF files.

    Returns:
        Cleanup summary with whether a run occurred and how many files were
        deleted.
    """
    global last_private_upload_cleanup_epoch

    now = time.time()
    if (
        now - last_private_upload_cleanup_epoch
        < settings.private_upload_cleanup_interval_seconds
    ):
        return {"ran": False, "deleted_files": 0}

    cutoff_epoch = now - (settings.private_upload_ttl_hours * 3600)
    deleted_files = 0

    try:
        expired_pdf_filenames = list_private_pdf_filenames(
            qdrant_client,
            cutoff_epoch=cutoff_epoch,
        )
        delete_expired_private_chunks(qdrant_client, cutoff_epoch)
        deleted_files = _delete_uploaded_pdf_files(expired_pdf_filenames)

        if expired_pdf_filenames:
            logger.info(
                "Private upload cleanup removed %s expired PDF files.",
                deleted_files,
            )
    except Exception as exc:
        # Swallow so a transient Qdrant error doesn't fail the user's request;
        # the cleanup is opportunistic and will retry after the next interval.
        logger.warning("Private upload cleanup failed (non-fatal): %s", exc)
    finally:
        # Always advance the epoch, even on failure, so a persistently broken
        # cleanup path doesn't hammer Qdrant on every request.
        last_private_upload_cleanup_epoch = now

    return {"ran": True, "deleted_files": deleted_files}


def format_chunk_citation(chunk: dict[str, Any]) -> str:
    """
    Format a retrieved chunk as a compact user-facing citation.

    Args:
        chunk: Retrieved chunk dictionary containing document and page metadata.

    Returns:
        Citation string such as "week3_cnn.pdf p. 12".
    """
    doc_name = chunk.get("doc_name", "Document")
    page_start = chunk.get("page_start") or chunk.get("page")
    page_end = chunk.get("page_end") or page_start

    if page_start and page_end and page_start != page_end:
        return f"{doc_name} pp. {page_start}-{page_end}"

    if page_start:
        return f"{doc_name} p. {page_start}"

    return doc_name

def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float when possible.

    Args:
        value: Unknown value.

    Returns:
        Float value, or None if conversion fails.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _average_available(values: list[float | None]) -> float | None:
    """
    Average only values that are not None.

    Args:
        values: Possible numeric values.

    Returns:
        Average of available values, or None.
    """
    available = [value for value in values if value is not None]

    if not available:
        return None

    return sum(available) / len(available)


def _normalize_against_top(
    value: float | None,
    top_value: float | None,
) -> float | None:
    """
    Normalize a score against the top score in the current result set.

    Args:
        value: Score to normalize.
        top_value: Best score in the result set.

    Returns:
        Normalized score between 0 and 1, or None.
    """
    if value is None or top_value is None or top_value == 0:
        return None

    return max(0.0, min(1.0, value / top_value))


def _add_normalized_scores(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add normalized retrieval and rerank scores to chunks.

    Args:
        chunks: Retrieved or reranked chunks.

    Returns:
        Chunks with normalized score fields added.
    """
    if not chunks:
        return []

    retrieval_scores = []
    for chunk in chunks:
        retrieval_score = _safe_float(chunk.get("retrieval_score"))
        if retrieval_score is not None:
            retrieval_scores.append(retrieval_score)

    top_retrieval_score = max(retrieval_scores, default=None)

    rerank_scores = []
    for chunk in chunks:
        rerank_score = _safe_float(chunk.get("rerank_score"))
        if rerank_score is not None:
            rerank_scores.append(rerank_score)

    min_rerank_score = min(rerank_scores) if rerank_scores else None
    max_rerank_score = max(rerank_scores) if rerank_scores else None

    normalized_chunks = []

    for chunk in chunks:
        retrieval_score = _safe_float(chunk.get("retrieval_score"))
        rerank_score = _safe_float(chunk.get("rerank_score"))

        normalized_retrieval = _normalize_against_top(
            retrieval_score,
            top_retrieval_score,
        )

        if (
            rerank_score is not None
            and min_rerank_score is not None
            and max_rerank_score is not None
            and max_rerank_score != min_rerank_score
        ):
            normalized_rerank = (rerank_score - min_rerank_score) / (
                max_rerank_score - min_rerank_score
            )
        elif rerank_score is not None:
            normalized_rerank = 1.0
        else:
            normalized_rerank = None

        normalized_chunks.append(
            {
                **chunk,
                "retrieval_score_normalized": normalized_retrieval,
                "rerank_score_normalized": normalized_rerank,
            }
        )

    return normalized_chunks


def _confidence_label(score: float) -> str:
    """
    Convert a numeric confidence score into a user-facing label.

    Args:
        score: Confidence score between 0 and 1.

    Returns:
        High, Medium, or Low.
    """
    if score >= 0.75:
        return "High"

    if score >= 0.45:
        return "Medium"

    return "Low"


@traceable(name="Confidence Build", run_type="chain")
def build_answer_confidence(
    chunks: list[dict[str, Any]],
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine retrieval, rerank, and optional judge scores into final confidence.

    Args:
        chunks: Reranked chunks with normalized score fields.
        judge_result: Output from the LLM grounding judge.

    Returns:
        Confidence payload for the frontend.
    """
    retrieval_score = _average_available(
        [
            _safe_float(chunk.get("retrieval_score_normalized"))
            for chunk in chunks
        ]
    )

    rerank_score = _average_available(
        [
            _safe_float(chunk.get("rerank_score_normalized"))
            for chunk in chunks
        ]
    )

    judge_score = (
        _safe_float(judge_result.get("score"))
        if judge_result.get("available")
        else None
    )

    if judge_score is not None:
        combined_score = (
            0.30 * (retrieval_score if retrieval_score is not None else 0.0)
            + 0.30 * (rerank_score if rerank_score is not None else 0.0)
            + 0.40 * judge_score
        )
        used_judge = True
    else:
        combined_score = (
            0.45 * (retrieval_score if retrieval_score is not None else 0.0)
            + 0.55 * (rerank_score if rerank_score is not None else 0.0)
        )
        used_judge = False

    combined_score = max(0.0, min(1.0, combined_score))

    return {
        "label": _confidence_label(combined_score),
        "score": combined_score,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
        "judge_score": judge_score,
        "judge_available": used_judge,
        "judge_reason": judge_result.get("reason"),
        "judge_error": judge_result.get("error"),
    }


def _build_qdrant_chunks(
    chunk_records: list[dict[str, Any]],
    doc_name: str,
    tag_name: str,
    tag_color: str,
    source_type: str,
    visibility: str = "public",
    owner_id: str | None = None,
    uploaded_at: str | None = None,
    uploaded_at_epoch: float | None = None,
    pdf_filename: str | None = None,
) -> list[dict[str, Any]]:
    """
    Combine chunk text, ingestion metadata, and upload metadata for Qdrant.

    Args:
        chunk_records: Chunk dictionaries produced by ingestion, each containing text, page metadata, and a chunk_id.
        doc_name: Display name for the uploaded document.
        tag_name: User-selected or newly created tag name.
        tag_color: Color associated with the tag.
        source_type: Source type selected during ingestion.
        visibility: Whether chunks are public demo data or private uploads.
        owner_id: Anonymous session ID for private uploads.
        uploaded_at: ISO timestamp for temporary upload cleanup.
        uploaded_at_epoch: Unix timestamp for temporary upload cleanup.
        pdf_filename: Stored PDF filename used for citation click-through.

    Returns:
        List of Qdrant-ready chunk dictionaries.
    """
    qdrant_chunks = []

    for i, record in enumerate(chunk_records):
        chunk_id = record.get("chunk_id", i)
        page = record.get("page") or record.get("page_start")

        qdrant_chunks.append(
            {
                **record,
                "id": (
                    f"{owner_id}::{doc_name}::{chunk_id}"
                    if owner_id
                    else f"{doc_name}::{chunk_id}"
                ),
                "text": record["text"],
                "doc_name": doc_name,
                "page": page,
                "page_start": record.get("page_start", page),
                "page_end": record.get("page_end", page),
                "chunk_id": chunk_id,
                "tag_name": tag_name,
                "tag_color": tag_color,
                "source_type": source_type,
                "visibility": visibility,
                "owner_id": owner_id,
                "uploaded_at": uploaded_at,
                "uploaded_at_epoch": uploaded_at_epoch,
                "pdf_filename": pdf_filename or doc_name,
            }
        )

    return qdrant_chunks


def load_runtime() -> None:
    """
    Load long-lived runtime dependencies for Lucid.

    This initializes the embedding model, reranker, Qdrant client, and Qdrant
    collection once at application startup.
    """
    global qdrant_client, dense_embedding_model, sparse_embedding_model, reranker, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings.dense_embedding_model_checkpoint,
        use_fast=True,
    )

    dense_embedding_model = build_dense_embedding_model(
        settings.dense_embedding_model_checkpoint
    )

    sparse_embedding_model = build_sparse_embedding_model(
        settings.sparse_embedding_model_checkpoint
    )

    reranker = build_reranker(settings.cross_encoder_model_checkpoint)
    # Pre-warm: force PyTorch's lazy kernel init (kernel compilation,
    # autograd graph setup, attention cache) here instead of letting it
    # hit the first concurrent request. Cost gets folded into the
    # cold_start_ms metric; warmup failure is non-fatal because the
    # model itself loaded fine and real .predict() calls will retry.
    try:
        reranker.predict([["warmup query", "warmup document text"]])
    except Exception as exc:
        logger.warning("CrossEncoder warmup failed (non-fatal): %s", exc)
    qdrant_client = get_qdrant_client()
    create_collection_if_not_exists(qdrant_client, settings.dense_vector_size)


def _runtime_models_loaded() -> bool:
    """Return whether all lifespan-loaded model dependencies are available."""
    return all(
        dependency is not None
        for dependency in (
            dense_embedding_model,
            sparse_embedding_model,
            reranker,
            tokenizer,
        )
    )


def _parse_cors_allowed_origins(raw_origins: str) -> list[str]:
    """
    Parse comma-separated CORS origins from settings.

    Args:
        raw_origins: Comma-separated origins, or "*" for development.

    Returns:
        List of origins suitable for FastAPI CORSMiddleware.
    """
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_langsmith_from_settings()
    if is_tracing_enabled():
        logger.info(
            "LangSmith tracing enabled (project=%s)",
            settings.langsmith_project,
        )
    else:
        logger.info("LangSmith tracing disabled.")
    with Timer() as t:
        load_runtime()
    set_cold_start_ms(t.elapsed_ms)
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_allowed_origins(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """
    Apply a per-path request timeout so a hung request doesn't pin a worker
    indefinitely under concurrency. Ingest gets a longer budget than ask
    endpoints because legitimate large-PDF ingestion is slow.

    Note: a timeout cancels the asyncio task but does not kill threads
    spawned via asyncio.to_thread. Those threads complete naturally and are
    then released. The OpenAI client's per-call timeout handles the most
    common hang path (slow LLM).
    """
    path = request.url.path
    if path.startswith("/ingest_pdf"):
        timeout = settings.ingest_timeout_seconds
    else:
        timeout = settings.request_timeout_seconds

    try:
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Request to %s exceeded %.1fs and was timed out.", path, timeout
        )
        return JSONResponse(
            status_code=504,
            content={"detail": f"Request timed out after {timeout:.0f}s."},
        )


@app.get("/metrics")
async def metrics():
    """
    Return cold-start and per-stage P50/P95 latency from the in-process
    ring buffer. Per-worker; one worker only for benchmarking.
    """
    return metrics_snapshot()


@app.get("/healthz")
async def healthz():
    """
    Return process readiness for model preload and Qdrant reachability.

    Returns:
        Health payload with status, model preload state, and Qdrant reachability.
    """
    model_loaded = _runtime_models_loaded()
    try:
        qdrant_reachable = await asyncio.to_thread(collection_has_points, qdrant_client)
    except Exception as exc:
        logger.warning("Health check could not reach Qdrant: %s", exc)
        qdrant_reachable = False

    is_healthy = model_loaded and qdrant_reachable
    payload = {
        "status": "ok" if is_healthy else "unhealthy",
        "model_loaded": model_loaded,
        "qdrant_reachable": qdrant_reachable,
    }

    if not is_healthy:
        return JSONResponse(status_code=503, content=payload)

    return payload


@app.get("/filter_options")
async def filter_options(session_id: str | None = None):
    """
    Return available tag, document, and source-type options for the frontend.

    Args:
        session_id: Optional anonymous session ID for private uploaded docs.

    Returns:
        Dictionary containing tags, documents, and source types discovered from
        Qdrant payloads visible to this session.
    """
    normalized_session_id = _normalize_session_id(session_id)
    await asyncio.to_thread(_run_private_upload_cleanup_if_due)
    return await asyncio.to_thread(
        list_payload_options,
        qdrant_client,
        normalized_session_id,
    )


@app.post("/ingest_pdf")
@limiter.limit(settings.rate_limit_ingest)
async def ingest_pdf_endpoint(
    request: Request,
    file: UploadFile = File(...),
    tag_name: str = Form("Untagged"),
    tag_color: str = Form("#64748B"),
    source_type: str = Form("unknown"),
    session_id: str | None = Form(None),
):
    """
    Ingest a PDF, embed its chunks, and store them in Qdrant with tag metadata.

    Args:
        file: Uploaded PDF file.
        tag_name: User-created or selected tag for the document.
        tag_color: Hex color associated with the tag.
        source_type: User-selected source type, such as lecture, exam, or notes.
        session_id: Optional anonymous session ID. When supplied, the upload is
            private to that session.

    Returns:
        Status response with document and chunk counts.
    """
    logger.info("Ingest request received")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF.")

    cleaned_tag_name = _normalize_filter_value(tag_name) or "Untagged"
    cleaned_source_type = _normalize_filter_value(source_type) or "unknown"
    normalized_session_id = _normalize_session_id(session_id)
    await asyncio.to_thread(_run_private_upload_cleanup_if_due)

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Upload directory ready")

    safe_filename = Path(file.filename or "uploaded.pdf").name
    doc_name = safe_filename
    visibility = "private" if normalized_session_id else "public"
    owner_id = normalized_session_id
    uploaded_at_dt = datetime.now(timezone.utc) if owner_id else None
    uploaded_at = uploaded_at_dt.isoformat() if uploaded_at_dt else None
    uploaded_at_epoch = uploaded_at_dt.timestamp() if uploaded_at_dt else None

    if owner_id:
        old_pdf_filenames = await asyncio.to_thread(
            list_private_pdf_filenames,
            qdrant_client,
            owner_id,
            doc_name,
        )
        await asyncio.to_thread(
            delete_owner_document_chunks,
            qdrant_client,
            owner_id,
            doc_name,
        )
        await asyncio.to_thread(_delete_uploaded_pdf_files, old_pdf_filenames)
        stored_pdf_filename = _build_private_pdf_filename(safe_filename, owner_id)
    else:
        stored_pdf_filename = safe_filename

    pdf_path = upload_dir / stored_pdf_filename
    pdf_path.write_bytes(await file.read())

    logger.info("PDF saved: %s", pdf_path)

    def run_ingestion() -> list[dict[str, Any]]:
        return list(
            ingest_pdf(
                pdf_path=pdf_path,
                tokenizer=tokenizer,
                max_tokens=settings.max_tokens_per_chunk,
                overlap_paragraphs=settings.overlap_paragraphs,
            )
        )

    chunk_records = await asyncio.to_thread(run_ingestion)

    if not chunk_records:
        if pdf_path.is_file():
            pdf_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="No text chunks could be extracted from this PDF.",
        )

    chunk_texts = [record["text"] for record in chunk_records]

    logger.info("Chunks created: %s", len(chunk_texts))

    dense_embeddings = embed_texts_dense(
        texts=chunk_texts,
        embedding_model=dense_embedding_model,
        batch_size=settings.dense_embedding_batch_size,
    )

    sparse_embeddings = embed_texts_sparse(
        texts=chunk_texts,
        sparse_model=sparse_embedding_model,
    )
    logger.info("Dense embeddings created with shape: %s", dense_embeddings.shape)
    logger.info("Sparse embeddings created: %s vectors", len(sparse_embeddings))

    qdrant_chunks = _build_qdrant_chunks(
        chunk_records=chunk_records,
        doc_name=doc_name,
        tag_name=cleaned_tag_name,
        tag_color=tag_color,
        source_type=cleaned_source_type,
        visibility=visibility,
        owner_id=owner_id,
        uploaded_at=uploaded_at,
        uploaded_at_epoch=uploaded_at_epoch,
        pdf_filename=stored_pdf_filename,
    )

    upsert_chunks(
        client=qdrant_client,
        chunks=qdrant_chunks,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    logger.info("Chunks upserted to Qdrant")

    return {
        "status": "ok",
        "chunks": len(chunk_texts),
        "doc_name": doc_name,
        "tag_name": cleaned_tag_name,
        "tag_color": tag_color,
        "source_type": cleaned_source_type,
        "visibility": visibility,
    }


@app.get("/pdfs/{doc_name}")
async def get_pdf(doc_name: str, session_id: str | None = None):
    """
    Serve a stored PDF for citation click-through.

    Files are read from the same `data/uploads/` directory that the
    ingest endpoint writes to. The browser handles range requests,
    caching, and the `#page=N` jump via PDF Open Parameters.

    Args:
        doc_name: User-facing document name.
        session_id: Optional anonymous session ID for private uploaded docs.
    """
    safe_name = Path(doc_name).name
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid document name.")

    normalized_session_id = _normalize_session_id(session_id)
    stored_pdf_filename = await asyncio.to_thread(
        find_authorized_pdf_filename,
        qdrant_client,
        safe_name,
        normalized_session_id,
    )

    if not stored_pdf_filename:
        raise HTTPException(status_code=404, detail="Document not found.")

    upload_dir = Path("data/uploads").resolve()
    safe_stored_filename = Path(stored_pdf_filename).name
    target = (upload_dir / safe_stored_filename).resolve()

    try:
        target.relative_to(upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document path.")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    return FileResponse(
        target,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.post("/clear_session_uploads")
async def clear_session_uploads(body: ClearSessionUploadsRequest):
    """
    Delete private uploaded documents for one anonymous browser session.

    Args:
        body: Request body containing the anonymous session ID.

    Returns:
        Deletion summary. Public demo documents are never deleted.
    """
    normalized_session_id = _normalize_session_id(body.session_id)
    if not normalized_session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    pdf_filenames = await asyncio.to_thread(
        list_private_pdf_filenames,
        qdrant_client,
        normalized_session_id,
    )
    await asyncio.to_thread(delete_owner_chunks, qdrant_client, normalized_session_id)
    deleted_files = await asyncio.to_thread(_delete_uploaded_pdf_files, pdf_filenames)

    return {
        "status": "ok",
        "deleted_pdf_files": deleted_files,
        "cleared_session": True,
    }


@app.post("/ask_question")
@limiter.limit(settings.rate_limit_ask)
async def ask_question(request: Request, body: AskRequest):
    """
    Answer a user question using query rewriting, filtered hybrid retrieval,
    reranking, grounded answer generation, and confidence scoring.
    """
    normalized_session_id = _normalize_session_id(body.session_id)
    timings: dict[str, float] = {}
    with Timer() as total_timer:
        try:
            with trace_span(
                name="Lucid Ask Request",
                run_type="chain",
                inputs={
                    "question": truncate_text(body.question),
                    "tag_name": body.tag_name,
                    "doc_name": body.doc_name,
                    "source_type": body.source_type,
                    "has_session_id": normalized_session_id is not None,
                },
            ) as parent_span:
                if not await asyncio.to_thread(collection_has_points, qdrant_client):
                    safe_add_metadata(parent_span, {"early_exit": "no_index"})
                    raise HTTPException(status_code=400, detail="No PDF indexed yet.")

                if body.rewrite_enabled:
                    with Timer() as t:
                        query_info = await asyncio.to_thread(
                            rewrite_query,
                            body.question,
                            history=body.history,
                        )
                    timings["rewrite"] = t.elapsed_ms
                else:
                    cleaned = body.question.strip()
                    query_info = {
                        "original_query": cleaned,
                        "rewritten_query": cleaned,
                        "rewrite_used": False,
                        "rewrite_error": None,
                    }

                original_query = query_info["original_query"]
                rewritten_query = query_info["rewritten_query"]

                safe_add_metadata(parent_span, {
                    "rewrite_enabled": body.rewrite_enabled,
                    "original_query": truncate_text(original_query),
                    "rewritten_query": truncate_text(rewritten_query),
                    "rewrite_used": query_info.get("rewrite_used", False),
                    "rewrite_error": query_info.get("rewrite_error"),
                })

                with Timer() as t:
                    dense_query = await asyncio.to_thread(
                        embed_query_dense, rewritten_query, dense_embedding_model
                    )
                    sparse_query = await asyncio.to_thread(
                        embed_query_sparse, rewritten_query, sparse_embedding_model
                    )
                timings["embed_query"] = t.elapsed_ms

                with Timer() as t:
                    retrieved_chunks = await asyncio.to_thread(
                        hybrid_search,
                        client=qdrant_client,
                        dense_query=dense_query,
                        sparse_query=sparse_query,
                        dense_top_k=settings.dense_top_k,
                        sparse_top_k=settings.sparse_top_k,
                        rrf_k=settings.rrf_k,
                        fusion_top_k=settings.fusion_top_k,
                        tag_name=_normalize_filter_value(body.tag_name),
                        doc_name=_normalize_filter_value(body.doc_name),
                        source_type=_normalize_filter_value(body.source_type),
                        session_id=normalized_session_id,
                    )
                timings["retrieve"] = t.elapsed_ms

                if not retrieved_chunks:
                    safe_add_metadata(parent_span, {"early_exit": "no_retrieval_results"})
                    return {
                        "answer": "I don't know based on the provided document.",
                        "citations": [],
                        "hits": [],
                        "original_query": original_query,
                        "rewritten_query": rewritten_query,
                        "rewrite_used": query_info.get("rewrite_used", False),
                        "confidence": {
                            "label": "Low",
                            "score": 0.0,
                            "retrieval_score": None,
                            "rerank_score": None,
                            "judge_score": None,
                            "judge_available": False,
                            "judge_reason": "No retrieved chunks.",
                            "judge_error": None,
                        },
                    }

                with Timer() as t:
                    reranked_chunks = await asyncio.to_thread(
                        rerank_chunks,
                        query=original_query,
                        chunks=retrieved_chunks,
                        reranker=reranker,
                        top_k=settings.rerank_top_k,
                    )
                timings["rerank"] = t.elapsed_ms

                reranked_chunks = _add_normalized_scores(reranked_chunks)

                with trace_span(
                    name="Prompt Build",
                    run_type="chain",
                    inputs={"question": truncate_text(original_query)},
                ) as prompt_span:
                    prompt_preview = build_rag_prompt(
                        question=original_query,
                        chunks=reranked_chunks,
                    )
                    safe_add_outputs(prompt_span, {
                        "prompt": truncate_text(prompt_preview),
                        "num_chunks": len(reranked_chunks),
                    })

                with Timer() as t:
                    answer = await asyncio.to_thread(
                        generate_answer,
                        original_query,
                        reranked_chunks,
                        settings.max_output_tokens,
                        history=body.history,
                    )
                timings["generate"] = t.elapsed_ms

                citations = [format_chunk_citation(chunk) for chunk in reranked_chunks]

                if "I don't know based on the provided document" in answer:
                    judge_result = {
                        "available": False,
                        "score": None,
                        "label": None,
                        "reason": "Answer refused due to insufficient document evidence.",
                        "error": None,
                    }
                    confidence = {
                        "label": "Low",
                        "score": 0.0,
                        "retrieval_score": None,
                        "rerank_score": None,
                        "judge_score": None,
                        "judge_available": False,
                        "judge_reason": judge_result["reason"],
                        "judge_error": None,
                    }
                else:
                    with Timer() as t:
                        judge_result = await asyncio.to_thread(
                            judge_grounding,
                            original_query,
                            answer,
                            reranked_chunks,
                        )
                    timings["judge"] = t.elapsed_ms

                    confidence = build_answer_confidence(
                        chunks=reranked_chunks,
                        judge_result=judge_result,
                    )

                safe_add_outputs(parent_span, {
                    "answer": truncate_text(answer),
                    "num_citations": len(citations),
                    "confidence_label": confidence.get("label"),
                    "confidence_score": confidence.get("score"),
                    "retrieved_chunks": sanitize_chunks(retrieved_chunks),
                    "reranked_chunks": sanitize_chunks(reranked_chunks),
                    "judge_available": judge_result.get("available"),
                    "judge_error": judge_result.get("error"),
                })

                return {
                    "answer": answer,
                    "citations": citations,
                    "hits": reranked_chunks,
                    "original_query": original_query,
                    "rewritten_query": rewritten_query,
                    "rewrite_used": query_info.get("rewrite_used", False),
                    "rewrite_error": query_info.get("rewrite_error"),
                    "confidence": confidence,
                    "trace_url": get_trace_url(parent_span),
                }
        finally:
            if timings:  # skip recording on pre-stage HTTPException (e.g. no_index)
                timings["total"] = total_timer.elapsed_ms
                record_request(timings)


@app.post("/ask_question_stream")
@limiter.limit(settings.rate_limit_ask)
async def ask_question_stream(request: Request, body: AskRequest):
    """
    Streamed variant of /ask_question.

    Emits newline-delimited JSON events:
      - {"type":"metadata", citations, hits, original_query, rewritten_query, ...}
      - {"type":"token", "text":"..."}    (one per text delta)
      - {"type":"error", "message":"..."} (only on mid-stream failure)
      - {"type":"done", "answer":"...", "confidence":{...}}

    The non-streaming /ask_question endpoint remains the canonical contract
    and is unchanged. This endpoint is purely additive.
    """
    normalized_session_id = _normalize_session_id(body.session_id)
    if not await asyncio.to_thread(collection_has_points, qdrant_client):
        raise HTTPException(status_code=400, detail="No PDF indexed yet.")

    def event_generator():
        timings: dict[str, float] = {}
        with Timer() as total_timer:
            try:
                with trace_span(
                        name="Lucid Ask Request (stream)",
                        run_type="chain",
                        inputs={
                            "question": truncate_text(body.question),
                            "tag_name": body.tag_name,
                            "doc_name": body.doc_name,
                            "source_type": body.source_type,
                            "stream": True,
                            "has_session_id": normalized_session_id is not None,
                        },
                ) as parent_span:
                    try:
                        if body.rewrite_enabled:
                            with Timer() as t:
                                query_info = rewrite_query(body.question, history=body.history)
                            timings["rewrite"] = t.elapsed_ms
                        else:
                            cleaned = body.question.strip()
                            query_info = {
                                "original_query": cleaned,
                                "rewritten_query": cleaned,
                                "rewrite_used": False,
                                "rewrite_error": None,
                            }

                        original_query = query_info["original_query"]
                        rewritten_query = query_info["rewritten_query"]

                        safe_add_metadata(parent_span, {
                            "history_turns": len(body.history) if body.history else 0,
                            "rewrite_enabled": body.rewrite_enabled,
                            "original_query": truncate_text(original_query),
                            "rewritten_query": truncate_text(rewritten_query),
                            "rewrite_used": query_info.get("rewrite_used", False),
                            "rewrite_error": query_info.get("rewrite_error"),
                        })

                        with Timer() as t:
                            dense_query = embed_query_dense(rewritten_query, dense_embedding_model)
                            sparse_query = embed_query_sparse(rewritten_query, sparse_embedding_model)
                        timings["embed_query"] = t.elapsed_ms

                        with Timer() as t:
                            retrieved_chunks = hybrid_search(
                                client=qdrant_client,
                                dense_query=dense_query,
                                sparse_query=sparse_query,
                                dense_top_k=settings.dense_top_k,
                                sparse_top_k=settings.sparse_top_k,
                                rrf_k=settings.rrf_k,
                                fusion_top_k=settings.fusion_top_k,
                                tag_name=_normalize_filter_value(body.tag_name),
                                doc_name=_normalize_filter_value(body.doc_name),
                                source_type=_normalize_filter_value(body.source_type),
                                session_id=normalized_session_id,
                            )
                        timings["retrieve"] = t.elapsed_ms

                        # No retrieval results: stream the same "I don't know" payload
                        # the non-streaming endpoint returns, just framed as events.
                        if not retrieved_chunks:
                            safe_add_metadata(parent_span, {"early_exit": "no_retrieval_results"})

                            no_results_answer = "I don't know based on the provided document."
                            confidence = {
                                "label": "Low",
                                "score": 0.0,
                                "retrieval_score": None,
                                "rerank_score": None,
                                "judge_score": None,
                                "judge_available": False,
                                "judge_reason": "No retrieved chunks.",
                                "judge_error": None,
                            }

                            yield json.dumps({
                                "type": "metadata",
                                "citations": [],
                                "hits": [],
                                "original_query": original_query,
                                "rewritten_query": rewritten_query,
                                "rewrite_used": query_info.get("rewrite_used", False),
                                "rewrite_error": query_info.get("rewrite_error"),
                            }) + "\n"

                            yield json.dumps({"type": "token", "text": no_results_answer}) + "\n"

                            yield json.dumps({
                                "type": "done",
                                "answer": no_results_answer,
                                "confidence": confidence,
                            }) + "\n"

                            safe_add_outputs(parent_span, {
                                "answer": no_results_answer,
                                "num_citations": 0,
                                "confidence_label": "Low",
                                "confidence_score": 0.0,
                            })
                            return

                        with Timer() as t:
                            reranked_chunks = rerank_chunks(
                                query=original_query,
                                chunks=retrieved_chunks,
                                reranker=reranker,
                                top_k=settings.rerank_top_k,
                            )
                        timings["rerank"] = t.elapsed_ms
                        reranked_chunks = _add_normalized_scores(reranked_chunks)

                        with trace_span(
                                name="Prompt Build",
                                run_type="chain",
                                inputs={"question": truncate_text(original_query)},
                        ) as prompt_span:
                            prompt_preview = build_rag_prompt(
                                question=original_query,
                                chunks=reranked_chunks,
                            )
                            safe_add_outputs(prompt_span, {
                                "prompt": truncate_text(prompt_preview),
                                "num_chunks": len(reranked_chunks),
                            })

                        citations = [format_chunk_citation(chunk) for chunk in reranked_chunks]

                        yield json.dumps({
                            "type": "metadata",
                            "citations": citations,
                            "hits": reranked_chunks,
                            "original_query": original_query,
                            "rewritten_query": rewritten_query,
                            "rewrite_used": query_info.get("rewrite_used", False),
                            "rewrite_error": query_info.get("rewrite_error"),
                            "trace_url": get_trace_url(parent_span),
                        }) + "\n"

                        with Timer() as t:
                            accumulated: list[str] = []
                            try:
                                for delta in stream_answer(
                                        original_query,
                                        reranked_chunks,
                                        settings.max_output_tokens,
                                        history=body.history,
                                ):
                                    accumulated.append(delta)
                                    yield json.dumps({"type": "token", "text": delta}) + "\n"
                            except Exception as exc:
                                logger.warning("Streaming generation error: %s", exc)
                                yield json.dumps({
                                    "type": "error",
                                    "message": "Answer generation was interrupted.",
                                }) + "\n"
                        timings["generate"] = t.elapsed_ms

                        full_answer = "".join(accumulated).strip()

                        if not full_answer:
                            full_answer = "An unexpected error occurred. Please try again."

                        if "I don't know based on the provided document" in full_answer:
                            confidence = {
                                "label": "Low",
                                "score": 0.0,
                                "retrieval_score": None,
                                "rerank_score": None,
                                "judge_score": None,
                                "judge_available": False,
                                "judge_reason": "Answer refused due to insufficient document evidence.",
                                "judge_error": None,
                            }
                        else:
                            with Timer() as t:
                                judge_result = judge_grounding(
                                    original_query,
                                    full_answer,
                                    reranked_chunks,
                                )
                            timings["judge"] = t.elapsed_ms
                            confidence = build_answer_confidence(
                                chunks=reranked_chunks,
                                judge_result=judge_result,
                            )

                        safe_add_outputs(parent_span, {
                            "answer": truncate_text(full_answer),
                            "num_citations": len(citations),
                            "confidence_label": confidence.get("label"),
                            "confidence_score": confidence.get("score"),
                            "retrieved_chunks": sanitize_chunks(retrieved_chunks),
                            "reranked_chunks": sanitize_chunks(reranked_chunks),
                        })

                        yield json.dumps({
                            "type": "done",
                            "answer": full_answer,
                            "confidence": confidence,
                        }) + "\n"

                    except Exception as exc:
                        logger.exception("Unexpected error during streaming ask")
                        safe_add_metadata(parent_span, {"error": str(exc)[:200]})
                        yield json.dumps({
                            "type": "error",
                            "message": "The server encountered an error. Please try again.",
                        }) + "\n"
                        yield json.dumps({
                            "type": "done",
                            "answer": "",
                            "confidence": {
                                "label": "Low",
                                "score": 0.0,
                                "retrieval_score": None,
                                "rerank_score": None,
                                "judge_score": None,
                                "judge_available": False,
                                "judge_reason": "Server error during answer generation.",
                                "judge_error": None,
                            },
                        }) + "\n"
            finally:
                if timings:
                    timings["total"] = total_timer.elapsed_ms
                    record_request(timings)

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


@app.post("/generate_flashcards")
@limiter.limit(settings.rate_limit_ask)
async def generate_flashcards_endpoint(request: Request, body: FlashcardRequest):
    """
    Generate flashcards from a selected document (and optional tag).

    Pulls candidate chunks from Qdrant, shuffles for diversity, generates one
    flashcard per chunk via a single LLM call, and re-attaches source metadata
    (doc_name, page, tag) to each card.

    Args:
        request: FastAPI Request, required by the slowapi rate limiter.
        body: Document name, optional tag, and requested card count.

    Returns:
        Dictionary with a "cards" list, each item containing front, back,
        doc_name, page, tag_name, and tag_color.
    """
    normalized_session_id = _normalize_session_id(body.session_id)
    with trace_span(
        name="Lucid Flashcard Request",
        run_type="chain",
        inputs={
            "doc_name": body.doc_name,
            "tag_name": body.tag_name,
            "num_cards": body.num_cards,
            "has_session_id": normalized_session_id is not None,
        },
    ) as parent_span:
        if not await asyncio.to_thread(collection_has_points, qdrant_client):
            safe_add_metadata(parent_span, {"early_exit": "no_index"})
            raise HTTPException(status_code=400, detail="No PDF indexed yet.")

        num = max(1, min(body.num_cards, 25))

        candidates = await asyncio.to_thread(
            scroll_chunks_for_filter,
            qdrant_client,
            body.doc_name,
            _normalize_filter_value(body.tag_name),
            normalized_session_id,
            512,
        )

        if not candidates:
            safe_add_metadata(parent_span, {"early_exit": "no_chunks"})
            raise HTTPException(
                status_code=404,
                detail="No chunks found for that document/tag.",
            )

        # Filter to chunks with enough substance to make a useful flashcard
        usable = [
            payload for payload in candidates
            if len((payload.get("text") or "").strip()) >= 80
        ]

        if not usable:
            safe_add_metadata(parent_span, {"early_exit": "no_substantial_chunks"})
            raise HTTPException(
                status_code=404,
                detail="No substantial passages found for that filter.",
            )

        random.shuffle(usable)
        chosen = usable[: min(num, len(usable))]

        passages = [(chunk.get("text") or "").strip() for chunk in chosen]

        generated = await asyncio.to_thread(generate_flashcards, passages)

        if not generated:
            safe_add_metadata(parent_span, {"early_exit": "llm_returned_no_cards"})
            raise HTTPException(
                status_code=502,
                detail="Flashcard generation produced no cards.",
            )

        cards: list[dict] = []
        for i, chunk in enumerate(chosen):
            if i >= len(generated):
                break
            item = generated[i]
            cards.append({
                "front": item["front"],
                "back": item["back"],
                "doc_name": chunk.get("doc_name") or body.doc_name,
                "page": chunk.get("page_start") or chunk.get("page"),
                "tag_name": chunk.get("tag_name"),
                "tag_color": chunk.get("tag_color"),
            })

        safe_add_outputs(parent_span, {
            "num_cards_requested": body.num_cards,
            "num_chunks_available": len(usable),
            "num_cards_returned": len(cards),
        })

        return {"cards": cards}
