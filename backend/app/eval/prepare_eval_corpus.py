from pathlib import Path
from typing import Any
import time
from transformers import AutoTokenizer

from backend.app.config import settings
from backend.app.ingestion.pipeline import ingest_pdf
from backend.app.retrieval.embeddings import (
    build_dense_embedding_model,
    build_sparse_embedding_model,
    embed_texts_dense,
    embed_texts_sparse,
)
from backend.app.retrieval.qdrant_store import (
    create_collection_if_not_exists,
    delete_document_chunks,
    get_qdrant_client,
    upsert_chunks,
)


EVAL_PDF_DIR = Path("data/eval_pdfs")

EVAL_DOCUMENTS = [
    {
        "filename": "RLbook2020.pdf",
        "document": "Reinforcement Learning: An Introduction",
        "tag_name": "Eval: Reinforcement Learning",
        "tag_color": "#2563EB",
        "source_type": "eval_textbook",
    },
    {
        "filename": "operating_systems_three_easy_pieces.pdf",
        "document": "Operating Systems: Three Easy Pieces",
        "tag_name": "Eval: Operating Systems",
        "tag_color": "#16A34A",
        "source_type": "eval_textbook",
    },
    {
        "filename": "csc263_notes.pdf",
        "document": "CSC263 Data Structures and Analysis Notes",
        "tag_name": "Eval: CSC263",
        "tag_color": "#9333EA",
        "source_type": "eval_notes",
    },
    {
        "filename": "MAT102-Notes-2017-Version1.pdf",
        "document": "MAT102 Introduction to Mathematical Proofs Notes",
        "tag_name": "Eval: MAT102",
        "tag_color": "#EA580C",
        "source_type": "eval_notes",
    },
]

def log_step(message: str) -> None:
    """
    Print a clear progress message for long-running eval corpus setup.
    """
    print(f"[eval-corpus] {message}", flush=True)


def build_eval_qdrant_chunks(
    chunk_records: list[dict[str, Any]],
    doc_meta: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Convert ingestion chunks into Qdrant-ready chunks for the eval corpus.

    Args:
        chunk_records: Chunks produced by the existing PDF ingestion pipeline.
        doc_meta: Metadata for the eval document.

    Returns:
        Chunks containing text, document metadata, page metadata, tag metadata,
        and stable IDs for Qdrant upsert.
    """
    qdrant_chunks = []

    for i, record in enumerate(chunk_records):
        chunk_id = record.get("chunk_id", f"{Path(doc_meta['filename']).stem}-{i}")
        page = record.get("page") or record.get("page_start")

        qdrant_chunks.append(
            {
                **record,
                "id": f"{doc_meta['filename']}::{chunk_id}",
                "text": record["text"],
                "document": doc_meta["document"],
                "doc_name": doc_meta["filename"],
                "topic": doc_meta["tag_name"],
                "page": page,
                "page_start": record.get("page_start", page),
                "page_end": record.get("page_end", page),
                "chunk_id": chunk_id,
                "tag_name": doc_meta["tag_name"],
                "tag_color": doc_meta["tag_color"],
                "source_type": doc_meta["source_type"],
            }
        )

    return qdrant_chunks


def ingest_eval_document(
    doc_meta: dict[str, str],
    tokenizer,
    dense_embedding_model,
    sparse_embedding_model,
    qdrant_client,
) -> int:
    """
    Ingest one eval PDF into Qdrant with visible progress output.
    """
    started_at = time.perf_counter()
    pdf_path = EVAL_PDF_DIR / doc_meta["filename"]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing eval PDF: {pdf_path}. "
            f"Place the file in data/eval_pdfs/ with this exact filename."
        )

    log_step(f"Starting document: {doc_meta['document']}")
    log_step(f"PDF path: {pdf_path}")

    try:
        log_step("Deleting old chunks for this document...")
        delete_document_chunks(qdrant_client, doc_meta["filename"])
        log_step("Old chunks deleted.")
    except Exception as exc:
        log_step(f"Could not delete old chunks, continuing anyway: {exc}")

    log_step("Extracting text with Docling and creating chunks...")
    chunk_records = list(
        ingest_pdf(
            pdf_path=pdf_path,
            tokenizer=tokenizer,
            max_tokens=settings.max_tokens_per_chunk,
            overlap_paragraphs=settings.overlap_paragraphs,
        )
    )

    if not chunk_records:
        raise RuntimeError(f"No chunks extracted from {pdf_path}")

    log_step(f"Created {len(chunk_records)} chunks.")

    qdrant_chunks = build_eval_qdrant_chunks(
        chunk_records=chunk_records,
        doc_meta=doc_meta,
    )

    chunk_texts = [chunk["text"] for chunk in qdrant_chunks]

    log_step("Creating dense embeddings...")
    dense_embeddings = embed_texts_dense(
        texts=chunk_texts,
        embedding_model=dense_embedding_model,
        batch_size=settings.dense_embedding_batch_size,
    )
    log_step(f"Dense embeddings done. Shape: {dense_embeddings.shape}")

    log_step("Creating sparse BM25 embeddings...")
    sparse_embeddings = embed_texts_sparse(
        texts=chunk_texts,
        sparse_model=sparse_embedding_model,
    )
    log_step(f"Sparse embeddings done. Count: {len(sparse_embeddings)}")

    log_step("Upserting chunks into Qdrant...")
    upsert_chunks(
        client=qdrant_client,
        chunks=qdrant_chunks,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    elapsed = time.perf_counter() - started_at
    log_step(
        f"Finished {doc_meta['document']} "
        f"({len(qdrant_chunks)} chunks, {elapsed:.1f}s)."
    )

    return len(qdrant_chunks)


def main() -> None:
    """
    Prepare the Lucid eval/demo corpus by ingesting four course documents
    into Qdrant.
    """
    started_at = time.perf_counter()

    EVAL_PDF_DIR.mkdir(parents=True, exist_ok=True)

    log_step("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.dense_embedding_model_checkpoint,
        use_fast=True,
    )

    log_step("Loading dense embedding model...")
    dense_embedding_model = build_dense_embedding_model(
        settings.dense_embedding_model_checkpoint
    )

    log_step("Loading sparse embedding model...")
    sparse_embedding_model = build_sparse_embedding_model(
        settings.sparse_embedding_model_checkpoint
    )

    log_step("Connecting to Qdrant...")
    qdrant_client = get_qdrant_client()
    create_collection_if_not_exists(qdrant_client, settings.dense_vector_size)

    total_chunks = 0

    for index, doc_meta in enumerate(EVAL_DOCUMENTS, start=1):
        log_step(f"Document {index}/{len(EVAL_DOCUMENTS)}")
        total_chunks += ingest_eval_document(
            doc_meta=doc_meta,
            tokenizer=tokenizer,
            dense_embedding_model=dense_embedding_model,
            sparse_embedding_model=sparse_embedding_model,
            qdrant_client=qdrant_client,
        )

    elapsed = time.perf_counter() - started_at

    log_step("Eval corpus ready.")
    log_step(f"Total chunks upserted: {total_chunks}")
    log_step(f"Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()