#!/usr/bin/env python3
"""
Load the four demo documents directly into Qdrant.

Runs the full ingestion pipeline (parse → chunk → embed → upsert) locally
instead of through the deployed backend's /ingest_pdf endpoint. This avoids
two problems with hitting Railway:

  1. Railway's edge proxy times out long-running requests (~5 min), and
     Docling parsing a 500-page textbook on a CPU container can exceed
     that budget.
  2. Peak RAM during Docling + embedding model coexistence has caused
     OOM kills on Railway's container.

Running locally lets the parse take as long as it needs and uses the dev
machine's RAM, while still writing public chunks to the production Qdrant.

Usage:
    QDRANT_URL=...  QDRANT_API_KEY=...  python scripts/load_demo_docs.py

Optional env vars:
    PDF_DIR      default data/uploads

If you want a clean slate first, run scripts/recreate_qdrant_collection.py
before this one.
"""

import os
import sys
from pathlib import Path

# Ensure the repo root is on the import path when running the script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from backend.app.config import settings  # noqa: E402
from backend.app.ingestion.pipeline import ingest_pdf  # noqa: E402
from backend.app.retrieval.embeddings import (  # noqa: E402
    build_dense_embedding_model,
    build_sparse_embedding_model,
    embed_texts_dense,
    embed_texts_sparse,
)
from backend.app.retrieval.qdrant_store import (  # noqa: E402
    create_collection_if_not_exists,
    get_qdrant_client,
    upsert_chunks,
)

PDF_DIR = Path(os.getenv("PDF_DIR", "data/uploads"))

# Each chunk carries a 384-dim dense vector + a sparse vector + payload.
# A single upsert with 1500+ chunks exceeds Qdrant Cloud's request body
# tolerance and times out. 128 keeps each request well under 10 MB.
UPSERT_BATCH_SIZE = 128

DOCS = [
    {
        "filename": "RLbook2020 2.pdf",
        "tag_name": "Reinforcement Learning",
        "tag_color": "#6366F1",
        "source_type": "textbook",
    },
    {
        "filename": "operating_systems_three_easy_pieces 2.pdf",
        "tag_name": "Operating Systems",
        "tag_color": "#10B981",
        "source_type": "textbook",
    },
    {
        "filename": "csc263_notes 2.pdf",
        "tag_name": "CSC263",
        "tag_color": "#F59E0B",
        "source_type": "notes",
    },
    {
        "filename": "MAT102-Notes-2017-Version1 copy 2.pdf",
        "tag_name": "MAT102",
        "tag_color": "#EC4899",
        "source_type": "notes",
    },
]


def _build_public_qdrant_chunks(
    chunk_records: list[dict],
    doc_name: str,
    tag_name: str,
    tag_color: str,
    source_type: str,
    pdf_filename: str,
) -> list[dict]:
    """
    Decorate raw chunk records from ingest_pdf with the document-level
    metadata fields that upsert_chunks expects in each point payload.

    Mirrors backend.app.main._build_qdrant_chunks for public docs so the
    points written here are byte-identical to what /ingest_pdf would
    produce:

      • `id` is set to "{doc_name}::{chunk_id}" so _stable_point_id
        derives the same UUID the endpoint would (re-ingest is then a
        true overwrite instead of a duplicate).
      • `doc_name` is overwritten with the display filename (full name
        with extension), while `chunk_id` is kept in its
        endpoint-native "{stem}-{idx}" format produced by ingest_pdf.
      • `page` is set to page_start as the citation-display fallback.
    """
    enriched: list[dict] = []
    for i, record in enumerate(chunk_records):
        chunk_id = record.get("chunk_id", i)
        page = record.get("page") or record.get("page_start")
        enriched.append(
            {
                **record,
                "id": f"{doc_name}::{chunk_id}",
                "text": record["text"],
                "doc_name": doc_name,
                "page": page,
                "page_start": record.get("page_start", page),
                "page_end": record.get("page_end", page),
                "chunk_id": chunk_id,
                "tag_name": tag_name,
                "tag_color": tag_color,
                "source_type": source_type,
                "visibility": "public",
                "owner_id": None,
                "uploaded_at": None,
                "uploaded_at_epoch": None,
                "pdf_filename": pdf_filename,
            }
        )
    return enriched


def ingest_one(
    pdf_path: Path,
    tag_name: str,
    tag_color: str,
    source_type: str,
    tokenizer,
    dense_model,
    sparse_model,
    qdrant_client,
) -> bool:
    if not pdf_path.exists():
        print(f"  ✗ Missing file: {pdf_path}")
        return False

    print(f"  → {pdf_path.name}  (tag={tag_name}, source={source_type})")

    doc_name = pdf_path.name

    print("     · parsing + chunking …")
    chunk_records = list(
        ingest_pdf(
            pdf_path=pdf_path,
            tokenizer=tokenizer,
            max_tokens=settings.max_tokens_per_chunk,
            overlap_paragraphs=settings.overlap_paragraphs,
        )
    )
    if not chunk_records:
        print("     ✗ no text chunks extracted")
        return False

    # ingest_pdf() sets doc_name = pdf_path.stem and
    # chunk_id = "{stem}-{idx}". _build_public_qdrant_chunks below
    # overrides doc_name to the full display filename (matching the
    # endpoint's _build_qdrant_chunks) while preserving the
    # stem-based chunk_id so point UUIDs and citation IDs match what
    # /ingest_pdf would write for the same PDF.

    chunk_texts = [record["text"] for record in chunk_records]
    print(f"     · {len(chunk_texts)} chunks → embedding …")

    dense_embeddings = embed_texts_dense(
        texts=chunk_texts,
        embedding_model=dense_model,
        batch_size=settings.dense_embedding_batch_size,
    )
    sparse_embeddings = embed_texts_sparse(
        texts=chunk_texts,
        sparse_model=sparse_model,
    )

    qdrant_chunks = _build_public_qdrant_chunks(
        chunk_records=chunk_records,
        doc_name=doc_name,
        tag_name=tag_name,
        tag_color=tag_color,
        source_type=source_type,
        pdf_filename=doc_name,
    )

    total = len(qdrant_chunks)
    print(f"     · upserting to Qdrant in batches of {UPSERT_BATCH_SIZE} …")
    for start in range(0, total, UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, total)
        upsert_chunks(
            client=qdrant_client,
            chunks=qdrant_chunks[start:end],
            dense_embeddings=dense_embeddings[start:end],
            sparse_embeddings=sparse_embeddings[start:end],
        )
        print(f"       · {end}/{total}")
    print(f"     ✓ {doc_name} — {total} chunks")
    return True


def main() -> int:
    print(f"Qdrant:  {settings.qdrant_url}")
    print(f"PDFs:    {PDF_DIR.resolve()}\n")

    if not PDF_DIR.exists():
        print(f"✗ PDF directory not found: {PDF_DIR.resolve()}")
        return 1

    print("Loading models (first run downloads, then cached) …")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.dense_embedding_model_checkpoint
    )
    dense_model = build_dense_embedding_model(
        settings.dense_embedding_model_checkpoint
    )
    sparse_model = build_sparse_embedding_model(
        settings.sparse_embedding_model_checkpoint
    )

    print("Connecting to Qdrant …")
    qdrant_client = get_qdrant_client()
    create_collection_if_not_exists(
        client=qdrant_client,
        dense_vector_size=settings.dense_vector_size,
    )
    print()

    successes = 0
    for doc in DOCS:
        ok = ingest_one(
            pdf_path=PDF_DIR / doc["filename"],
            tag_name=doc["tag_name"],
            tag_color=doc["tag_color"],
            source_type=doc["source_type"],
            tokenizer=tokenizer,
            dense_model=dense_model,
            sparse_model=sparse_model,
            qdrant_client=qdrant_client,
        )
        if ok:
            successes += 1

    print(f"\n{successes}/{len(DOCS)} documents ingested.")
    return 0 if successes == len(DOCS) else 1


if __name__ == "__main__":
    sys.exit(main())
