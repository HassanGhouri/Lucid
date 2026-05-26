from pathlib import Path
from typing import Any, Iterator

from backend.app.ingestion.loaders import extract_pdf_pages
from backend.app.ingestion.chunking import chunk_pages_by_paragraphs

def ingest_pdf(
    pdf_path: str | Path,
    tokenizer: Any,
    max_tokens: int,
    overlap_paragraphs: int = 1,
) -> Iterator[dict]:
    """
    Extract, chunk, and annotate a PDF for ingestion.

    Args:
        pdf_path: Path to the uploaded PDF.
        tokenizer: Tokenizer used for token-aware chunking.
        max_tokens: Maximum tokens per chunk.
        overlap_paragraphs: Number of paragraphs to overlap between chunks.

    Yields:
        Chunk records containing text, page metadata, document name, and chunk ID.
    """

    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem

    pages = extract_pdf_pages(pdf_path)

    for chunk in chunk_pages_by_paragraphs(
        pages=pages,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_paragraphs=overlap_paragraphs,
    ):
        yield {
            "doc_name": doc_name,
            "chunk_id": f"{doc_name}-{chunk['chunk_index']}",
            **chunk,
        }
