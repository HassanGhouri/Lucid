from pathlib import Path
from typing import Iterator, Tuple
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling_core.types.doc.document import TableItem, TextItem
import logging

from backend.app.ingestion.chunking import normalize_pdf_text

logger = logging.getLogger(__name__)


def _build_pdf_converter() -> DocumentConverter:
    """
    Build a Docling PDF converter pinned to CPU.

    Docling's layout model can select MPS on Apple Silicon, but the current
    transformer path uses float64 positional embeddings that MPS rejects.
    CPU keeps ingestion portable and matches the backend model-device choice.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _get_page_number(element) -> int | None:
    """
    Safely extract page number from a Docling element.
    """
    try:
        return element.prov[0].page_no
    except (AttributeError, IndexError, TypeError):
        return None


def _get_element_text(element, doc) -> str:
    """
    Extract text from a Docling element, handling tables specially.
    """
    try:
        if isinstance(element, TableItem):
            return element.export_to_markdown(doc)
        elif isinstance(element, TextItem):
            return element.text or ""
    except Exception as e:
        logger.warning("Failed to extract Docling element text: %s", e)
    return ""


def extract_pdf_pages(pdf_path: Path) -> Iterator[Tuple[int, str]]:
    """
    Extract text from a PDF page by page using Docling.

    Docling handles complex layouts, tables, multi-column text, and
    scanned pages (via built-in OCR) — replacing both PyPDF2 and
    the manual pytesseract fallback.

    Args:
        pdf_path: Path to the PDF file.

    Yields:
        A tuple containing:
            - page number, starting at 1
            - normalized extracted text for that page
    """
    converter = _build_pdf_converter()
    result = converter.convert(str(pdf_path))

    # Group Docling's elements by page number
    pages: dict[int, list[str]] = {}

    for element, _level in result.document.iterate_items():
        page_number = _get_page_number(element)

        if page_number is None:
            continue

        text = _get_element_text(element, result.document)

        if text:
            pages.setdefault(page_number, []).append(text)

    for page_number in sorted(pages.keys()):
        page_text = "\n\n".join(pages[page_number])
        yield page_number, normalize_pdf_text(page_text)
