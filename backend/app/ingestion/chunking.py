import re
from typing import Any, Iterator

def normalize_whitespace(text: str) -> str:
    """
    Normalize repeated whitespace into a single space and trim edges.

    Args:
        text: Raw input string.

    Returns:
        Cleaned text.
    """
    return re.sub(r"\s+", " ", text).strip()

def normalize_pdf_text(text: str) -> str:
    """
    Normalize extracted PDF text while preserving table-like markdown rows.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text with normalized line endings and spacing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Only collapse spaces on non-table lines
    lines = []
    for line in text.split("\n"):
        if line.startswith("|"):  # preserve table rows
            lines.append(line)
        else:
            lines.append(re.sub(r"[ \t]+", " ", line))
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into cleaned paragraphs.

    Args:
        text: Normalized page text.

    Returns:
        List of non-empty paragraphs with collapsed whitespace.
    """
    raw_paragraphs = re.split(r"\n\s*\n+", text)

    return [
        normalize_whitespace(paragraph)
        for paragraph in raw_paragraphs
        if normalize_whitespace(paragraph)
    ]

def count_tokens(text: str, tokenizer: Any) -> int:
    """
    Count the number of tokens in text.

    Args:
        text: Text to count.
        tokenizer: Hugging Face-compatible tokenizer.

    Returns:
        Number of tokens without special tokens.
    """
    return len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )

def chunk_pages_by_paragraphs(
    pages: Iterator[tuple[int, str]],
    tokenizer: Any,
    max_tokens: int,
    overlap_paragraphs: int = 1,
) -> Iterator[dict]:
    """
    Build paragraph-aware chunks across pages while preserving page metadata.

    Args:
        pages: Iterator of (page_number, page_text).
        tokenizer: Tokenizer used to count tokens.
        max_tokens: Maximum number of tokens per chunk.
        overlap_paragraphs: Number of paragraphs to repeat between chunks.

    Yields:
        Chunk records containing text and page-range metadata.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs must be >= 0")

    current_paragraphs: list[str] = []
    current_pages: list[int] = []
    current_token_count = 0
    chunk_index = 0

    for page_number, page_text in pages:
        paragraphs = split_into_paragraphs(page_text)

        for paragraph in paragraphs:
            paragraph_token_count = count_tokens(paragraph, tokenizer)

            if (
                current_paragraphs
                and current_token_count + paragraph_token_count > max_tokens
            ):
                yield build_chunk_record(
                    chunk_index=chunk_index,
                    paragraphs=current_paragraphs,
                    pages=current_pages,
                )

                chunk_index += 1

                if overlap_paragraphs > 0:
                    current_paragraphs = current_paragraphs[-overlap_paragraphs:]
                    current_pages = current_pages[-overlap_paragraphs:]
                else:
                    current_paragraphs = []
                    current_pages = []

                current_token_count = sum(
                    count_tokens(text, tokenizer) for text in current_paragraphs
                )

            current_paragraphs.append(paragraph)
            current_pages.append(page_number)
            current_token_count += paragraph_token_count

    if current_paragraphs:
        yield build_chunk_record(
            chunk_index=chunk_index,
            paragraphs=current_paragraphs,
            pages=current_pages,
        )


def build_chunk_record(
    chunk_index: int,
    paragraphs: list[str],
    pages: list[int],
) -> dict:
    """
    Build a JSON-serializable chunk record.

    Args:
        chunk_index: Zero-based index for the chunk.
        paragraphs: Paragraph text included in the chunk.
        pages: Page numbers represented by the chunk paragraphs.

    Returns:
        Chunk dictionary containing text, page range, page list, and index.
    """
    unique_pages = sorted(set(pages))

    return {
        "chunk_index": chunk_index,
        "text": "\n\n".join(paragraphs),
        "page_start": unique_pages[0],
        "page_end": unique_pages[-1],
        "pages": unique_pages,
    }
