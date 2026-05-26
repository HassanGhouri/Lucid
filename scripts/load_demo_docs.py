#!/usr/bin/env python3
"""
Load the four demo documents into Qdrant via the /ingest_pdf endpoint.

Usage:
    python scripts/load_demo_docs.py

Optional env vars:
    BACKEND_URL  default http://localhost:8000
    PDF_DIR      default data/uploads

If you want a clean slate first, run scripts/recreate_qdrant_collection.py
before this one.
"""

import os
import sys
import requests
from pathlib import Path

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PDF_DIR = Path(os.getenv("PDF_DIR", "data/uploads"))

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


def ingest(pdf_path: Path, tag_name: str, tag_color: str, source_type: str) -> bool:
    if not pdf_path.exists():
        print(f"  ✗ Missing file: {pdf_path}")
        return False

    print(f"  → {pdf_path.name}  (tag={tag_name}, source={source_type})")

    with pdf_path.open("rb") as fh:
        files = {"file": (pdf_path.name, fh, "application/pdf")}
        data = {
            "tag_name": tag_name,
            "tag_color": tag_color,
            "source_type": source_type,
        }
        try:
            r = requests.post(
                f"{BACKEND_URL}/ingest_pdf",
                files=files,
                data=data,
                timeout=600,
            )
        except requests.RequestException as exc:
            print(f"     ✗ request failed: {exc}")
            return False

    if r.ok:
        payload = r.json()
        chunks = payload.get("chunks", "?")
        doc_name = payload.get("doc_name", pdf_path.name)
        print(f"     ✓ {doc_name} — {chunks} chunks")
        return True

    print(f"     ✗ {r.status_code}: {r.text[:200]}")
    return False


def main() -> int:
    print(f"Backend: {BACKEND_URL}")
    print(f"PDFs:    {PDF_DIR.resolve()}\n")

    if not PDF_DIR.exists():
        print(f"✗ PDF directory not found: {PDF_DIR.resolve()}")
        return 1

    try:
        requests.get(f"{BACKEND_URL}/filter_options", timeout=10).raise_for_status()
    except Exception as exc:
        print(f"✗ Backend not reachable: {exc}")
        return 1

    successes = 0
    for doc in DOCS:
        if ingest(
            pdf_path=PDF_DIR / doc["filename"],
            tag_name=doc["tag_name"],
            tag_color=doc["tag_color"],
            source_type=doc["source_type"],
        ):
            successes += 1

    print(f"\n{successes}/{len(DOCS)} documents ingested.")
    return 0 if successes == len(DOCS) else 1


if __name__ == "__main__":
    sys.exit(main())