"""
Lucid backend launcher with configurable worker count.

Multi-worker mode is the single biggest concurrency lever for handling
concurrent users. Each worker is a separate Python process with its own
copy of the embedding models, reranker, and tokenizer, so RAM scales
linearly with --workers.

Multi-worker mode is incompatible with --reload. For hot-reload dev, run
plain `uvicorn backend.app.main:app --reload` instead.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --workers 4
    python scripts/run_server.py --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from backend.app.config import settings


def default_port() -> int:
    """Return the server port, honoring Railway's PORT env var when present."""
    raw_port = os.getenv("PORT")
    if raw_port is None:
        return 8000
    try:
        return int(raw_port)
    except ValueError as exc:
        raise ValueError("PORT must be an integer.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lucid backend with N workers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port())
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help=f"Worker process count (default from settings: {settings.workers})",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(
        f"Starting Lucid: workers={args.workers} "
        f"host={args.host} port={args.port} log_level={args.log_level}"
    )

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
