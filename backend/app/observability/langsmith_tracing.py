"""
LangSmith tracing helpers for Lucid.

This module is a thin, fail-safe wrapper around the langsmith SDK. Every
public helper is designed to be a no-op when tracing is disabled and to
swallow any tracing-side exception so observability never breaks production
behavior.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, Iterator

from backend.app.config import settings

logger = logging.getLogger(__name__)

# Try to import langsmith. If missing, all helpers below fall back to no-ops.
try:
    from langsmith import traceable as _ls_traceable
    try:
        from langsmith import trace as _ls_trace
    except ImportError:
        from langsmith.run_helpers import trace as _ls_trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    _ls_traceable = None
    _ls_trace = None


def get_trace_url(span: Any) -> str | None:
    """
    Best-effort: return the LangSmith viewer URL for a span / run tree.

    Returns None when tracing is disabled, the span is a no-op stand-in,
    the SDK doesn't expose get_url(), or any error occurs. Never raises.
    """
    if not is_tracing_enabled():
        return None
    if span is None or isinstance(span, _NullSpan):
        return None
    try:
        get_url = getattr(span, "get_url", None)
        if not callable(get_url):
            return None
        url = get_url()
        if isinstance(url, str) and url.startswith("http"):
            return url
    except Exception as exc:
        logger.debug("get_trace_url failed: %s", exc)
    return None


def is_tracing_enabled() -> bool:
    """
    Return True only if all conditions for active LangSmith tracing are met:
    package installed, flag set in settings, and API key present.
    """
    if not LANGSMITH_AVAILABLE:
        return False
    if not getattr(settings, "langsmith_tracing", False):
        return False
    if not getattr(settings, "langsmith_api_key", None):
        return False
    return True


def configure_langsmith_from_settings() -> None:
    """
    Apply LangSmith config to process env vars.

    LangSmith's SDK reads LANGSMITH_TRACING, LANGSMITH_API_KEY, etc., from
    the environment. Setting them here from pydantic settings keeps config
    centralized in .env / settings. Uses setdefault so any pre-existing env
    var (e.g. set by docker-compose) wins.
    """
    if not LANGSMITH_AVAILABLE:
        return
    try:
        if getattr(settings, "langsmith_tracing", False):
            os.environ.setdefault("LANGSMITH_TRACING", "true")
        if getattr(settings, "langsmith_api_key", None):
            os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        if getattr(settings, "langsmith_project", None):
            os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        if getattr(settings, "langsmith_endpoint", None):
            os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    except Exception as exc:
        logger.warning("LangSmith env setup failed: %s", exc)


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

def truncate_text(text: Any, max_chars: int | None = None) -> str:
    """
    Truncate a string for safe inclusion in traces.

    Long PDF chunks, long prompts, or raw LLM responses can be very large and
    cost money/bandwidth to ship to LangSmith. We cap the size and append a
    marker so users can tell when content was trimmed.
    """
    if text is None:
        return ""
    try:
        text_str = str(text)
    except Exception:
        return ""
    cap = max_chars or getattr(settings, "trace_max_text_chars", 1200)
    if len(text_str) <= cap:
        return text_str
    return text_str[:cap] + f"... [truncated, {len(text_str) - cap} chars]"


def sanitize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce a retrieval chunk dict to trace-relevant fields with truncated text.
    Drops embeddings, raw payloads, and other heavy/sensitive fields.
    """
    if not isinstance(chunk, dict):
        return {}
    keys = (
        "doc_name", "page", "page_start", "page_end", "chunk_id",
        "tag_name", "source_type",
        "retrieval_score", "dense_score", "sparse_score", "rrf_score",
        "rerank_score",
        "retrieval_score_normalized", "rerank_score_normalized",
    )
    out: dict[str, Any] = {key: chunk.get(key) for key in keys if key in chunk}
    if "text" in chunk:
        out["text"] = truncate_text(chunk["text"])
    return out


def sanitize_chunks(
    chunks: list[dict[str, Any]] | None,
    max_chunks: int = 10,
) -> list[dict[str, Any]]:
    """Sanitize and cap a list of retrieval chunks for tracing."""
    if not chunks:
        return []
    return [sanitize_chunk(chunk) for chunk in chunks[:max_chunks]]


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def traceable(*args, **kwargs):
    """
    Drop-in replacement for langsmith.traceable.

    Returns the real decorator when LangSmith is installed and importable,
    a no-op pass-through decorator otherwise. The decorated function runs
    normally regardless of tracing state.

    Usage:
        @traceable(name="Dense Query Embedding", run_type="embedding")
        def embed_query_dense(...): ...
    """
    if LANGSMITH_AVAILABLE and _ls_traceable is not None:
        try:
            return _ls_traceable(*args, **kwargs)
        except Exception as exc:
            logger.debug("langsmith.traceable construction failed: %s", exc)

    def _noop_decorator(func: Callable) -> Callable:
        return func
    if args and callable(args[0]):
        return args[0]
    return _noop_decorator


class _NullSpan:
    """No-op stand-in for a LangSmith run tree, mirroring its shape."""
    def add_inputs(self, *_a, **_k) -> None: ...
    def add_outputs(self, *_a, **_k) -> None: ...
    def add_metadata(self, *_a, **_k) -> None: ...
    def end(self, *_a, **_k) -> None: ...


@contextlib.contextmanager
def trace_span(
    name: str,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """
    Context manager for ad-hoc trace spans.

    Yields a span object with add_inputs / add_outputs / add_metadata methods.
    When tracing is disabled or unavailable, yields a no-op object so call
    sites look identical either way. All errors are swallowed.
    """
    if not is_tracing_enabled() or _ls_trace is None:
        yield _NullSpan()
        return

    try:
        with _ls_trace(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata or {},
        ) as run_tree:
            yield run_tree
    except Exception as exc:
        logger.warning("LangSmith span %r failed (continuing): %s", name, exc)
        yield _NullSpan()


def safe_add_outputs(span: Any, outputs: dict[str, Any]) -> None:
    """Best-effort: attach outputs to a span without raising."""
    try:
        if span is not None and hasattr(span, "add_outputs"):
            span.add_outputs(outputs)
    except Exception as exc:
        logger.debug("safe_add_outputs failed: %s", exc)


def safe_add_metadata(span: Any, metadata: dict[str, Any]) -> None:
    """Best-effort: attach metadata to a span without raising."""
    try:
        if span is not None and hasattr(span, "add_metadata"):
            span.add_metadata(metadata)
    except Exception as exc:
        logger.debug("safe_add_metadata failed: %s", exc)