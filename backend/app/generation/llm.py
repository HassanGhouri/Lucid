import json
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, BadRequestError
from typing import Generator

from backend.app.observability.langsmith_tracing import traceable
from backend.app.config import settings
from backend.app.generation.prompts import (
    build_grounding_judge_prompt,
    build_query_rewrite_prompt,
    build_rag_prompt,
)

MAX_QUESTION_CHARS = 2000


def _extract_json_object(text: str) -> dict:
    """
    Parse a JSON object from a model response.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON object.

    Raises:
        json.JSONDecodeError: If no valid JSON object can be parsed.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


# Module-level OpenAI client, constructed once at import. Reusing a single
# client across requests reuses its underlying httpx connection pool, so
# concurrent /ask_question calls don't pay TCP+TLS handshake cost per call.
# The per-call timeout caps individual API requests so a slow OpenAI
# response cannot indefinitely pin a worker thread; APITimeoutError is a
# subclass of APIError, so existing error handling in generate_answer /
# rewrite_query / judge_grounding / stream_answer / generate_flashcards
# catches timeouts without further changes.
_openai_client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=settings.openai_timeout_seconds,
)


@traceable(name="Answer Generation", run_type="llm")
def generate_answer(
    question: str,
    chunks: list[dict],
    max_output_tokens: int,
    history: list[dict] | None = None,
) -> str:
    """
    Generate a grounded answer from retrieved document chunks.

    Args:
        question: User question.
        chunks: Retrieved and reranked chunks.
        max_output_tokens: Maximum tokens to request from the OpenAI model.
        history: Optional bounded chat history used only for conversational
            context.

    Returns:
        Answer text from the configured OpenAI model, or a safe fallback message.
    """

    question = question.strip()

    if len(question) > MAX_QUESTION_CHARS:
        return "Please ask a shorter question."

    if not question:
        return "Please enter a question."

    if not chunks:
        return "I couldn't find relevant information in the provided document."

    prompt = build_rag_prompt(question=question, chunks=chunks, history=history)

    try:
        response = _openai_client.responses.create(
            model=settings.openai_model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        return response.output_text

    except AuthenticationError:
        return "The AI service is not configured correctly."

    except RateLimitError:
        return "The app is temporarily rate limited or out of quota. Please try again later."

    except BadRequestError:
        return "The request could not be processed. Try asking a shorter or clearer question."

    except APIError:
        return "There was an issue contacting the AI service. Please try again."

    except Exception:
        return "An unexpected error occurred. Please try again."


@traceable(name="Query Rewrite", run_type="llm")
def rewrite_query(question: str, history: list[dict] | None = None) -> dict:
    """
    Rewrite a user question into a retrieval-focused query.

    Args:
        question: Original user question.
        history: Optional bounded chat history used to resolve follow-up
            references.

    Returns:
        Dictionary containing the original query, rewritten query, and whether
        rewriting succeeded. If rewriting fails, the rewritten query falls back
        to the original question.
    """
    original_query = question.strip()

    if not original_query:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_used": False,
            "rewrite_error": "Empty question.",
        }

    if len(original_query) > MAX_QUESTION_CHARS:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_used": False,
            "rewrite_error": "Question too long.",
        }

    prompt = build_query_rewrite_prompt(original_query, history=history)

    try:
        response = _openai_client.responses.create(
            model=settings.query_rewrite_model,
            input=prompt,
            max_output_tokens=80,
        )

        rewritten_query = response.output_text.strip().strip('"').strip("'")

        if not rewritten_query:
            rewritten_query = original_query

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_used": rewritten_query != original_query,
            "rewrite_error": None,
        }

    except (AuthenticationError, RateLimitError, BadRequestError, APIError) as e:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_used": False,
            "rewrite_error": type(e).__name__,
        }

    except Exception as e:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_used": False,
            "rewrite_error": type(e).__name__,
        }


@traceable(name="Grounding Judge", run_type="llm")
def judge_grounding(
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Judge whether a generated answer is grounded in retrieved chunks.

    Args:
        question: Original user question.
        answer: Generated answer.
        chunks: Retrieved and reranked evidence chunks.

    Returns:
        Dictionary with judge availability, score, label, and reason.
        If judging fails, available is False and score is None.
    """
    if not answer.strip() or not chunks:
        return {
            "available": False,
            "score": None,
            "label": None,
            "reason": "Missing answer or chunks.",
            "error": None,
        }

    prompt = build_grounding_judge_prompt(
        question=question,
        answer=answer,
        chunks=chunks,
    )

    try:
        response = _openai_client.responses.create(
            model=settings.judge_model,
            input=prompt,
            max_output_tokens=180,
        )

        raw_text = response.output_text.strip()

        parsed = _extract_json_object(raw_text)

        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        label = parsed.get("label")
        if label not in {"high", "medium", "low"}:
            if score >= 0.75:
                label = "high"
            elif score >= 0.45:
                label = "medium"
            else:
                label = "low"

        return {
            "available": True,
            "score": score,
            "label": label,
            "reason": str(parsed.get("reason", "")).strip(),
            "error": None,
        }

    except (json.JSONDecodeError, ValueError) as e:
        return {
            "available": False,
            "score": None,
            "label": None,
            "reason": "Judge returned invalid JSON.",
            "error": type(e).__name__,
        }

    except (AuthenticationError, RateLimitError, BadRequestError, APIError) as e:
        return {
            "available": False,
            "score": None,
            "label": None,
            "reason": "Judge call failed.",
            "error": type(e).__name__,
        }

    except Exception as e:
        return {
            "available": False,
            "score": None,
            "label": None,
            "reason": "Unexpected judge failure.",
            "error": type(e).__name__,
        }


@traceable(name="Answer Generation (stream)", run_type="llm")
def stream_answer(
    question: str,
    chunks: list[dict],
    max_output_tokens: int,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Stream a grounded answer as text deltas.

    Mirrors generate_answer's guards and error handling. On any guard failure
    or OpenAI error, yields a single fallback string and returns so the caller
    sees a complete (if brief) response either way.

    Args:
        question: User question.
        chunks: Retrieved and reranked chunks.
        max_output_tokens: Maximum tokens to request from the OpenAI model.
        history: Optional bounded chat history used only for conversational
            context.

    Yields:
        Text deltas from the model, or one fallback message on failure.
    """
    question = question.strip()

    if len(question) > MAX_QUESTION_CHARS:
        yield "Please ask a shorter question."
        return

    if not question:
        yield "Please enter a question."
        return

    if not chunks:
        yield "I couldn't find relevant information in the provided document."
        return

    prompt = build_rag_prompt(question=question, chunks=chunks, history=history)

    try:
        stream = _openai_client.responses.create(
            model=settings.openai_model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            stream=True,
        )

        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield delta

    except AuthenticationError:
        yield "The AI service is not configured correctly."

    except RateLimitError:
        yield "The app is temporarily rate limited or out of quota. Please try again later."

    except BadRequestError:
        yield "The request could not be processed. Try asking a shorter or clearer question."

    except APIError:
        yield "There was an issue contacting the AI service. Please try again."

    except Exception:
        yield "An unexpected error occurred. Please try again."


FLASHCARD_SYSTEM_PROMPT = (
    "You are a study-aid generator. For each numbered passage, produce ONE flashcard:\n"
    "- 'front': a clear, self-contained question testing a key concept in the passage. "
    "Do NOT reference 'the passage', 'the text', 'this section', or similar; the student "
    "will not see the passage on the front.\n"
    "- 'back': a concise 1-2 sentence answer grounded ONLY in that passage.\n"
    "Return strict JSON of the form: "
    "{\"cards\": [{\"front\": \"...\", \"back\": \"...\"}, ...]} "
    "with exactly one entry per numbered passage, in the same order."
)


@traceable(name="Flashcard Generation", run_type="llm")
def generate_flashcards(passages: list[str]) -> list[dict]:
    """
    Generate one flashcard per input passage via a single OpenAI call.

    Args:
        passages: Cleaned passage texts in the order cards should be produced.

    Returns:
        A list of {"front": str, "back": str} dicts aligned positionally with
        `passages`. Entries with missing front or back are filtered out, so the
        returned list may be shorter than `passages` on a partial failure.
    """
    if not passages:
        return []

    passages_block = "\n\n".join(
        f"[{i + 1}]\n{(text or '').strip()[:1200]}"
        for i, text in enumerate(passages)
    )
    user_msg = (
        f"Generate exactly {len(passages)} flashcards, one per numbered passage:\n\n"
        f"{passages_block}"
    )

    try:
        resp = _openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": FLASHCARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        items = parsed.get("cards", []) or []
    except Exception:
        return []

    cleaned: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        front = (item.get("front") or "").strip()
        back = (item.get("back") or "").strip()
        if front and back:
            cleaned.append({"front": front, "back": back})

    return cleaned
