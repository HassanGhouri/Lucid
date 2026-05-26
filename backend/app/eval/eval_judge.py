import json
import re
from typing import Any
from openai import OpenAI

from backend.app.config import settings


# Module-level client mirrors the pattern in backend/app/generation/llm.py:
# one client construction shared across all judge calls in an eval run, with
# the same per-call timeout used by the production answer/judge paths. A
# 100-question, 4-mode eval makes ~400 judge calls; reusing the httpx pool
# avoids re-doing TCP+TLS each time.
_openai_client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=settings.openai_timeout_seconds,
)


def get_eval_judge_model() -> str:
    """
    Return the eval judge model name, falling back to gpt-4.1-mini.

    Set JUDGE_MODEL in .env to override (e.g. JUDGE_MODEL=gpt-4.1 for a
    stronger judge during evaluation).
    """
    return getattr(settings, "judge_model", None) or "gpt-4.1-mini"


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a model response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response.")

    return json.loads(match.group(0))


def clamp_score(value: Any) -> float | None:
    """
    Convert a score to a float in [0, 1].
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(1.0, score))


def build_eval_judge_prompt(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: list[str],
) -> str:
    """
    Build the LLM-as-judge prompt for one eval result.
    """
    context_text = "\n\n".join(
        f"[Context {i}]\n{context[:2500]}"
        for i, context in enumerate(contexts, start=1)
    )

    return f"""
You are an impartial RAG evaluation judge.

Evaluate the Lucid answer using only:
1. The question
2. The retrieved contexts
3. The ground-truth answer

Score each item from 0 to 1:
- judge_grounding_score: Is the answer supported by the retrieved contexts?
- judge_correctness_score: Is the answer correct compared with the ground truth?
- judge_usefulness_score: Is the answer useful, clear, and complete for a student?

Return ONLY valid JSON with this schema:
{{
  "judge_grounding_score": 0.0,
  "judge_correctness_score": 0.0,
  "judge_usefulness_score": 0.0,
  "judge_explanation": "brief explanation"
}}

Question:
{question}

Ground-truth answer:
{ground_truth}

Lucid answer:
{answer}

Retrieved contexts:
{context_text}
""".strip()


def judge_eval_result(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: list[str],
) -> dict[str, Any]:
    """
    Score one generated answer using an OpenAI LLM judge.

    This function fails gracefully and returns judge_error instead of raising,
    so one failed judge call does not stop the full eval run.
    """
    if not answer or "unexpected error" in answer.lower():
        return {
            "judge_grounding_score": None,
            "judge_correctness_score": None,
            "judge_usefulness_score": None,
            "judge_explanation": None,
            "judge_error": "Skipped judge because answer generation failed.",
        }

    prompt = build_eval_judge_prompt(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        contexts=contexts,
    )

    try:
        response = _openai_client.responses.create(
            model=get_eval_judge_model(),
            input=prompt,
            max_output_tokens=600,
        )

        parsed = extract_json_object(response.output_text)

        return {
            "judge_grounding_score": clamp_score(parsed.get("judge_grounding_score")),
            "judge_correctness_score": clamp_score(parsed.get("judge_correctness_score")),
            "judge_usefulness_score": clamp_score(parsed.get("judge_usefulness_score")),
            "judge_explanation": parsed.get("judge_explanation"),
            "judge_error": None,
        }

    except Exception as exc:
        return {
            "judge_grounding_score": None,
            "judge_correctness_score": None,
            "judge_usefulness_score": None,
            "judge_explanation": None,
            "judge_error": str(exc),
        }