import math
import os
from typing import Any
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

from backend.app.config import settings


RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def safe_float(value: Any) -> float | None:
    """
    Convert a value to float when possible. Treat NaN as None so failed
    RAGAS metric cells do not poison downstream averages or displays.
    """
    try:
        if value is None:
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def load_ragas_metrics():
    """
    Load RAGAS metrics using the common public API.
    """
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]


def build_ragas_llm_and_embeddings():
    """
    Build explicit RAGAS-compatible LLM and embeddings wrappers.

    Passing these explicitly to evaluate() prevents RAGAS from auto-selecting
    its internal OpenAIEmbeddings class (which lacks embed_query) and its
    default low-max-tokens LLM (which truncates faithfulness statement lists).
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat_model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        max_tokens=4000,
        max_retries=2,
        timeout=120,
        temperature=0.0,
    )

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )

    return (
        LangchainLLMWrapper(chat_model),
        LangchainEmbeddingsWrapper(embedding_model),
    )


def compute_ragas_for_mode(
    mode_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute RAGAS metrics for all results from one eval mode.

    Failed examples produce NaN cells in the RAGAS dataframe; we convert those
    to None via safe_float so the summary mean averages only valid values.
    """
    if not mode_results:
        return mode_results, {metric: None for metric in RAGAS_METRIC_NAMES}

    try:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

        dataset_rows = [
            {
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": result["ground_truth"],
            }
            for result in mode_results
        ]

        dataset = Dataset.from_list(dataset_rows)

        ragas_llm, ragas_embeddings = build_ragas_llm_and_embeddings()

        ragas_result = evaluate(
            dataset=dataset,
            metrics=load_ragas_metrics(),
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(
                timeout=180,
                max_retries=2,
                max_wait=60,
                max_workers=4,
            ),
            raise_exceptions=False,
            show_progress=True,
        )

        ragas_df = ragas_result.to_pandas()
        ragas_records = ragas_df.to_dict("records")

        updated_results = []

        for result, ragas_row in zip(mode_results, ragas_records):
            enriched = dict(result)

            for metric in RAGAS_METRIC_NAMES:
                enriched[metric] = safe_float(ragas_row.get(metric))

            enriched["ragas_error"] = None
            updated_results.append(enriched)

        summary: dict[str, Any] = {}

        for metric in RAGAS_METRIC_NAMES:
            values = [
                safe_float(result.get(metric))
                for result in updated_results
                if safe_float(result.get(metric)) is not None
            ]
            summary[metric] = sum(values) / len(values) if values else None

        return updated_results, summary

    except Exception as exc:
        updated_results = []

        for result in mode_results:
            enriched = dict(result)
            for metric in RAGAS_METRIC_NAMES:
                enriched[metric] = None
            enriched["ragas_error"] = str(exc)
            updated_results.append(enriched)

        summary = {metric: None for metric in RAGAS_METRIC_NAMES}
        summary["ragas_error"] = str(exc)

        return updated_results, summary