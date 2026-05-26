import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
import time
import math

from backend.app.eval.eval_judge import judge_eval_result
from backend.app.eval.eval_pipeline import (
    MODE_LABELS,
    EvalMode,
    build_eval_runtime,
    run_eval_example,
)
from backend.app.eval.ragas_eval import RAGAS_METRIC_NAMES, compute_ragas_for_mode


EVAL_DIR = Path("backend/app/eval")
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# Reranker-only eval: dense_only and hybrid don't touch the CrossEncoder,
# so they'd produce identical numbers to the previous electra-base run.
# Restore the full list after deciding on the swap.
EVAL_MODES: list[EvalMode] = [
    "hybrid_plus_rerank",
    "hybrid_plus_rerank_plus_rewrite",
]

JUDGE_METRIC_NAMES = [
    "judge_grounding_score",
    "judge_correctness_score",
    "judge_usefulness_score",
]

SUMMARY_COLUMNS = [
    "mode",
    "mode_label",
    "num_examples",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "judge_grounding_score",
    "judge_correctness_score",
    "judge_usefulness_score",
]


def log_eval(message: str) -> None:
    """
    Print a clear progress message during evaluation.
    """
    print(f"[eval] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    """
    Format seconds as a readable duration.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def load_eval_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Load eval examples from eval_dataset.json.

    Args:
        limit: Optional number of examples to keep for quick runs.

    Returns:
        List of eval examples.
    """
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        examples = json.load(file)

    if limit is not None:
        return examples[:limit]

    return examples


def safe_mean(values: list[Any]) -> float | None:
    """
    Compute mean over numeric values only. Skips None and NaN so failed
    judge or RAGAS cells do not contaminate per-mode averages.
    """
    numeric_values = []

    for value in values:
        try:
            if value is None:
                continue
            numeric = float(value)
            if math.isnan(numeric):
                continue
            numeric_values.append(numeric)
        except (TypeError, ValueError):
            continue

    if not numeric_values:
        return None

    return mean(numeric_values)


def add_judge_scores(result: dict[str, Any]) -> dict[str, Any]:
    """
    Add LLM-as-judge scores to one eval result.
    """
    judge_scores = judge_eval_result(
        question=result["question"],
        answer=result["answer"],
        ground_truth=result["ground_truth"],
        contexts=result["contexts"],
    )

    return {
        **result,
        **judge_scores,
    }


def aggregate_mode_summary(
    mode: EvalMode,
    mode_results: list[dict[str, Any]],
    ragas_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one summary row for a mode.
    """
    summary = {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "num_examples": len(mode_results),
    }

    for metric in RAGAS_METRIC_NAMES:
        summary[metric] = ragas_summary.get(metric)

    for metric in JUDGE_METRIC_NAMES:
        summary[metric] = safe_mean([result.get(metric) for result in mode_results])

    return summary


def format_metric(value: Any) -> str:
    """
    Format a metric for console and Markdown output.

    Returns "N/A" for None and for NaN values (RAGAS produces NaN when an
    individual example's metric computation fails). Both should display as
    a clean "N/A" so failed metrics do not look like real zeros or "nan".
    """
    try:
        if value is None:
            return "N/A"
        numeric = float(value)
        if math.isnan(numeric):
            return "N/A"
        return f"{numeric:.3f}"
    except (TypeError, ValueError):
        return "N/A"


def print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    """
    Print a clean comparison table to the terminal.
    """
    headers = [
        "Mode",
        "Faith",
        "Rel",
        "CtxPrec",
        "CtxRec",
        "JudgeGround",
        "JudgeCorrect",
        "JudgeUseful",
    ]

    rows = []

    for row in summary_rows:
        rows.append(
            [
                row["mode_label"],
                format_metric(row.get("faithfulness")),
                format_metric(row.get("answer_relevancy")),
                format_metric(row.get("context_precision")),
                format_metric(row.get("context_recall")),
                format_metric(row.get("judge_grounding_score")),
                format_metric(row.get("judge_correctness_score")),
                format_metric(row.get("judge_usefulness_score")),
            ]
        )

    widths = [
        max(len(str(item)) for item in [header] + [row[i] for row in rows])
        for i, header in enumerate(headers)
    ]

    print("\nEvaluation Summary")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for row in rows:
        print(" | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row)))

    print("-" * (sum(widths) + 3 * (len(widths) - 1)))


def write_summary_csv(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save summary metrics as CSV.
    """
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def build_readme_table(summary_rows: list[dict[str, Any]]) -> str:
    """
    Build a README.md-ready Markdown results table.
    """
    lines = [
        "| Mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Judge Grounding | Judge Correctness | Judge Usefulness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            f"{row['mode_label']} | "
            f"{format_metric(row.get('faithfulness'))} | "
            f"{format_metric(row.get('answer_relevancy'))} | "
            f"{format_metric(row.get('context_precision'))} | "
            f"{format_metric(row.get('context_recall'))} | "
            f"{format_metric(row.get('judge_grounding_score'))} | "
            f"{format_metric(row.get('judge_correctness_score'))} | "
            f"{format_metric(row.get('judge_usefulness_score'))} |"
        )

    return "\n".join(lines) + "\n"


def save_outputs(
    all_results: dict[str, list[dict[str, Any]]],
    summary_rows: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> None:
    """
    Save detailed JSON, summary CSV, and README.md-ready Markdown table.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detailed_output = {
        "metadata": run_metadata,
        "summary": summary_rows,
        "results_by_mode": all_results,
    }

    results_json_path = RESULTS_DIR / "eval_results.json"
    summary_csv_path = RESULTS_DIR / "eval_summary.csv"
    readme_table_path = RESULTS_DIR / "readme_table.md"

    with results_json_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_output, file, indent=2, ensure_ascii=False)

    write_summary_csv(summary_rows, summary_csv_path)

    readme_table_path.write_text(
        build_readme_table(summary_rows),
        encoding="utf-8",
    )

    print(f"\nSaved detailed results: {results_json_path}")
    print(f"Saved summary CSV:      {summary_csv_path}")
    print(f"Saved README.md table:     {readme_table_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run Lucid Phase 2 evaluation.")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N examples for a quick test.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full 100-example evaluation.",
    )

    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS scoring and only run generation plus LLM judge.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run Lucid Phase 2 eval across all four retrieval/generation modes
    with visible progress output.
    """
    run_started_at = time.perf_counter()
    args = parse_args()

    if args.full:
        limit = None
    else:
        limit = args.limit if args.limit is not None else 5

    examples = load_eval_dataset(limit=limit)

    total_mode_examples = len(EVAL_MODES) * len(examples)

    log_eval(f"Loaded {len(examples)} eval examples")
    log_eval(f"Modes to run: {len(EVAL_MODES)}")
    log_eval(f"Total mode/example runs: {total_mode_examples}")
    log_eval("Loading eval runtime models...")

    runtime = build_eval_runtime()

    log_eval("Runtime loaded.")

    all_results: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    completed_mode_examples = 0

    for mode_index, mode in enumerate(EVAL_MODES, start=1):
        mode_started_at = time.perf_counter()

        log_eval(f"--- Mode {mode_index}/{len(EVAL_MODES)}: {MODE_LABELS[mode]} ---")

        mode_results = []

        for example_index, example in enumerate(examples, start=1):
            example_started_at = time.perf_counter()

            result = run_eval_example(example=example, mode=mode, runtime=runtime)
            result = add_judge_scores(result)

            mode_results.append(result)
            completed_mode_examples += 1

            example_elapsed = time.perf_counter() - example_started_at
            n_ctx = len(result.get("contexts", []))
            judge_summary = (
                f"g={format_metric(result.get('judge_grounding_score'))} "
                f"c={format_metric(result.get('judge_correctness_score'))} "
                f"u={format_metric(result.get('judge_usefulness_score'))}"
            )

            # One line per example: id, ctx count, judge scores, time. Flag any
            # anomaly inline so debugging needs no scrollback.
            warn = ""
            if n_ctx == 0:
                warn = " [WARN no contexts]"
            elif result.get("answer", "").startswith(("An unexpected error", "I couldn't find")):
                warn = " [WARN gen fallback]"

            log_eval(
                f"  {example_index:>2}/{len(examples)} {example['id']:<10} "
                f"ctx={n_ctx} {judge_summary} "
                f"{format_seconds(example_elapsed)}{warn}"
            )

        if args.skip_ragas:
            log_eval(f"Skipping RAGAS for {MODE_LABELS[mode]}.")
            ragas_summary = {metric: None for metric in RAGAS_METRIC_NAMES}

            for result in mode_results:
                for metric in RAGAS_METRIC_NAMES:
                    result[metric] = None
                result["ragas_error"] = "RAGAS skipped by --skip-ragas."
        else:
            log_eval(f"Computing RAGAS metrics for {MODE_LABELS[mode]}...")
            ragas_started_at = time.perf_counter()

            mode_results, ragas_summary = compute_ragas_for_mode(mode_results)

            ragas_elapsed = time.perf_counter() - ragas_started_at
            log_eval(f"RAGAS finished in {format_seconds(ragas_elapsed)}.")

        all_results[mode] = mode_results

        summary_rows.append(
            aggregate_mode_summary(
                mode=mode,
                mode_results=mode_results,
                ragas_summary=ragas_summary,
            )
        )

        mode_elapsed = time.perf_counter() - mode_started_at
        log_eval(f"Finished mode {MODE_LABELS[mode]} in {format_seconds(mode_elapsed)}.")

    print_summary_table(summary_rows)

    run_metadata = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "num_examples": len(examples),
        "modes": EVAL_MODES,
        "ragas_skipped": args.skip_ragas,
        "dataset_path": str(DATASET_PATH),
        "total_runtime_seconds": round(time.perf_counter() - run_started_at, 2),
    }

    save_outputs(
        all_results=all_results,
        summary_rows=summary_rows,
        run_metadata=run_metadata,
    )

    total_elapsed = time.perf_counter() - run_started_at
    log_eval(f"Evaluation complete in {format_seconds(total_elapsed)}.")


if __name__ == "__main__":
    main()
