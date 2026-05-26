"""
Lucid benchmark harness.

Runs N sequential /ask_question requests against a running backend,
measures client-side wall-clock latency per request, then fetches
/metrics for the server-side per-stage P50/P95 breakdown.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --n 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

DEFAULT_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key contributions.",
    "What methods are used?",
    "What are the limitations?",
    "What conclusions are drawn?",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Lucid benchmark harness")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=30, help="Measured request count (excludes warmup)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--questions", type=Path, default=None, help="Optional JSON file with a list of strings")
    parser.add_argument("--doc-name", default=None)
    parser.add_argument("--tag-name", default=None)
    parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=True)
    parser.add_argument("--no-rewrite", dest="rewrite", action="store_false")
    args = parser.parse_args()

    if args.questions and args.questions.exists():
        questions = json.loads(args.questions.read_text())
    else:
        questions = DEFAULT_QUESTIONS

    print(f"Hitting {args.url}/ask_question  n={args.n}  warmup={args.warmup}  rewrite={args.rewrite}")

    client_latencies_ms: list[float] = []
    errors = 0

    for i in range(args.n + args.warmup):
        q = questions[i % len(questions)]
        body = {
            "question": q,
            "doc_name": args.doc_name,
            "tag_name": args.tag_name,
            "rewrite_enabled": args.rewrite,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{args.url}/ask_question", json=body, timeout=300)
            resp.raise_for_status()
        except Exception as exc:
            errors += 1
            print(f"  [{i}] error: {exc}", file=sys.stderr)
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            client_latencies_ms.append(elapsed_ms)
        tag = "warmup  " if i < args.warmup else "measured"
        print(f"  [{i:>3}] {tag} {elapsed_ms:8.1f} ms  q={q[:50]!r}")

    print()
    print(f"=== Client-side wall-clock (N={len(client_latencies_ms)}, errors={errors}) ===")
    if client_latencies_ms:
        print(f"  mean    {statistics.mean(client_latencies_ms):8.1f} ms")
        print(f"  median  {statistics.median(client_latencies_ms):8.1f} ms")
        print(f"  p95     {_percentile(client_latencies_ms, 95):8.1f} ms")
        print(f"  max     {max(client_latencies_ms):8.1f} ms")

    try:
        metrics = requests.get(f"{args.url}/metrics", timeout=10).json()
    except Exception as exc:
        print(f"could not fetch /metrics: {exc}", file=sys.stderr)
        return 0

    print()
    print("=== Server-side /metrics ===")
    cs = metrics.get("cold_start_ms")
    print(f"  cold_start  {cs:.1f} ms" if cs is not None else "  cold_start  n/a")
    print(f"  buffered    {metrics.get('request_count')} requests")
    print()
    print(f"  {'stage':<14} {'count':>6}  {'p50_ms':>10}  {'p95_ms':>10}")
    for name, s in metrics.get("stages", {}).items():
        p50 = s.get("p50_ms")
        p95 = s.get("p95_ms")
        p50_s = f"{p50:10.1f}" if p50 is not None else f"{'n/a':>10}"
        p95_s = f"{p95:10.1f}" if p95 is not None else f"{'n/a':>10}"
        print(f"  {name:<14} {s.get('count', 0):>6}  {p50_s}  {p95_s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())