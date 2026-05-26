"""
Load test for Lucid /ask_question.

Run on the DEPLOY HOST only (Linux). Do NOT run against Mac dev — MPS
rerank contention under concurrency produces misleading numbers.

Setup before running:
  1. Bump rate_limit_ask in deploy .env to e.g. "10000/minute" — otherwise
     slowapi returns 429 immediately and the test measures rate-limit
     behavior, not real latency.
  2. Make sure Qdrant on the deploy host has the eval corpus loaded:
       python -m backend.app.eval.prepare_eval_corpus
  3. Start the server with production worker count:
       python -m scripts.run_server   # honors settings.workers
  4. pip install locust (on whichever box runs locust, not necessarily the
     server box).

Run for each concurrency level (1 / 5 / 10 / 20):

  for N in 1 5 10 20; do
    locust -f scripts/loadtest/locustfile.py \\
        --host http://<deploy-host>:8000 \\
        --users "$N" --spawn-rate 1 \\
        --run-time 5m --headless \\
        --csv "loadtest_${N}users"
  done

Headline number per run = P95 from the /ask_question row of
loadtest_${N}users_stats.csv.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

EVAL_DATASET_PATH = Path("backend/app/eval/eval_dataset.json")


def _load_questions() -> list[str]:
    with EVAL_DATASET_PATH.open(encoding="utf-8") as fh:
        examples = json.load(fh)
    return [example["question"] for example in examples]


QUESTIONS = _load_questions()


class LucidUser(HttpUser):
    # Realistic think time between consecutive turns. We're measuring
    # response latency under steady concurrent load, not max throughput.
    wait_time = between(1, 3)

    @task
    def ask_question(self) -> None:
        payload = {
            "question": random.choice(QUESTIONS),
            "rewrite_enabled": True,
        }
        self.client.post("/ask_question", json=payload, name="/ask_question")