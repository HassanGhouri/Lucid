# Phase 6 — Performance Measurement

Deferred from Phase 5. Run on the deploy host (Linux), not the Mac dev
box. MPS contention under concurrency makes local numbers misleading.

## 1. Load test

Goal: P95 question latency at 1 / 5 / 10 / 20 concurrent users.
Harness: `scripts/loadtest/locustfile.py`. Setup + run commands at the
top of that file.

Record from each `loadtest_${N}users_stats.csv`, `/ask_question` row:

| Users | Requests | P50 (ms) | P95 (ms) | P99 (ms) | Failures |
|---:|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD |

## 2. Performance table for README

### Cold start

Restart server, hit `GET /metrics` once startup completes, read
`cold_start_ms`. For the "Before" column, check out the pre-Phase-5 tag
(or `git stash` the lifespan-preload + cross-encoder-warmup commits) and
repeat.

| | Before Phase 5 | After Phase 5 |
|---|---:|---:|
| Cold start (ms) | TBD | TBD |

### Single-user per-stage latency

From a warmed server, hit `/ask_question` 30–50 times sequentially, then
`GET /metrics`:

| Stage | P50 (ms) | P95 (ms) |
|---|---:|---:|
| rewrite | TBD | TBD |
| embed_query | TBD | TBD |
| retrieve | TBD | TBD |
| rerank | TBD | TBD |
| generate | TBD | TBD |
| judge | TBD | TBD |
| total | TBD | TBD |

### Concurrent P95 (headline)

Pulled directly from Item 1.

| Users | P95 (ms) |
|---:|---:|
| 1 | TBD |
| 5 | TBD |
| 10 | TBD |
| 20 | TBD |

## 3. Where the numbers go

Once filled in:

1. **Project README**: paste the three tables under a new "Performance"
   section, sibling to the existing "Evaluation" section.
2. **Streamlit eval tab**: add a third sub-page ("Performance") that
   renders the same three tables. Static snapshot — read from a saved
   JSON or hardcode. Don't wire to live `/metrics`; these are point-in-
   time measurements, not a dashboard.
3. **Resume bullet** (suggested template):
   *"Sustained P95 < {X}s at {N} concurrent users on a single instance
   with {workers}-worker uvicorn; per-stage latency captured via
   in-process timing middleware exposed at /metrics."*