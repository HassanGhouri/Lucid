# Lucid Phase 2 Evaluation

This folder contains the Phase 2 evaluation pipeline for Lucid.

It evaluates four system modes:

1. Dense Only
2. Hybrid
3. Hybrid + Rerank
4. Hybrid + Rerank + Rewrite

The goal is to produce real retrieval and answer-quality numbers for the README, resume, and project writeup.

## Files

- `eval_dataset.json`  
  100 manually curated evaluation questions.

- `prepare_eval_corpus.py`  
  Preloads the four demo/eval PDFs into Qdrant.

- `eval_pipeline.py`  
  Runs one question through one selected Lucid mode.

- `eval_judge.py`  
  Uses an OpenAI LLM judge to score grounding, correctness, and usefulness.

- `ragas_eval.py`  
  Computes RAGAS metrics.

- `run_eval.py`  
  Main evaluation runner.

## Required eval PDFs

Create this folder:

```bash
mkdir -p data/eval_pdfs