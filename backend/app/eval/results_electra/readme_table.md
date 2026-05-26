## Evaluation

StudyBuddy 2.0 includes a Phase 2 evaluation pipeline that compares four retrieval/generation modes across a 100-question academic QA benchmark.

The benchmark covers four documents:
- *Reinforcement Learning: An Introduction*
- *Operating Systems: Three Easy Pieces*
- CSC263 Data Structures and Analysis notes
- MAT102 Introduction to Mathematical Proofs notes

Each mode is evaluated using RAGAS metrics and a custom OpenAI LLM-as-judge for grounding, correctness, and usefulness.

### Results

| Mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Judge Grounding | Judge Correctness | Judge Usefulness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Only | 0.956 | 0.886 | 0.780 | 0.958 | 0.981 | 0.990 | 0.987 |
| Hybrid | 0.957 | 0.893 | 0.810 | 0.982 | 0.990 | 1.000 | 0.991 |
| Hybrid + Rerank | **0.962** | 0.894 | **0.830** | **0.982** | **0.993** | **1.000** | 0.991 |
| Hybrid + Rerank + Rewrite | 0.956 | **0.898** | 0.828 | 0.977 | 0.993 | **1.000** | **0.992** |

### Takeaway

Hybrid retrieval improves recall compared with dense-only retrieval, while CrossEncoder reranking improves context precision and answer faithfulness. Query rewriting improves answer relevancy slightly, but the best overall configuration is **Hybrid + Rerank**.
Judge scores cluster near the top of the scale across all modes, so retrieval-quality metrics (CtxPrec/CtxRec/Faithfulness) carry more of the discriminative signal here.