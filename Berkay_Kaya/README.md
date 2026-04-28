# Team 11 — Austrian Tax Law Q&A with LLMs

**Course:** WU Wien 4805 Applications of Data Science — LLMs (SS26)  
**Author:** Berkay Kaya  
**Task:** Answer 643 Austrian tax-law questions in German using three different LLM approaches.  
**Full evaluation report:** [`REPORT_v2.md`](REPORT_v2.md)

---

## Models

| # | Approach | Base model | Infrastructure |
|---|----------|------------|----------------|
| 1 | Zero-shot via API | LLaMA-3.3-70B (Groq) | Local Mac |
| 2 | QLoRA fine-tuning | Gemma-2-2B-it | Kaggle T4 |
| 3 | RAG + FAISS | Gemma-2-2B-it | Kaggle T4 |

---

## Results (summary)

| Model | ROUGE-1 vs M1 | BLEU vs M1 | Citation grounded rate |
|-------|:---:|:---:|:---:|
| M1 — API (LLaMA-3.3-70B) | 1.000 | 100.00 | 100 % (0 hallucinated) |
| M3 — RAG (Gemma-2-2B) | 0.317 | 7.88 | 99.6 % (4 hallucinated) |
| M2 — Fine-tuned (Gemma-2-2B) | 0.348 | 5.99 | 88.4 % (82 hallucinated) |

Consistent ranking across both evaluation methods: **M1 > M3 > M2**

---

## Evaluation

Two complementary methods, both covering all 643 questions:

1. **Proxy similarity** — ROUGE-1/L and BLEU of each model against Model 1 as a silver reference (`evaluation.py`)
2. **Citation validity** — every `§ X <lawname>` citation parsed and checked against the indexed law PDFs (`citation_check.py`)

```bash
cd codes
python3 run_all_evaluations.py   # runs all three stages in order
```

---

## File structure

```
Berkay_Kaya/
├── README.md
├── REPORT_v2.md                        ← full evaluation report
├── codes/
│   ├── model1_api_inference.ipynb      ← Model 1 inference
│   ├── model2_finetune.ipynb           ← Model 2 fine-tuning + inference
│   ├── model3_rag.ipynb                ← Model 3 RAG + inference
│   ├── evaluation.py                   ← Stage 1: proxy similarity
│   ├── citation_check.py               ← Stage 2: citation validity
│   ├── visualize_results.py            ← Stage 3: figures
│   └── run_all_evaluations.py          ← orchestrator
└── results/
    ├── model1_api_results.csv
    ├── model2_finetuned_results.csv
    ├── model3_rag_results.csv
    ├── evaluation_main_table.csv
    ├── evaluation_pairwise.csv
    ├── evaluation_per_question.csv
    ├── evaluation_error_analysis.md
    ├── citation_check_summary.csv
    ├── citation_check_per_answer.csv
    └── visualizations/
        ├── fig_main_results.png
        ├── fig_citation_validity.png
        └── fig_diagnostic_profile.png
```
