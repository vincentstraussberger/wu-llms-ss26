"""
evaluation.py - Stage 1: broad proxy evaluation (WU LLMs SS26, Team 11)

Part of the two-script evaluation pipeline for REPORT_v2.md:
  Script 1 - evaluation.py     (proxy evaluation, all 643 Qs)   → REPORT_v2 §3
  Script 2 - citation_check.py (citation validity, all 643 Qs)  → REPORT_v2 §4.2

No verified gold answers exist for this dataset. A course-shared EStG-§23
file contains LLM-generated answers (not human-written), so it is not used
as a reference. The evaluation relies on two complementary reference-free or
proxy methods, both covering all 643 questions:
  1. Proxy similarity: ROUGE/BLEU of each model against Model 1 as silver ref.
  2. Citation validity: citation_check.py — checks whether cited § numbers
     actually exist in the named law's PDF (no reference model required).

Goal: Compute ROUGE, BLEU, and intrinsic stats for all 643 questions using
Model 1 (LLaMA-3.3-70B via Groq) as a silver / pseudo-reference, and produce
the per-question ROUGE-L matrix that feeds the §4 error-analysis ranking.

Citation metric note: the `§`-count and `% answers with §` columns in the main
table are a *crude existence-of-symbol* check (§3.1 of REPORT.md). They count
the `§` character; they do NOT verify that the cited paragraph exists in the
law. That is a separate, stricter check produced by citation_check.py and
discussed in REPORT.md §4.2.

Metrics:
  * ROUGE-1, ROUGE-2, ROUGE-L (word/sequence overlap)
  * BLEU (sacrebleu, corpus-level)
  * Intrinsic: average answer length, legal citation count ("§" markers)

BERTScore is optional (requires torch); see the BERTSCORE_NOTE at the end.

Usage:
    python3 evaluation.py
Outputs:
    ../results/evaluation_main_table.csv    - main table (one row per model)
    ../results/evaluation_pairwise.csv      - pairwise model similarity
    ../results/evaluation_per_question.csv  - per-question ROUGE-L (for §4 ranking)
    ../results/evaluation_error_analysis.md - 10 lowest-agreement examples
"""

import os
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
DATASET_CSV = os.path.abspath(os.path.join(HERE, "..", "..", "dataset_clean.csv"))

MODEL_FILES = {
    "Model1_API_Llama70B":      os.path.join(RESULTS_DIR, "model1_api_results.csv"),
    "Model2_Finetuned_Gemma2B": os.path.join(RESULTS_DIR, "model2_finetuned_results.csv"),
    "Model3_RAG_Gemma2B":       os.path.join(RESULTS_DIR, "model3_rag_results.csv"),
}

# Model 1 serves as pseudo-reference (largest model, API-based teacher).
REFERENCE_MODEL = "Model1_API_Llama70B"


# ---------------------------------------------------------------------------
# 2. Load all model outputs + dataset
# ---------------------------------------------------------------------------
def load_answers():
    """Load all model CSVs and align them by id with the dataset."""
    dataset = pd.read_csv(DATASET_CSV)
    # Strip BOM from first column if present
    dataset.columns = [c.lstrip("\ufeff") for c in dataset.columns]

    merged = dataset[["id", "prompt"]].copy()
    for name, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        df = df.rename(columns={"answer": name})
        merged = merged.merge(df[["id", name]], on="id", how="left")

    # Sanity: no NaNs, 643 rows
    assert len(merged) == 643, f"expected 643 rows, got {len(merged)}"
    for name in MODEL_FILES:
        assert merged[name].notna().all(), f"{name} has missing answers"
    return merged


# ---------------------------------------------------------------------------
# 3. Metric helpers
# ---------------------------------------------------------------------------
def compute_rouge(preds, refs):
    """Return mean ROUGE-1, ROUGE-2, ROUGE-L F-measures + per-item ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (reference, prediction)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1),
        "rouge2": sum(r2) / len(r2),
        "rougeL": sum(rl) / len(rl),
    }, rl


def compute_bleu(preds, refs):
    """Corpus-level BLEU using sacrebleu with German tokenizer."""
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl")
    return bleu.score  # 0-100 scale


def intrinsic_stats(answers):
    """Simple intrinsic descriptors of answer style."""
    n_chars = [len(a) for a in answers]
    n_words = [len(a.split()) for a in answers]
    n_cites = [a.count("§") for a in answers]          # count of legal § refs
    has_cite = [1 if c > 0 else 0 for c in n_cites]
    return {
        "avg_chars": sum(n_chars) / len(n_chars),
        "avg_words": sum(n_words) / len(n_words),
        "avg_paragraph_cites": sum(n_cites) / len(n_cites),
        "pct_answers_with_cite": 100 * sum(has_cite) / len(has_cite),
    }


# ---------------------------------------------------------------------------
# 4. Evaluation pipeline
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_answers()
    print(f"Loaded {len(df)} questions and {len(MODEL_FILES)} models.\n")

    models = list(MODEL_FILES.keys())

    # 4a - Intrinsic stats per model
    print("Computing intrinsic stats...")
    intrinsic_rows = []
    for m in models:
        s = intrinsic_stats(df[m].tolist())
        s["model"] = m
        intrinsic_rows.append(s)
    intrinsic_df = pd.DataFrame(intrinsic_rows).set_index("model")

    # 4b - Main table: each model vs the pseudo-reference (Model 1)
    print(f"Computing ROUGE/BLEU vs reference ({REFERENCE_MODEL})...")
    ref = df[REFERENCE_MODEL].tolist()
    main_rows = []
    per_question = {"id": df["id"].tolist(), "prompt": df["prompt"].tolist()}
    for m in models:
        preds = df[m].tolist()
        rouge, rl_list = compute_rouge(preds, ref)
        bleu = compute_bleu(preds, ref)
        main_rows.append({
            "model": m,
            "rouge1_vs_ref": rouge["rouge1"],
            "rouge2_vs_ref": rouge["rouge2"],
            "rougeL_vs_ref": rouge["rougeL"],
            "bleu_vs_ref":   bleu,
            **intrinsic_df.loc[m].to_dict(),
        })
        per_question[f"rougeL_{m}"] = rl_list
    main_df = pd.DataFrame(main_rows)

    # 4c - Pairwise similarity between every pair of models.
    # NOTE: ROUGE-F and BLEU are DIRECTIONAL (one argument is prediction, the
    # other is reference). To produce a symmetric "similarity" we compute both
    # directions and report the mean. Both raw directions are also kept in the
    # CSV for full transparency.
    print("Computing pairwise similarity (symmetric)...")
    pair_rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            a_txt, b_txt = df[a].tolist(), df[b].tolist()
            # a treated as prediction, b as reference
            rouge_ab, _ = compute_rouge(a_txt, b_txt)
            bleu_ab = compute_bleu(a_txt, b_txt)
            # b treated as prediction, a as reference
            rouge_ba, _ = compute_rouge(b_txt, a_txt)
            bleu_ba = compute_bleu(b_txt, a_txt)
            pair_rows.append({
                "model_a": a, "model_b": b,
                "rouge1_mean": (rouge_ab["rouge1"] + rouge_ba["rouge1"]) / 2,
                "rouge2_mean": (rouge_ab["rouge2"] + rouge_ba["rouge2"]) / 2,
                "rougeL_mean": (rouge_ab["rougeL"] + rouge_ba["rougeL"]) / 2,
                "bleu_mean":   (bleu_ab + bleu_ba) / 2,
                # raw directional values for transparency
                "bleu_a_as_pred": bleu_ab,
                "bleu_b_as_pred": bleu_ba,
            })
    pair_df = pd.DataFrame(pair_rows)

    # 4d - Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_main_table.csv"), index=False)
    pair_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_pairwise.csv"), index=False)
    pd.DataFrame(per_question).to_csv(
        os.path.join(RESULTS_DIR, "evaluation_per_question.csv"), index=False
    )

    # 4e - Error analysis: examples where both small models disagree most with reference
    print("Running simple error analysis...")
    pq = pd.DataFrame(per_question)
    pq["avg_rougeL_small"] = (
        pq["rougeL_Model2_Finetuned_Gemma2B"] + pq["rougeL_Model3_RAG_Gemma2B"]
    ) / 2
    worst = pq.nsmallest(10, "avg_rougeL_small")
    lines = ["# Error analysis - 10 questions with lowest agreement to reference", ""]
    for _, row in worst.iterrows():
        lines.append(f"## {row['id']}  (avg ROUGE-L of M2/M3 vs M1 = {row['avg_rougeL_small']:.3f})")
        lines.append(f"**Prompt:** {row['prompt']}")
        lines.append(f"- M1 ref excerpt: {df[df['id']==row['id']][REFERENCE_MODEL].iloc[0][:250]}...")
        lines.append(f"- M2 excerpt:     {df[df['id']==row['id']]['Model2_Finetuned_Gemma2B'].iloc[0][:250]}...")
        lines.append(f"- M3 excerpt:     {df[df['id']==row['id']]['Model3_RAG_Gemma2B'].iloc[0][:250]}...")
        lines.append("")
    with open(os.path.join(RESULTS_DIR, "evaluation_error_analysis.md"), "w") as f:
        f.write("\n".join(lines))

    # 4f - Pretty print main table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== MAIN TABLE (reference = Model 1, Llama-3.3-70B) ===")
    print(main_df.to_string(index=False))
    print("\n=== PAIRWISE SIMILARITY ===")
    print(pair_df.to_string(index=False))
    print("\nFiles written to:", RESULTS_DIR)


# BERTSCORE_NOTE:
# BERTScore needs torch + transformers, which is heavy to install locally.
# To compute it, run this extra block on Colab / Kaggle:
#
#   !pip install bert-score
#   from bert_score import score
#   P, R, F1 = score(preds, refs, lang="de", model_type="bert-base-multilingual-cased")
#   print("BERTScore F1:", F1.mean().item())
#
# We report BLEU and ROUGE in the main table; BERTScore can be added later
# without changing the rest of the pipeline.


if __name__ == "__main__":
    main()
