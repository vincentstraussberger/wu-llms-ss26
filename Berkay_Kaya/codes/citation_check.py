"""
citation_check.py - Stage 2: systematic citation validity check
(WU LLMs SS26, Team 11)

Part of the two-script evaluation pipeline for REPORT_v2.md:
  Stage 1 - evaluation.py        (broad proxy evaluation, all 643 Qs)
  Stage 2 - citation_check.py    (systematic citation validity, all 643 Qs)
  Stage 3 - visualize_results.py (figures from the final CSVs)
Orchestrator: run_all_evaluations.py

Note: evaluation_gold.py is NOT part of this pipeline. The course-shared
EStG-§23 file contains LLM-generated answers and is not a valid reference.
See REPORT_v2.md §2.1 for the rationale.

Feeds REPORT_v2.md §4.2 ("Citation hallucination - systematic count").

Context: evaluation.py (Stage 1) only counts whether an answer contains a §
symbol (§3.1 of REPORT.md). It cannot tell whether the cited paragraph
actually exists. This script closes that gap by building a paragraph index
from the three indexed PDFs (EStG, KStG, UStG) and classifying every § citation
produced by the three models across 1,929 model answers (643 questions x 3
models). The total number of parsed citations is computed at runtime and
reported in the summary CSV (typically a few thousand; the exact number
depends on the regex match and the models' citation habits).

Each parsed citation is classified as:
  * grounded      - § number exists in the cited law (EStG / KStG / UStG)
  * hallucinated  - § number does NOT exist in the cited law
  * out_of_scope  - cited law is not one we indexed (ABGB, GewO, GrEStG, BAO, ...)

This is a strict existence check. It does NOT verify that the § is the
correct norm for the question (semantic citation correctness / misattribution);
that would require human annotation and is discussed anecdotally in §4.3 of
REPORT.md and flagged as a remaining caveat in §4.7.

Usage:
    python3 citation_check.py
Outputs:
    ../results/citation_check_summary.csv    - one row per model
    ../results/citation_check_per_answer.csv - per-answer counts
"""

import os
import re
import pandas as pd
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
CONTEXT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "context", "Gesetze"))

PDF_FILES = {
    "EStG": os.path.join(CONTEXT_DIR, "EStG.pdf"),
    "KStG": os.path.join(CONTEXT_DIR, "KStG 1988, Fassung vom 03.04.2026.pdf"),
    "UStG": os.path.join(CONTEXT_DIR, "UStG.pdf"),
}

MODEL_FILES = {
    "Model1_API_Llama70B":      os.path.join(RESULTS_DIR, "model1_api_results.csv"),
    "Model2_Finetuned_Gemma2B": os.path.join(RESULTS_DIR, "model2_finetuned_results.csv"),
    "Model3_RAG_Gemma2B":       os.path.join(RESULTS_DIR, "model3_rag_results.csv"),
}

# Laws the user's answers might cite. Those marked True are indexed and
# verifiable; the rest are counted as out-of-scope.
INDEXED_LAWS = {"EStG", "KStG", "UStG"}
KNOWN_LAWS = INDEXED_LAWS | {
    "ABGB", "GewO", "GrEStG", "BAO", "BewG", "KomStG", "FinStrG",
    "GmbHG", "AktG", "EStR", "UStR", "KStR", "DBA", "SVG", "FamLAG",
    "AVRAG", "AngG", "VBG", "BDG", "DVG", "PSG", "WEG", "MRG",
    "UGB", "EKEG", "InsO", "KO", "IO", "UmgrStG",
}

# ---------------------------------------------------------------------------
# 2. Build paragraph index from the three PDFs
# ---------------------------------------------------------------------------
# We scan each PDF and collect every § number that appears. A § that appears
# anywhere in the document (heading or cross-reference) is counted as "exists"
# - this is a LENIENT check on purpose: we only flag clear hallucinations like
# a § number that never appears in the law at all.
# Pattern: §, optional whitespace, a 1-3 digit number, optional single letter
# suffix (e.g. "§ 37a").
PARAGRAPH_RE = re.compile(r"§\s*(\d{1,3}[a-z]?)", re.IGNORECASE)

def build_paragraph_index():
    index = {}
    for law, path in PDF_FILES.items():
        doc = fitz.open(path)
        text = " ".join(page.get_text() for page in doc)
        nums = set(m.group(1).lower() for m in PARAGRAPH_RE.finditer(text))
        index[law] = nums
        print(f"  {law}: {len(nums)} distinct § numbers indexed (sample: {sorted(list(nums))[:5]})")
    return index


# ---------------------------------------------------------------------------
# 3. Parse citations from a model answer
# ---------------------------------------------------------------------------
# A citation is: "§ <num>[suffix] ... <lawname>" within a ~80 char window.
# Examples matched:
#   "§ 23 EStG"
#   "§ 23 Abs. 1 Z 1 EStG 1988"
#   "§ 3 Abs. 1 Z 13 EStG"
LAW_ALTERNATION = "|".join(sorted(KNOWN_LAWS, key=len, reverse=True))
CITATION_RE = re.compile(
    r"§\s*(\d{1,3}[a-z]?)"              # § <num>[letter]
    r"(?:[^§\n]{0,80}?)"                # up to 80 chars, no § or newline
    r"\b(" + LAW_ALTERNATION + r")\b",  # law name
    re.IGNORECASE
)

def parse_citations(text):
    cites = []
    for m in CITATION_RE.finditer(text):
        num = m.group(1).lower()
        law = m.group(2)
        # Canonicalize law casing
        for known in KNOWN_LAWS:
            if known.lower() == law.lower():
                law = known
                break
        cites.append((num, law))
    return cites


# ---------------------------------------------------------------------------
# 4. Classify each citation
# ---------------------------------------------------------------------------
def classify(num, law, index):
    if law not in INDEXED_LAWS:
        return "out_of_scope"
    if num in index[law]:
        return "grounded"
    return "hallucinated"


# ---------------------------------------------------------------------------
# 5. Run over all answers for all models
# ---------------------------------------------------------------------------
def main():
    print("Building paragraph index from PDFs...")
    index = build_paragraph_index()
    print()

    summary_rows   = []
    per_answer_rows = []

    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        n_ans          = len(df)
        total_cites    = 0
        n_grounded     = 0
        n_hallucinated = 0
        n_out_of_scope = 0
        ans_with_grounded      = 0
        ans_with_hallucinated  = 0
        ans_with_any_citation  = 0

        for _, row in df.iterrows():
            cites = parse_citations(str(row["answer"]))
            if cites:
                ans_with_any_citation += 1
            has_g, has_h = False, False
            for num, law in cites:
                total_cites += 1
                cls = classify(num, law, index)
                if   cls == "grounded":      n_grounded     += 1; has_g = True
                elif cls == "hallucinated":  n_hallucinated += 1; has_h = True
                else:                        n_out_of_scope += 1
            if has_g: ans_with_grounded     += 1
            if has_h: ans_with_hallucinated += 1

            per_answer_rows.append({
                "model": model,
                "id": row["id"],
                "n_citations": len(cites),
                "n_grounded": sum(1 for n, l in cites if classify(n, l, index) == "grounded"),
                "n_hallucinated": sum(1 for n, l in cites if classify(n, l, index) == "hallucinated"),
                "n_out_of_scope": sum(1 for n, l in cites if classify(n, l, index) == "out_of_scope"),
            })

        # Grounded-rate is computed only over verifiable citations (in-scope),
        # because out-of-scope ones we cannot prove right OR wrong.
        verifiable = n_grounded + n_hallucinated
        grounded_rate = (n_grounded / verifiable) if verifiable else float("nan")

        summary_rows.append({
            "model": model,
            "n_answers": n_ans,
            "answers_with_any_citation":        ans_with_any_citation,
            "pct_answers_with_any_citation":    100 * ans_with_any_citation / n_ans,
            "total_citations_parsed":           total_cites,
            "grounded":                         n_grounded,
            "hallucinated":                     n_hallucinated,
            "out_of_scope":                     n_out_of_scope,
            "grounded_rate_of_verifiable":      grounded_rate,
            "answers_with_grounded_citation":   ans_with_grounded,
            "answers_with_hallucinated_cite":   ans_with_hallucinated,
        })

    summary_df   = pd.DataFrame(summary_rows)
    per_ans_df   = pd.DataFrame(per_answer_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "citation_check_summary.csv"),   index=False)
    per_ans_df.to_csv(os.path.join(RESULTS_DIR, "citation_check_per_answer.csv"), index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("=== CITATION CHECK SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nFiles written to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
