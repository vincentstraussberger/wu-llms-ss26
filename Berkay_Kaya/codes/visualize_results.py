"""
visualize_results.py - Stage 3: report-ready figures from the final CSVs
(WU LLMs SS26, Team 11)

Part of the two-script evaluation pipeline for REPORT_v2.md:
  Stage 1 - evaluation.py        (broad proxy evaluation, all 643 Qs)
  Stage 2 - citation_check.py    (systematic citation validity, all 643 Qs)
  Stage 3 - visualize_results.py (figures from the final CSVs)
Orchestrator: run_all_evaluations.py

Reads the CSV outputs from Stages 1-2 and writes three PNGs into
../results/visualizations/. Every plotted number comes directly from the CSVs
- no values are hard-coded or manually edited.

Note: fig_gold_results was removed. The course-shared EStG-§23 file contains
LLM-generated answers (not verified gold), so it is not used as a reference.
See REPORT_v2.md §2.1 for the rationale.

Figures:
  * fig_main_results.png      - grouped bar: ROUGE-1 / ROUGE-L / BLEU vs
                                the M1 silver reference (all 643 Qs).
                                ROUGE on left axis (0-1), BLEU on right (0-100).
  * fig_citation_validity.png - stacked bar per model: grounded /
                                hallucinated / out-of-scope citation counts
                                (all 643 Qs, all 1 929 parsed citations).
  * fig_diagnostic_profile.png - diagnostic signal profile per model (§4):
                                share of the 643 answers in each of four
                                deterministic diagnostic categories computed
                                from the citation and per-question CSVs.
                                Renamed from "error-category profile" because
                                several categories are diagnostic signals,
                                not confirmed errors.

Usage:
    python3 visualize_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths and display labels
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)

# Mapping from the raw model keys used in the CSVs to clean display labels.
DISPLAY = {
    "Model1_API_Llama70B":      "M1 API\n(LLaMA-3.3-70B)",
    "Model2_Finetuned_Gemma2B": "M2 Fine-tuned\n(Gemma-2-2B)",
    "Model3_RAG_Gemma2B":       "M3 RAG\n(Gemma-2-2B)",
}

# Stable model order so all four figures use the same x-axis layout.
MODEL_ORDER = list(DISPLAY.keys())

# Matplotlib default cycle colours are sufficient and friendly for print.
C_ROUGE1 = "#4C72B0"
C_ROUGEL = "#55A868"
C_BLEU   = "#C44E52"

C_GROUND = "#4C72B0"
C_HALL   = "#C44E52"
C_OOS    = "#B0B0B0"


def _display_labels(models):
    return [DISPLAY.get(m, m) for m in models]


# ---------------------------------------------------------------------------
# Figure 1 - main proxy results vs M1 silver reference
# ---------------------------------------------------------------------------

def fig_main_results():
    csv = os.path.join(RESULTS_DIR, "evaluation_main_table.csv")
    df = pd.read_csv(csv).set_index("model").reindex(MODEL_ORDER)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    x = np.arange(len(df))
    width = 0.25

    ax1.bar(x - width, df["rouge1_vs_ref"], width,
            color=C_ROUGE1, label="ROUGE-1")
    ax1.bar(x,         df["rougeL_vs_ref"], width,
            color=C_ROUGEL, label="ROUGE-L")
    ax2.bar(x + width, df["bleu_vs_ref"],   width,
            color=C_BLEU,   label="BLEU")

    ax1.set_ylabel("ROUGE F-measure (0-1)")
    ax2.set_ylabel("BLEU (0-100)")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(_display_labels(df.index), fontsize=9)
    ax1.set_title(
        "Stage 1 - Proxy evaluation vs M1 silver reference (all 643 Qs)"
    )

    # One combined legend, placed outside the bars.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_main_results.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 - citation validity stacked bar
# ---------------------------------------------------------------------------

def fig_citation_validity():
    csv = os.path.join(RESULTS_DIR, "citation_check_summary.csv")
    df = pd.read_csv(csv).set_index("model").reindex(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))

    grounded = df["grounded"].values
    halluc   = df["hallucinated"].values
    oos      = df["out_of_scope"].values

    ax.bar(x, grounded, color=C_GROUND, label="grounded")
    ax.bar(x, halluc,   bottom=grounded, color=C_HALL, label="hallucinated")
    ax.bar(x, oos,      bottom=grounded + halluc, color=C_OOS,
           label="out-of-scope")

    # Annotate each segment with its absolute count for direct readability.
    for i, (g, h, o) in enumerate(zip(grounded, halluc, oos)):
        if g > 0:
            ax.text(i, g / 2, str(int(g)), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        if h > 0:
            ax.text(i, g + h / 2, str(int(h)), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        if o > 0:
            ax.text(i, g + h + o / 2, str(int(o)), ha="center", va="center",
                    color="black", fontsize=9)

    ax.set_ylabel("parsed citations")
    ax.set_xticks(x)
    ax.set_xticklabels(_display_labels(df.index), fontsize=9)
    ax.set_title(
        "Stage 2 - Citation validity (existence check, all 643 Qs)"
    )
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_citation_validity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 - diagnostic signal profile per model
# ---------------------------------------------------------------------------
#
# Four deterministic categories, each expressed as a share of the 643 answers
# per model. All values come from existing Stage 3 CSV outputs:
#
#   1. No verifiable citation
#        = (n_answers - answers_with_any_citation) / n_answers
#      Source: citation_check_summary.csv. The answer contains no parseable
#      "§ X ... <lawname>" pair the citation check could verify.
#
#   2. Any hallucinated citation
#        = answers_with_hallucinated_cite / n_answers
#      Source: citation_check_summary.csv. At least one fabricated § number
#      (the law is indexed but the number does not exist in it).
#
#   3. Any out-of-scope citation
#        = count_{id : n_out_of_scope > 0} / n_answers
#      Source: citation_check_per_answer.csv. The answer cites at least one
#      law we did not index (ABGB, GewO, GrEStG, BAO, ...), so existence
#      cannot be confirmed or refuted.
#
#   4. Low silver agreement (rougeL < 0.10)
#      Source: evaluation_per_question.csv. Share of questions where the
#      per-question ROUGE-L vs the M1 silver reference is below 0.10. M1 is
#      trivially 0 % because it is the reference (ROUGE-L = 1.0 on every Q).
#
# No hand-labelling, no thresholds beyond the explicit rougeL < 0.10 cut, no
# invented counts.

LOW_AGREEMENT_THRESHOLD = 0.10

ERROR_CATEGORIES = [
    "No verifiable\ncitation",
    "Any hallucinated\ncitation",
    "Any out-of-scope\ncitation",
    f"Low silver agreement\n(ROUGE-L < {LOW_AGREEMENT_THRESHOLD:.2f})",
]


def _compute_error_profile():
    """Return a DataFrame: rows = models (MODEL_ORDER), cols = ERROR_CATEGORIES,
    values = percentage of the 643 answers in each category."""
    summary = pd.read_csv(
        os.path.join(RESULTS_DIR, "citation_check_summary.csv")
    ).set_index("model")
    per_answer = pd.read_csv(
        os.path.join(RESULTS_DIR, "citation_check_per_answer.csv")
    )
    per_q = pd.read_csv(
        os.path.join(RESULTS_DIR, "evaluation_per_question.csv")
    )

    rows = {}
    for m in MODEL_ORDER:
        n = int(summary.loc[m, "n_answers"])
        no_cite   = n - int(summary.loc[m, "answers_with_any_citation"])
        any_hall  = int(summary.loc[m, "answers_with_hallucinated_cite"])
        any_oos   = int(
            (per_answer[per_answer["model"] == m]["n_out_of_scope"] > 0).sum()
        )
        low_agree = int((per_q[f"rougeL_{m}"] < LOW_AGREEMENT_THRESHOLD).sum())

        rows[m] = [
            100.0 * no_cite   / n,
            100.0 * any_hall  / n,
            100.0 * any_oos   / n,
            100.0 * low_agree / n,
        ]

    return pd.DataFrame.from_dict(
        rows, orient="index", columns=ERROR_CATEGORIES
    ).reindex(MODEL_ORDER)


def fig_diagnostic_profile():
    profile = _compute_error_profile()

    fig, ax = plt.subplots(figsize=(10, 5))
    n_cat = len(ERROR_CATEGORIES)
    n_mod = len(MODEL_ORDER)
    x = np.arange(n_cat)
    width = 0.8 / n_mod

    colours = {
        "Model1_API_Llama70B":      C_ROUGE1,
        "Model2_Finetuned_Gemma2B": C_BLEU,
        "Model3_RAG_Gemma2B":       C_ROUGEL,
    }

    for i, m in enumerate(MODEL_ORDER):
        offset = (i - (n_mod - 1) / 2) * width
        vals = profile.loc[m].values
        bars = ax.bar(
            x + offset, vals, width,
            color=colours[m],
            label=DISPLAY[m].replace("\n", " "),
        )
        for b, v in zip(bars, vals):
            if v >= 1.0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )
            elif v > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    color="gray",
                )

    ax.set_ylabel("share of the 643 answers (%)")
    ax.set_ylim(0, max(55, profile.values.max() * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(ERROR_CATEGORIES, fontsize=9)
    ax.set_title(
        "Diagnostic signal profile per model (§4, all 643 Qs)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_diagnostic_profile.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Echo the exact plotted values so the numbers are auditable at run time.
    print("  error-profile values (% of 643 answers):")
    for m in MODEL_ORDER:
        vals = ", ".join(
            f"{cat.replace(chr(10), ' ')}={profile.loc[m, cat]:.2f}%"
            for cat in ERROR_CATEGORIES
        )
        print(f"    {m}: {vals}")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Stage 3 - writing figures to", FIG_DIR)
    fig_main_results()
    fig_citation_validity()
    fig_diagnostic_profile()
    print("done.")


if __name__ == "__main__":
    main()
