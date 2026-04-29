"""
run_all_evaluations.py - Orchestrator for the Stage 3 evaluation pipeline
(WU LLMs SS26, Team 11)

Runs the three stages in order:
  Stage 1 - evaluation.py        (broad proxy evaluation, all 643 Qs)
  Stage 2 - citation_check.py    (systematic citation validity, all 643 Qs)
  Stage 3 - visualize_results.py (figures from the final CSVs)

Note: evaluation_gold.py is NOT run here. The course-shared EStG-§23 file
contains LLM-generated answers and is not used as a reference standard.
See REPORT_v2.md §2.1 for the rationale.

Usage:
    python3 run_all_evaluations.py

Exits non-zero on the first failure.
"""

import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    ("Stage 1 - broad proxy evaluation",        "evaluation.py"),
    ("Stage 2 - citation validity check",       "citation_check.py"),
    ("Stage 3 - figures from the final CSVs",   "visualize_results.py"),
]


def run_stage(label, script):
    path = os.path.join(HERE, script)
    print(f"\n=== {label} ===")
    print(f"  running: {script}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=HERE)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED after {dt:.1f}s (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  ok ({dt:.1f}s)")


def main():
    print("Team 11 Stage 3 evaluation pipeline")
    for label, script in PIPELINE:
        run_stage(label, script)
    print("\nAll stages completed.")


if __name__ == "__main__":
    main()
