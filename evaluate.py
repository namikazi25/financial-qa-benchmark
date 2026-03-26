"""
evaluate.py

Scores predictions from both models using:
  1. LLM-as-judge (Claude Sonnet 4.5) — correctness, completeness, groundedness
  2. Numeric consistency — do the key financial figures match?

Reads from:
  outputs/predictions_gpt4o_mini.jsonl
  outputs/predictions_gemini_25_flash_lite.jsonl

Writes to:
  outputs/evaluation_gpt4o_mini.jsonl
  outputs/evaluation_gemini_25_flash_lite.jsonl
  outputs/comparison_summary.csv

Run:
    python evaluate.py

Make sure main.py has been run first.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from main import load_predictions
from models import ANSWER_MODELS, JUDGE_MODEL, GPT4O_MINI, GEMINI_25_FLASH_LITE, call_model
from prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt

OUTPUTS_DIR = Path("outputs")

EVAL_FILES = {
    GPT4O_MINI:           OUTPUTS_DIR / "evaluation_gpt4o_mini.jsonl",
    GEMINI_25_FLASH_LITE: OUTPUTS_DIR / "evaluation_gemini_25_flash_lite.jsonl",
}

SUMMARY_FILE = OUTPUTS_DIR / "comparison_summary.csv"

DELAY_BETWEEN_CALLS = 1.5  # Claude is the bottleneck — give it breathing room


# ---------------------------------------------------------------------------
# Numeric consistency
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> set[str]:
    """
    Extract all numeric tokens from text.
    Captures integers, decimals, percentages, dollar amounts.
    Examples: "14.16", "22%", "$4.2", "1,200"
    """
    if not text:
        return set()
    return set(re.findall(r"\d[\d,]*\.?\d*%?", text))


def numeric_consistency(predicted: str | None, reference: str) -> float:
    """
    What fraction of the numbers in the reference appear in the prediction?

    Returns:
        1.0  — all reference numbers present
        0.5  — half present
        0.0  — none present, or prediction is None
        None — reference has no numbers (metric not applicable)
    """
    ref_numbers = _extract_numbers(reference)

    if not ref_numbers:
        return None  # not applicable for this row

    if not predicted:
        return 0.0

    # Guard against answer being a dict or non-string (can happen with malformed model output)
    if not isinstance(predicted, str):
        predicted = json.dumps(predicted)

    pred_numbers = _extract_numbers(predicted)
    matched = ref_numbers & pred_numbers
    return round(len(matched) / len(ref_numbers), 3)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def run_judge(record: dict) -> dict:
    """
    Call Claude Sonnet 4.5 to score a single prediction.

    Returns judge scores merged with the original record.
    """
    user_message = build_judge_prompt(
        question=record["question"],
        context=record["context"],
        reference_answer=record["reference_answer"],
        predicted_answer=record["answer"],
        abstained=record["abstain"],
    )

    response = call_model(
        model=JUDGE_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=256,
    )

    parsed = response["parsed"] or {}

    correctness   = parsed.get("correctness",   None)
    completeness  = parsed.get("completeness",  None)
    groundedness  = parsed.get("groundedness",  None)
    verdict       = parsed.get("verdict",       None)

    # Recompute total ourselves — don't trust the model's arithmetic
    if all(v is not None for v in [correctness, completeness, groundedness]):
        total = correctness + completeness + groundedness
    else:
        total = None

    return {
        **record,
        "judge_correctness":  correctness,
        "judge_completeness": completeness,
        "judge_groundedness": groundedness,
        "judge_total":        total,
        "judge_verdict":      verdict,
        "judge_error":        response["error"],
        "judge_cost_usd":     response["estimated_cost_usd"],
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(model: str) -> list[dict]:
    predictions = load_predictions(model)
    out_path    = EVAL_FILES[model]
    results     = []

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model}")
    print(f"{'=' * 60}")

    total_judge_cost = 0.0
    judge_errors     = 0

    for record in predictions:
        sample_id = record["sample_id"]

        # Numeric consistency — no API call needed
        num_score = numeric_consistency(record["answer"], record["reference_answer"])

        # LLM judge
        evaluated = run_judge(record)
        evaluated["numeric_consistency"] = num_score

        results.append(evaluated)
        total_judge_cost += evaluated.get("judge_cost_usd") or 0.0

        if evaluated["judge_error"]:
            judge_errors += 1

        status = "ABSTAIN" if record["abstain"] else "OK"
        total  = evaluated["judge_total"]
        score_str = f"{total}/9" if total is not None else "ERR"

        print(
            f"  [{sample_id:02d}] {status:<8} "
            f"judge={score_str:<5}  "
            f"num={str(num_score):<5}  "
            f"subset={record.get('subset_type','?')}"
        )

        time.sleep(DELAY_BETWEEN_CALLS)

    # Save per-row evaluation
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n  Judge errors : {judge_errors}")
    print(f"  Judge cost   : ${total_judge_cost:.4f}")
    print(f"  Saved to     : {out_path}")

    return results


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def build_summary(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Aggregate per-row scores into a model-level comparison table.
    """
    rows = []

    for model, results in all_results.items():
        df = pd.DataFrame(results)

        total_rows     = len(df)
        answered       = df[~df["abstain"]]
        edge_cases     = df[df["subset_type"] == "edge_case"]
        edge_abstained = edge_cases[edge_cases["abstain"]]

        # Judge scores — only for answered rows (abstentions score separately)
        judge_scores = answered["judge_total"].dropna()
        correctness  = answered["judge_correctness"].dropna()
        completeness = answered["judge_completeness"].dropna()
        groundedness = answered["judge_groundedness"].dropna()

        # Numeric consistency — only rows where metric applies
        num_applicable = df["numeric_consistency"].dropna()

        # Cost
        answer_cost = df["estimated_cost_usd"].sum()
        judge_cost  = df["judge_cost_usd"].sum()

        # Cost per correct answer — "correct" = judge_total >= 7 out of 9
        correct_answers = (answered["judge_total"] >= 7).sum()
        cost_per_correct = (
            round(answer_cost / correct_answers, 5) if correct_answers > 0 else None
        )

        rows.append({
            "model":                    model,
            "total_rows":               total_rows,
            "answered":                 len(answered),
            "abstentions":              total_rows - len(answered),
            "abstention_rate":          round((total_rows - len(answered)) / total_rows, 3),
            "edge_abstention_rate":     round(len(edge_abstained) / max(len(edge_cases), 1), 3),
            "avg_judge_total":          round(judge_scores.mean(), 2) if len(judge_scores) else None,
            "avg_correctness":          round(correctness.mean(),  2) if len(correctness)  else None,
            "avg_completeness":         round(completeness.mean(), 2) if len(completeness) else None,
            "avg_groundedness":         round(groundedness.mean(), 2) if len(groundedness) else None,
            "avg_numeric_consistency":  round(num_applicable.mean(), 3) if len(num_applicable) else None,
            "correct_answers (>=7/9)":  correct_answers,
            "avg_latency_seconds":      round(df["latency_seconds"].mean(), 3),
            "total_answer_cost_usd":    round(answer_cost, 4),
            "total_judge_cost_usd":     round(judge_cost,  4),
            "cost_per_correct_usd":     cost_per_correct,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    start = _time.time()

    all_results = {}
    for model in ANSWER_MODELS:
        all_results[model] = evaluate_model(model)

    print(f"\n{'=' * 60}")
    print("Building comparison summary...")
    summary = build_summary(all_results)

    print(f"\nSaved to: {SUMMARY_FILE}")
    print(f"\n{summary.T.to_string()}")

    elapsed = round(_time.time() - start, 1)
    print(f"\nTotal evaluation runtime: {elapsed}s")