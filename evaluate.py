"""Evaluation pipeline: LLM-as-judge scoring and numeric consistency.

Scores predictions from both models using:
  1. LLM-as-judge (Claude Sonnet 4.5) -- correctness, completeness, groundedness
  2. Numeric consistency -- do the key financial figures match?

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

import json
import re
import time

import pandas as pd

from config import CONFIG, PROJECT_ROOT
from logger import get_logger
from main import load_predictions
from models import ANSWER_MODELS, GEMINI_25_FLASH_LITE, GPT4O_MINI, JUDGE_MODEL, call_model
from prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt

logger = get_logger(__name__)

_eval_cfg = CONFIG.evaluation
_pipe_cfg = CONFIG.pipeline
OUTPUTS_DIR = PROJECT_ROOT / _pipe_cfg.outputs_dir

EVAL_FILES = {
    GPT4O_MINI: OUTPUTS_DIR / "evaluation_gpt4o_mini.jsonl",
    GEMINI_25_FLASH_LITE: OUTPUTS_DIR / "evaluation_gemini_25_flash_lite.jsonl",
}

SUMMARY_FILE = PROJECT_ROOT / _eval_cfg.summary_file
DELAY_BETWEEN_CALLS = _eval_cfg.delay_between_calls


# ---------------------------------------------------------------------------
# Numeric consistency
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric tokens from text.

    Captures integers, decimals, percentages, and dollar amounts.
    Examples: "14.16", "22%", "$4.2", "1,200".

    Args:
        text: Input text to scan for numbers.

    Returns:
        Set of numeric string tokens found.
    """
    if not text:
        return set()
    return set(re.findall(r"\d[\d,]*\.?\d*%?", text))


def numeric_consistency(predicted: str | None, reference: str) -> float | None:
    """Compute what fraction of reference numbers appear in the prediction.

    Args:
        predicted: Model's predicted answer, or None if abstained.
        reference: Ground truth reference answer.

    Returns:
        Float between 0.0 and 1.0, or None if the reference has no numbers
        (metric not applicable).
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
    """Call Claude Sonnet 4.5 to score a single prediction.

    Args:
        record: A prediction dict containing question, context,
            reference_answer, answer, and abstain fields.

    Returns:
        The original record merged with judge scores (correctness,
        completeness, groundedness, total, verdict) and judge metadata.
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

    correctness = parsed.get("correctness", None)
    completeness = parsed.get("completeness", None)
    groundedness = parsed.get("groundedness", None)
    verdict = parsed.get("verdict", None)

    # Recompute total ourselves — don't trust the model's arithmetic
    if all(v is not None for v in [correctness, completeness, groundedness]):
        total = correctness + completeness + groundedness
    else:
        total = None

    return {
        **record,
        "judge_correctness": correctness,
        "judge_completeness": completeness,
        "judge_groundedness": groundedness,
        "judge_total": total,
        "judge_verdict": verdict,
        "judge_error": response["error"],
        "judge_cost_usd": response["estimated_cost_usd"],
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(model: str) -> list[dict]:
    """Run judge evaluation and numeric consistency on all predictions for a model.

    Args:
        model: Model identifier string.

    Returns:
        List of evaluated record dicts with judge scores and numeric consistency.
    """
    predictions = load_predictions(model)
    out_path = EVAL_FILES[model]
    results: list[dict] = []

    logger.info("%s", "=" * 60)
    logger.info("Evaluating: %s", model)
    logger.info("%s", "=" * 60)

    total_judge_cost = 0.0
    judge_errors = 0

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
        total = evaluated["judge_total"]
        score_str = f"{total}/9" if total is not None else "ERR"

        logger.info(
            "  [%02d] %-8s judge=%-5s  num=%-5s  subset=%s",
            sample_id,
            status,
            score_str,
            str(num_score),
            record.get("subset_type", "?"),
        )

        time.sleep(DELAY_BETWEEN_CALLS)

    # Save per-row evaluation
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("  Judge errors : %d", judge_errors)
    logger.info("  Judge cost   : $%.4f", total_judge_cost)
    logger.info("  Saved to     : %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------


def build_summary(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    """Aggregate per-row scores into a model-level comparison table.

    Args:
        all_results: Mapping of model identifier to list of evaluated records.

    Returns:
        Summary DataFrame with one row per model, saved to comparison_summary.csv.
    """
    rows: list[dict] = []
    threshold = _eval_cfg.judge_correct_threshold

    for model, results in all_results.items():
        df = pd.DataFrame(results)

        total_rows = len(df)
        answered = df[~df["abstain"]]
        edge_cases = df[df["subset_type"] == "edge_case"]
        edge_abstained = edge_cases[edge_cases["abstain"]]

        # Judge scores — only for answered rows (abstentions score separately)
        judge_scores = answered["judge_total"].dropna()
        correctness = answered["judge_correctness"].dropna()
        completeness = answered["judge_completeness"].dropna()
        groundedness = answered["judge_groundedness"].dropna()

        # Numeric consistency — only rows where metric applies
        num_applicable = df["numeric_consistency"].dropna()

        # Cost
        answer_cost = df["estimated_cost_usd"].sum()
        judge_cost = df["judge_cost_usd"].sum()

        # Cost per correct answer
        correct_answers = (answered["judge_total"] >= threshold).sum()
        cost_per_correct = round(answer_cost / correct_answers, 5) if correct_answers > 0 else None

        rows.append(
            {
                "model": model,
                "total_rows": total_rows,
                "answered": len(answered),
                "abstentions": total_rows - len(answered),
                "abstention_rate": round((total_rows - len(answered)) / total_rows, 3),
                "edge_abstention_rate": round(len(edge_abstained) / max(len(edge_cases), 1), 3),
                "avg_judge_total": round(judge_scores.mean(), 2) if len(judge_scores) else None,
                "avg_correctness": round(correctness.mean(), 2) if len(correctness) else None,
                "avg_completeness": round(completeness.mean(), 2) if len(completeness) else None,
                "avg_groundedness": round(groundedness.mean(), 2) if len(groundedness) else None,
                "avg_numeric_consistency": (
                    round(num_applicable.mean(), 3) if len(num_applicable) else None
                ),
                f"correct_answers (>={threshold}/9)": correct_answers,
                "avg_latency_seconds": round(df["latency_seconds"].mean(), 3),
                "total_answer_cost_usd": round(answer_cost, 4),
                "total_judge_cost_usd": round(judge_cost, 4),
                "cost_per_correct_usd": cost_per_correct,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()

    all_results = {}
    for m in ANSWER_MODELS:
        all_results[m] = evaluate_model(m)

    logger.info("%s", "=" * 60)
    logger.info("Building comparison summary...")
    summary = build_summary(all_results)

    logger.info("Saved to: %s", SUMMARY_FILE)
    logger.info("\n%s", summary.T.to_string())

    elapsed = round(time.time() - start, 1)
    logger.info("Total evaluation runtime: %ss", elapsed)
