"""BERTScore computation for semantic similarity assessment.

Computes BERTScore for both models using saved scoring files.
No API calls -- runs entirely locally.

BERTScore measures semantic similarity between predicted and reference answers
using contextual BERT embeddings. Used here as model-independent corroboration
of the LLM-as-judge scores.

Reads from:
    outputs/evaluation_gpt4o_mini.jsonl
    outputs/evaluation_gemini_25_flash_lite.jsonl

Saves scores to:
    outputs/bertscore_results.csv

Run:
    python bertscore.py
"""

import json
from pathlib import Path

import pandas as pd
from bert_score import score as bert_score

from config import CONFIG, PROJECT_ROOT
from logger import get_logger

logger = get_logger(__name__)

_bs_cfg = CONFIG.bertscore
_pipe_cfg = CONFIG.pipeline
OUTPUTS_DIR = PROJECT_ROOT / _pipe_cfg.outputs_dir

SCORING_FILES = {
    "gpt4o_mini": OUTPUTS_DIR / "evaluation_gpt4o_mini.jsonl",
    "gemini_25_flash_lite": OUTPUTS_DIR / "evaluation_gemini_25_flash_lite.jsonl",
}

OUTPUT_FILE = PROJECT_ROOT / _bs_cfg.output_file
BERT_MODEL = _bs_cfg.model


def load_scoring_records(path: Path) -> list[dict]:
    """Load scoring JSONL records from disk.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of record dicts.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_bertscore(records: list[dict], model_name: str) -> list[dict]:
    """Compute BERTScore for answered rows only.

    Abstentions are skipped and assigned None for all BERTScore metrics.

    Args:
        records: List of record dicts.
        model_name: Display name for logging.

    Returns:
        Copy of records with bertscore_precision, bertscore_recall, and
        bertscore_f1 fields added.
    """
    # Split into answered and abstained
    answered = [(i, r) for i, r in enumerate(records) if not r["abstain"] and r["answer"]]

    if not answered:
        logger.warning("  No answered rows found for %s", model_name)
        return records

    rows = [r for _, r in answered]
    preds = [str(r["answer"]) for r in rows]
    refs = [str(r["reference_answer"]) for r in rows]

    logger.info("  Running BERTScore on %d answered rows...", len(preds))

    P, R, F1 = bert_score(
        preds,
        refs,
        model_type=BERT_MODEL,
        lang="en",
        verbose=False,
    )

    # Write scores back into records
    results = [r.copy() for r in records]
    for idx, (orig_idx, _) in enumerate(answered):
        results[orig_idx]["bertscore_precision"] = round(P[idx].item(), 4)
        results[orig_idx]["bertscore_recall"] = round(R[idx].item(), 4)
        results[orig_idx]["bertscore_f1"] = round(F1[idx].item(), 4)

    # Abstained rows get None
    for i, r in enumerate(results):
        if r.get("abstain") or not r.get("answer"):
            results[i].setdefault("bertscore_precision", None)
            results[i].setdefault("bertscore_recall", None)
            results[i].setdefault("bertscore_f1", None)

    return results


def summarise(results: list[dict], model_name: str) -> dict:
    """Compute average BERTScore metrics for a model.

    Args:
        results: List of records with BERTScore fields populated.
        model_name: Display name for logging.

    Returns:
        Summary dict with model name, average scores, and row counts.
    """
    f1_scores = [r["bertscore_f1"] for r in results if r.get("bertscore_f1") is not None]
    p_scores = [
        r["bertscore_precision"] for r in results if r.get("bertscore_precision") is not None
    ]
    r_scores = [r["bertscore_recall"] for r in results if r.get("bertscore_recall") is not None]

    avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None
    avg_p = round(sum(p_scores) / len(p_scores), 4) if p_scores else None
    avg_r = round(sum(r_scores) / len(r_scores), 4) if r_scores else None

    logger.info("  %s", model_name)
    logger.info("    BERTScore F1        : %s", avg_f1)
    logger.info("    BERTScore Precision : %s", avg_p)
    logger.info("    BERTScore Recall    : %s", avg_r)
    logger.info("    Rows scored         : %d / %d", len(f1_scores), len(results))

    return {
        "model": model_name,
        "bertscore_f1": avg_f1,
        "bertscore_precision": avg_p,
        "bertscore_recall": avg_r,
        "rows_scored": len(f1_scores),
        "rows_abstained": len(results) - len(f1_scores),
    }


if __name__ == "__main__":
    all_summaries: list[dict] = []
    all_rows: list[dict] = []

    for model_name, scoring_path in SCORING_FILES.items():
        if not scoring_path.exists():
            logger.warning("File not found: %s — run evaluate.py first", scoring_path)
            continue

        logger.info("%s", "=" * 50)
        logger.info("Model: %s", model_name)
        logger.info("%s", "=" * 50)

        records = load_scoring_records(scoring_path)
        scored = compute_bertscore(records, model_name)
        summary = summarise(scored, model_name)

        all_summaries.append(summary)

        for r in scored:
            all_rows.append(
                {
                    "model": model_name,
                    "sample_id": r["sample_id"],
                    "subset_type": r.get("subset_type"),
                    "question_type": r.get("question_type"),
                    "abstain": r["abstain"],
                    "judge_total": r.get("judge_total"),
                    "bertscore_f1": r.get("bertscore_f1"),
                    "bertscore_precision": r.get("bertscore_precision"),
                    "bertscore_recall": r.get("bertscore_recall"),
                }
            )

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved to: %s", OUTPUT_FILE)

    # Comparison
    logger.info("%s", "=" * 50)
    logger.info("BERTSCORE COMPARISON")
    logger.info("%s", "=" * 50)
    summary_df = pd.DataFrame(all_summaries)
    logger.info("\n%s", summary_df.to_string(index=False))

    # Quick check: does BERTScore agree with judge ranking?
    if len(all_summaries) == 2:
        f1_vals = [(s["model"], s["bertscore_f1"]) for s in all_summaries if s["bertscore_f1"]]
        if len(f1_vals) == 2:
            winner = max(f1_vals, key=lambda x: x[1])
            logger.info("BERTScore winner: %s", winner[0])
            logger.info("Check: does this match the LLM judge ranking?")
