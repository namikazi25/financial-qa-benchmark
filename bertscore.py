"""
Computes BERTScore for both models using saved prediction files.
No API calls. Runs entirely locally.

BERTScore measures semantic similarity between predicted and reference answers
using contextual BERT embeddings. Used here as model-independent corroboration
of the LLM-as-judge scores.

Reads from:
    outputs/evaluation_gpt4o_mini.jsonl
    outputs/evaluation_gemini_25_flash_lite.jsonl

Prints per-model averages and saves scores to:
    outputs/bertscore_results.csv

Install:
    pip install bert-score

Run:
    python bertscore.py
"""

import json
from pathlib import Path

import pandas as pd
from bert_score import score as bert_score


OUTPUTS_DIR = Path("outputs")

EVAL_FILES = {
    "gpt4o_mini":          OUTPUTS_DIR / "evaluation_gpt4o_mini.jsonl",
    "gemini_25_flash_lite": OUTPUTS_DIR / "evaluation_gemini_25_flash_lite.jsonl",
}

OUTPUT_FILE = OUTPUTS_DIR / "bertscore_results.csv"

# Use distilbert for speed — good enough for comparison purposes
# Switch to "roberta-large" for higher quality scores if you have time
BERT_MODEL = "distilbert-base-uncased"


def load_eval(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_bertscore(records: list[dict], model_name: str) -> list[dict]:
    """
    Compute BERTScore for answered rows only.
    Abstentions are skipped and assigned None.
    """
    # Split into answered and abstained
    answered = [(i, r) for i, r in enumerate(records) if not r["abstain"] and r["answer"]]

    if not answered:
        print(f"  No answered rows found for {model_name}")
        return records

    indices  = [i for i, _ in answered]
    rows     = [r for _, r in answered]
    preds    = [str(r["answer"]) for r in rows]
    refs     = [str(r["reference_answer"]) for r in rows]

    print(f"  Running BERTScore on {len(preds)} answered rows...")

    P, R, F1 = bert_score(
        preds, refs,
        model_type=BERT_MODEL,
        lang="en",
        verbose=False,
    )

    # Write scores back into records
    results = [r.copy() for r in records]
    for idx, (orig_idx, _) in enumerate(answered):
        results[orig_idx]["bertscore_precision"] = round(P[idx].item(), 4)
        results[orig_idx]["bertscore_recall"]    = round(R[idx].item(), 4)
        results[orig_idx]["bertscore_f1"]        = round(F1[idx].item(), 4)

    # Abstained rows get None
    for i, r in enumerate(results):
        if r.get("abstain") or not r.get("answer"):
            results[i].setdefault("bertscore_precision", None)
            results[i].setdefault("bertscore_recall",    None)
            results[i].setdefault("bertscore_f1",        None)

    return results


def summarise(results: list[dict], model_name: str) -> dict:
    f1_scores = [r["bertscore_f1"] for r in results if r.get("bertscore_f1") is not None]
    p_scores  = [r["bertscore_precision"] for r in results if r.get("bertscore_precision") is not None]
    r_scores  = [r["bertscore_recall"] for r in results if r.get("bertscore_recall") is not None]

    avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None
    avg_p  = round(sum(p_scores)  / len(p_scores),  4) if p_scores  else None
    avg_r  = round(sum(r_scores)  / len(r_scores),  4) if r_scores  else None

    print(f"\n  {model_name}")
    print(f"    BERTScore F1        : {avg_f1}")
    print(f"    BERTScore Precision : {avg_p}")
    print(f"    BERTScore Recall    : {avg_r}")
    print(f"    Rows scored         : {len(f1_scores)} / {len(results)}")

    return {
        "model":               model_name,
        "bertscore_f1":        avg_f1,
        "bertscore_precision": avg_p,
        "bertscore_recall":    avg_r,
        "rows_scored":         len(f1_scores),
        "rows_abstained":      len(results) - len(f1_scores),
    }


if __name__ == "__main__":
    all_summaries = []
    all_rows      = []

    for model_name, eval_path in EVAL_FILES.items():
        if not eval_path.exists():
            print(f"File not found: {eval_path} — run evaluate.py first")
            continue

        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"{'=' * 50}")

        records = load_eval(eval_path)
        scored  = compute_bertscore(records, model_name)
        summary = summarise(scored, model_name)

        all_summaries.append(summary)

        for r in scored:
            all_rows.append({
                "model":               model_name,
                "sample_id":           r["sample_id"],
                "subset_type":         r.get("subset_type"),
                "question_type":       r.get("question_type"),
                "abstain":             r["abstain"],
                "judge_total":         r.get("judge_total"),
                "bertscore_f1":        r.get("bertscore_f1"),
                "bertscore_precision": r.get("bertscore_precision"),
                "bertscore_recall":    r.get("bertscore_recall"),
            })

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Print comparison
    print(f"\n{'=' * 50}")
    print("BERTSCORE COMPARISON")
    print(f"{'=' * 50}")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))

    # Quick check: does BERTScore agree with judge ranking?
    if len(all_summaries) == 2:
        f1_scores = [(s["model"], s["bertscore_f1"]) for s in all_summaries if s["bertscore_f1"]]
        if len(f1_scores) == 2:
            winner = max(f1_scores, key=lambda x: x[1])
            print(f"\nBERTScore winner: {winner[0]}")
            print("Check: does this match the LLM judge ranking?")