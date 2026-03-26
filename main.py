"""
main.py

Runs the answer generation pipeline on the frozen 50-row benchmark.

For each row, calls both models and saves predictions to:
  outputs/predictions_gpt4o_mini.jsonl
  outputs/predictions_gemini_25_flash_lite.jsonl

Run:
    python main.py

Make sure data/final_benchmark_50.csv exists first.
If it doesn't, run dataset.py to generate it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dataset import load_benchmark
from models import ANSWER_MODELS, GPT4O_MINI, GEMINI_25_FLASH_LITE, call_model
from prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt

OUTPUTS_DIR = Path("outputs")

OUTPUT_FILES = {
    GPT4O_MINI:           OUTPUTS_DIR / "predictions_gpt4o_mini.jsonl",
    GEMINI_25_FLASH_LITE: OUTPUTS_DIR / "predictions_gemini_25_flash_lite.jsonl",
}

# Pause between API calls to avoid rate limit errors
DELAY_BETWEEN_CALLS = 1.0  # seconds


def run_pipeline() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark...")
    benchmark = load_benchmark()
    print(f"Loaded {len(benchmark)} rows\n")

    for model in ANSWER_MODELS:
        out_path = OUTPUT_FILES[model]
        print(f"{'=' * 60}")
        print(f"Model: {model}")
        print(f"Output: {out_path}")
        print(f"{'=' * 60}")

        results = []
        total_cost = 0.0
        errors = 0

        for _, row in benchmark.iterrows():
            sample_id   = int(row["sample_id"])
            question    = row["question"]
            context     = str(row["context"]) if row["context"] else ""
            reference   = row["reference_answer"]
            subset_type = row.get("subset_type", "unknown")
            question_type = row.get("question_type", "unknown")

            user_message = build_answer_prompt(question, context)

            response = call_model(
                model=model,
                system_prompt=ANSWER_SYSTEM_PROMPT,
                user_message=user_message,
            )

            # Pull structured fields from parsed JSON
            # Fall back to safe defaults if the model returned invalid JSON
            parsed = response["parsed"] or {}

            answer     = parsed.get("answer", None)
            abstain    = parsed.get("abstain", False)
            confidence = parsed.get("confidence", None)
            reason     = parsed.get("reason", None)

            # If parsing failed entirely, treat as an error and abstain
            if not response["parsed"]:
                abstain    = True
                answer     = None
                confidence = None
                reason     = "JSON parsing failed"
                errors    += 1

            # Normalise abstain — some models return "true"/"false" as strings
            if isinstance(abstain, str):
                abstain = abstain.lower() == "true"

            # If model said abstain, answer should be null regardless
            if abstain:
                answer = None

            record = {
                # Benchmark fields
                "sample_id":      sample_id,
                "question":       question,
                "reference_answer": reference,
                "context":        context,
                "subset_type":    subset_type,
                "question_type":  question_type,
                # Model output
                "model":          model,
                "answer":         answer,
                "abstain":        abstain,
                "confidence":     confidence,
                "reason":         reason,
                # Operational metrics
                "input_tokens":   response["input_tokens"],
                "output_tokens":  response["output_tokens"],
                "latency_seconds": response["latency_seconds"],
                "estimated_cost_usd": response["estimated_cost_usd"],
                "api_error":      response["error"],
            }

            results.append(record)
            total_cost += response["estimated_cost_usd"] or 0.0

            status = "ABSTAIN" if abstain else "OK"
            if response["error"]:
                status = "API_ERROR"

            print(
                f"  [{sample_id:02d}] {status:<10} "
                f"conf={str(confidence):<5} "
                f"latency={response['latency_seconds']}s  "
                f"cost=${response['estimated_cost_usd']:.5f}"
            )

            time.sleep(DELAY_BETWEEN_CALLS)

        # Save to JSONL — one JSON object per line
        with open(out_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        abstain_count = sum(1 for r in results if r["abstain"])

        print(f"\nDone.")
        print(f"  Rows processed : {len(results)}")
        print(f"  Abstentions    : {abstain_count}")
        print(f"  Parse errors   : {errors}")
        print(f"  Total cost     : ${total_cost:.4f}")
        print(f"  Saved to       : {out_path}\n")


def load_predictions(model: str) -> list[dict]:
    """Load saved predictions for a given model. Used by evaluate.py."""
    path = OUTPUT_FILES.get(model)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Predictions not found at {path}. Run main.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    start = time.time()
    run_pipeline()
    elapsed = round(time.time() - start, 1)
    print(f"Total runtime: {elapsed}s")
