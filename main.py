"""Answer generation pipeline for the financial QA benchmark.

Runs the frozen 50-row benchmark through both models and saves predictions to
JSONL files in the outputs directory.

Run:
    python main.py

Make sure data/final_benchmark_50.csv exists first.
If it doesn't, run dataset.py to generate it.
"""

import json
import time

from config import CONFIG, PROJECT_ROOT
from dataset import load_benchmark
from logger import get_logger
from models import ANSWER_MODELS, GEMINI_25_FLASH_LITE, GPT4O_MINI, call_model
from prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt

logger = get_logger(__name__)

_cfg = CONFIG.pipeline
OUTPUTS_DIR = PROJECT_ROOT / _cfg.outputs_dir
DELAY_BETWEEN_CALLS = _cfg.delay_between_calls

OUTPUT_FILES = {
    GPT4O_MINI: OUTPUTS_DIR / "predictions_gpt4o_mini.jsonl",
    GEMINI_25_FLASH_LITE: OUTPUTS_DIR / "predictions_gemini_25_flash_lite.jsonl",
}


def run_pipeline() -> None:
    """Run answer generation for all models on the frozen benchmark.

    Iterates over each model and each benchmark row, calls the model via
    OpenRouter, and writes one JSONL file per model to the outputs directory.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark...")
    benchmark = load_benchmark()
    logger.info("Loaded %d rows", len(benchmark))

    for model in ANSWER_MODELS:
        out_path = OUTPUT_FILES[model]
        logger.info("%s", "=" * 60)
        logger.info("Model: %s", model)
        logger.info("Output: %s", out_path)
        logger.info("%s", "=" * 60)

        results: list[dict] = []
        total_cost = 0.0
        errors = 0

        for _, row in benchmark.iterrows():
            sample_id = int(row["sample_id"])
            question = row["question"]
            context = str(row["context"]) if row["context"] else ""
            reference = row["reference_answer"]
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

            answer = parsed.get("answer", None)
            abstain = parsed.get("abstain", False)
            confidence = parsed.get("confidence", None)
            reason = parsed.get("reason", None)

            # If parsing failed entirely, treat as an error and abstain
            if not response["parsed"]:
                abstain = True
                answer = None
                confidence = None
                reason = "JSON parsing failed"
                errors += 1

            # Normalise abstain — some models return "true"/"false" as strings
            if isinstance(abstain, str):
                abstain = abstain.lower() == "true"

            # If model said abstain, answer should be null regardless
            if abstain:
                answer = None

            record = {
                # Benchmark fields
                "sample_id": sample_id,
                "question": question,
                "reference_answer": reference,
                "context": context,
                "subset_type": subset_type,
                "question_type": question_type,
                # Model output
                "model": model,
                "answer": answer,
                "abstain": abstain,
                "confidence": confidence,
                "reason": reason,
                # Operational metrics
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "latency_seconds": response["latency_seconds"],
                "estimated_cost_usd": response["estimated_cost_usd"],
                "api_error": response["error"],
            }

            results.append(record)
            total_cost += response["estimated_cost_usd"] or 0.0

            status = "ABSTAIN" if abstain else "OK"
            if response["error"]:
                status = "API_ERROR"

            logger.info(
                "  [%02d] %-10s conf=%-5s latency=%ss  cost=$%.5f",
                sample_id,
                status,
                str(confidence),
                response["latency_seconds"],
                response["estimated_cost_usd"],
            )

            time.sleep(DELAY_BETWEEN_CALLS)

        # Save to JSONL — one JSON object per line
        with open(out_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        abstain_count = sum(1 for r in results if r["abstain"])

        logger.info("Done.")
        logger.info("  Rows processed : %d", len(results))
        logger.info("  Abstentions    : %d", abstain_count)
        logger.info("  Parse errors   : %d", errors)
        logger.info("  Total cost     : $%.4f", total_cost)
        logger.info("  Saved to       : %s", out_path)


def load_predictions(model: str) -> list[dict]:
    """Load saved predictions for a given model.

    Args:
        model: Model identifier string matching a key in OUTPUT_FILES.

    Returns:
        List of prediction dicts, one per benchmark row.

    Raises:
        FileNotFoundError: If the predictions JSONL file doesn't exist.
    """
    path = OUTPUT_FILES.get(model)
    if not path or not path.exists():
        raise FileNotFoundError(f"Predictions not found at {path}. Run main.py first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    start = time.time()
    run_pipeline()
    elapsed = round(time.time() - start, 1)
    logger.info("Total runtime: %ss", elapsed)
