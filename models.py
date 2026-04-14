"""Unified model interface via OpenRouter.

All three models are called through one function using the OpenAI SDK
pointed at OpenRouter's base URL.

Models:
  - openai/gpt-4o-mini              (answer generation)
  - google/gemini-2.5-flash-lite    (answer generation)
  - anthropic/claude-sonnet-4-5     (judge)

GPT-4o-mini and Gemini 2.5 Flash Lite are both the cost-optimised tier
from their respective vendors -- a fair cross-vendor comparison.
Claude Sonnet 4.5 judges both to avoid self-serving bias.
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)
load_dotenv()

# ---------------------------------------------------------------------------
# Model name constants — OpenRouter format: "provider/model-name"
# ---------------------------------------------------------------------------

_cfg = CONFIG.models

GPT4O_MINI = _cfg.gpt4o_mini
GEMINI_25_FLASH_LITE = _cfg.gemini_25_flash_lite
CLAUDE_SONNET_45 = _cfg.claude_sonnet_45

ANSWER_MODELS = [GPT4O_MINI, GEMINI_25_FLASH_LITE]
JUDGE_MODEL = CLAUDE_SONNET_45

# ---------------------------------------------------------------------------
# Fallback cost table (USD per 1M tokens) — used only if OpenRouter doesn't
# return a cost in the response.
# ---------------------------------------------------------------------------

COST_PER_1M = _cfg.cost_per_1m


class ModelError(Exception):
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _get_client() -> OpenAI:
    """Create an OpenAI client pointed at the OpenRouter API.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ModelError: If the OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ModelError("OPENROUTER_API_KEY not found. Make sure it's set in your .env file.")
    return OpenAI(
        api_key=api_key,
        base_url=_cfg.api_base_url,
    )


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------


def call_model(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = _cfg.default_temperature,
    max_tokens: int = _cfg.default_max_tokens,
    retries: int = _cfg.default_retries,
) -> dict[str, Any]:
    """Call any model through OpenRouter using the OpenAI-compatible interface.

    Uses response_format=json_object to enforce valid JSON output -- more
    reliable than prompt-only instructions, especially for Gemini.

    Args:
        model: OpenRouter model string e.g. "openai/gpt-4o-mini".
        system_prompt: System prompt text.
        user_message: User message text.
        temperature: Sampling temperature. 0.0 for deterministic outputs.
        max_tokens: Max tokens to generate.
        retries: Retry attempts on transient errors (exponential backoff).

    Returns:
        Dict with keys: content, parsed, input_tokens, output_tokens,
        latency_seconds, estimated_cost_usd, model, error.
    """
    client = _get_client()

    for attempt in range(retries):
        try:
            start = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                # Enforce JSON output at the API level — not just via prompt.
                # OpenRouter passes this to providers that support it.
                response_format={"type": "json_object"},
            )

            latency = round(time.time() - start, 3)
            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            # OpenRouter returns actual cost in usage.cost when available.
            # Fall back to our own estimate if it's missing.
            cost = _get_cost(response, model, input_tokens, output_tokens)

            return {
                "content": content,
                "parsed": _safe_parse_json(content),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": latency,
                "estimated_cost_usd": cost,
                "model": model,
                "error": None,
            }

        except Exception as e:
            if attempt < retries - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "[%s] Attempt %d failed: %s. Retrying in %ds...",
                    model,
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                return {
                    "content": "",
                    "parsed": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_seconds": 0.0,
                    "estimated_cost_usd": 0.0,
                    "model": model,
                    "error": str(e),
                }

    return {}  # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cost(response: Any, model: str, input_tokens: int, output_tokens: int) -> float:
    """Extract cost from OpenRouter response, falling back to the cost table.

    Args:
        response: Raw API response object.
        model: Model identifier string.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        Estimated cost in USD.
    """
    try:
        if response.usage and hasattr(response.usage, "cost") and response.usage.cost:
            return round(float(response.usage.cost), 6)
    except Exception:
        pass
    return _estimate_cost(model, input_tokens, output_tokens)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from the fallback pricing table.

    Args:
        model: Model identifier string.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        Estimated cost in USD, or 0.0 if model is not in the cost table.
    """
    if model not in COST_PER_1M:
        return 0.0
    rates = COST_PER_1M[model]
    cost = (input_tokens / 1_000_000) * rates["input"]
    cost += (output_tokens / 1_000_000) * rates["output"]
    return round(cost, 6)


def _safe_parse_json(text: str) -> dict | None:
    """Parse JSON from model output.

    Strips markdown code fences some models add even with json_object mode on.

    Args:
        text: Raw model output text.

    Returns:
        Parsed dict, or None if parsing fails.
    """
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Quick connectivity test — run this before main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_system = (
        "You are a test assistant. "
        'Respond only with this exact JSON: {"status": "ok", "model": "your model name here"}'
    )
    test_user = "Confirm you are working."

    for m in [GPT4O_MINI, GEMINI_25_FLASH_LITE, CLAUDE_SONNET_45]:
        logger.info("Testing: %s", m)
        result = call_model(m, test_system, test_user, max_tokens=64)

        if result["error"]:
            logger.error("  ERROR: %s", result["error"])
        else:
            logger.info("  Response : %s", result["content"][:100])
            logger.info("  Parsed   : %s", result["parsed"])
            logger.info(
                "  Tokens   : %d in / %d out", result["input_tokens"], result["output_tokens"]
            )
            logger.info("  Cost     : $%s", result["estimated_cost_usd"])
            logger.info("  Latency  : %ss", result["latency_seconds"])
