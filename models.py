"""
models.py

Unified model interface via OpenRouter.
All three models are called through one function using the OpenAI SDK
pointed at OpenRouter's base URL.

Models:
  - openai/gpt-4o-mini              (answer generation)
  - google/gemini-2.5-flash-lite    (answer generation)
  - anthropic/claude-sonnet-4-5     (judge)

GPT-4o-mini and Gemini 2.5 Flash Lite are both the cost-optimised tier
from their respective vendors — a fair cross-vendor comparison.
Claude Sonnet 4.5 judges both to avoid self-serving bias.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model name constants — OpenRouter format: "provider/model-name"
# ---------------------------------------------------------------------------

GPT4O_MINI            = "openai/gpt-4o-mini"
GEMINI_25_FLASH_LITE  = "google/gemini-2.5-flash-lite"
CLAUDE_SONNET_45      = "anthropic/claude-sonnet-4-5"

ANSWER_MODELS = [GPT4O_MINI, GEMINI_25_FLASH_LITE]
JUDGE_MODEL   = CLAUDE_SONNET_45

# ---------------------------------------------------------------------------
# Fallback cost table (USD per 1M tokens) — used only if OpenRouter doesn't
# return a cost in the response. Prices from OpenRouter as of March 2026.
# ---------------------------------------------------------------------------

COST_PER_1M = {
    GPT4O_MINI:           {"input": 0.15,  "output": 0.60},
    GEMINI_25_FLASH_LITE: {"input": 0.10,  "output": 0.40},
    CLAUDE_SONNET_45:     {"input": 3.00,  "output": 15.00},
}


class ModelError(Exception):
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ModelError(
            "OPENROUTER_API_KEY not found. Make sure it's set in your .env file."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def call_model(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
) -> dict:
    """
    Call any model through OpenRouter using the OpenAI-compatible interface.

    Uses response_format=json_object to enforce valid JSON output — more
    reliable than prompt-only instructions, especially for Gemini.

    Args:
        model:         OpenRouter model string e.g. "openai/gpt-4o-mini"
        system_prompt: System prompt text
        user_message:  User message text
        temperature:   0.0 for deterministic outputs
        max_tokens:    Max tokens to generate
        retries:       Retry attempts on transient errors (exponential backoff)

    Returns:
        {
            "content":              str,         # raw response text
            "parsed":               dict | None, # parsed JSON or None if invalid
            "input_tokens":         int,
            "output_tokens":        int,
            "latency_seconds":      float,
            "estimated_cost_usd":   float,       # from OpenRouter response if available
            "model":                str,
            "error":                str | None,
        }
    """
    client = _get_client()

    for attempt in range(retries):
        try:
            start = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                # Enforce JSON output at the API level — not just via prompt.
                # OpenRouter passes this to providers that support it.
                response_format={"type": "json_object"},
            )

            latency       = round(time.time() - start, 3)
            content       = response.choices[0].message.content or ""
            input_tokens  = response.usage.prompt_tokens     if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            # OpenRouter returns actual cost in usage.cost when available.
            # Fall back to our own estimate if it's missing.
            cost = _get_cost(response, model, input_tokens, output_tokens)

            return {
                "content":            content,
                "parsed":             _safe_parse_json(content),
                "input_tokens":       input_tokens,
                "output_tokens":      output_tokens,
                "latency_seconds":    latency,
                "estimated_cost_usd": cost,
                "model":              model,
                "error":              None,
            }

        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[{model}] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {
                    "content":            "",
                    "parsed":             None,
                    "input_tokens":       0,
                    "output_tokens":      0,
                    "latency_seconds":    0.0,
                    "estimated_cost_usd": 0.0,
                    "model":              model,
                    "error":              str(e),
                }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cost(response, model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Use OpenRouter's reported cost from the response if available.
    Falls back to our cost table estimate.
    """
    try:
        if response.usage and hasattr(response.usage, "cost") and response.usage.cost:
            return round(float(response.usage.cost), 6)
    except Exception:
        pass
    return _estimate_cost(model, input_tokens, output_tokens)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in COST_PER_1M:
        return 0.0
    rates = COST_PER_1M[model]
    cost  = (input_tokens  / 1_000_000) * rates["input"]
    cost += (output_tokens / 1_000_000) * rates["output"]
    return round(cost, 6)


def _safe_parse_json(text: str) -> Optional[dict]:
    """
    Parse JSON from model output.
    Strips markdown code fences some models add even with json_object mode on.
    """
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
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
        'You are a test assistant. '
        'Respond only with this exact JSON: {"status": "ok", "model": "your model name here"}'
    )
    test_user = "Confirm you are working."

    for model in [GPT4O_MINI, GEMINI_25_FLASH_LITE, CLAUDE_SONNET_45]:
        print(f"\nTesting: {model}")
        result = call_model(model, test_system, test_user, max_tokens=64)

        if result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Response : {result['content'][:100]}")
            print(f"  Parsed   : {result['parsed']}")
            print(f"  Tokens   : {result['input_tokens']} in / {result['output_tokens']} out")
            print(f"  Cost     : ${result['estimated_cost_usd']}")
            print(f"  Latency  : {result['latency_seconds']}s")