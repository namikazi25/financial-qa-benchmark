from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_FILE = DATA_DIR / "Financial-QA-10k.csv"
BENCHMARK_FILE = DATA_DIR / "final_benchmark_50.csv"

# The Kaggle file uses "answer" instead of "reference_answer" in some versions
COLUMN_ALIASES = {
    "answer": "reference_answer",
    "reference": "reference_answer",
}

STANDARD_COUNT = 40
EDGE_CASE_COUNT = 10


class DatasetError(Exception):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else RAW_DATA_FILE

    if not path.exists():
        raise DatasetError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    df = _normalise_columns(df)

    missing = [c for c in ["question", "reference_answer", "context"] if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    for col in ["question", "reference_answer", "context"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if "metadata" not in df.columns:
        df["metadata"] = None

    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: COLUMN_ALIASES[col]
        for col in df.columns
        if col in COLUMN_ALIASES and COLUMN_ALIASES[col] not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # Build metadata from ticker/filing if it's not already there
    if "metadata" not in df.columns:
        ticker = df.get("ticker", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        filing = df.get("filing", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()

        df["metadata"] = [
            " | ".join(p for p in [t, f] if p) or None
            for t, f in zip(ticker, filing)
        ]

    return df


# ---------------------------------------------------------------------------
# Curation helpers — these drive benchmark selection
# ---------------------------------------------------------------------------

def detect_question_type(question: str, answer: str) -> str:
    # numeric: answer contains a number, dollar amount, percentage, etc.
    if re.search(r"\b\d[\d,.\-%$]*\b", answer):
        return "numeric"

    # list: answer has 3+ comma/and-separated items
    items = [x.strip() for x in re.split(r",| and ", answer) if x.strip()]
    if len(items) >= 3:
        return "list"

    # short fact: six words or fewer
    if len(answer.split()) <= 6:
        return "entity_or_short_fact"

    return "descriptive"


def _token_overlap(answer: str, context: str) -> float:
    def tokens(text):
        return set(re.findall(r"[A-Za-z0-9$%.-]+", text.lower()))

    a_tokens = tokens(answer)
    if not a_tokens:
        return 0.0
    return len(a_tokens & tokens(context)) / len(a_tokens)


def _looks_truncated(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    return bool(re.search(r"(\.\.\.|[\"'(\[{]|[:;,|-])$", s))


def _sparse_table(text: str) -> bool:
    if "|" not in text:
        return False
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return len(parts) <= 4 and len(text.split()) <= 12


def flag_edge_case(question: str, answer: str, context: str) -> str:
    ctx_len = len(context)
    overlap = _token_overlap(answer, context)

    if ctx_len == 0:
        return "empty_context"
    if ctx_len < 20:
        return "very_short_context"
    if _looks_truncated(context):
        return "truncated_context"
    if _sparse_table(context):
        return "sparse_table_fragment"
    if len(answer) > 120 and ctx_len < 80:
        return "answer_richer_than_context"
    if overlap < 0.15 and len(answer.split()) >= 5:
        return "low_overlap"

    return ""


# ---------------------------------------------------------------------------
# Benchmark creation
# ---------------------------------------------------------------------------

def build_curated_benchmark(
    csv_path: Optional[Path | str] = None,
    n_standard: int = STANDARD_COUNT,
    n_edge: int = EDGE_CASE_COUNT,
    save: bool = True,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Build the frozen 50-row benchmark.

    40 standard rows: clearly answerable from context, balanced across question types.
    10 edge-case rows: weak context, truncated snippets, low overlap — tests abstention.
    """
    df = load_dataset(csv_path).reset_index(drop=True)

    df["question_type"] = df.apply(
        lambda r: detect_question_type(r["question"], r["reference_answer"]), axis=1
    )
    df["edge_case_reason"] = df.apply(
        lambda r: flag_edge_case(r["question"], r["reference_answer"], r["context"]), axis=1
    )
    df["overlap"] = df.apply(
        lambda r: _token_overlap(r["reference_answer"], r["context"]), axis=1
    )

    edge_pool = df[df["edge_case_reason"] != ""].copy()
    standard_pool = df[df["edge_case_reason"] == ""].copy()

    if len(edge_pool) < n_edge:
        raise DatasetError(
            f"Only found {len(edge_pool)} edge-case rows, need {n_edge}. "
            "Try loosening the flag thresholds."
        )

    # For standard rows: sample evenly across question types so we get coverage
    standard_rows = _stratified_sample(standard_pool, n_standard)

    # For edge cases: pick the weakest contexts (lowest overlap, shortest context)
    edge_pool = edge_pool.copy()
    edge_pool["weakness_score"] = (
        (1 - edge_pool["overlap"]) * 0.6
        + (1 / (edge_pool["context"].str.len().clip(lower=1))) * 0.4
    )
    edge_rows = edge_pool.nlargest(n_edge, "weakness_score")

    benchmark = pd.concat([standard_rows, edge_rows], ignore_index=True)
    benchmark = benchmark.sample(frac=1, random_state=42).reset_index(drop=True)
    benchmark.insert(0, "sample_id", range(len(benchmark)))

    keep_cols = ["sample_id", "question", "reference_answer", "context", "metadata",
                 "subset_type", "question_type"]
    benchmark["subset_type"] = benchmark["edge_case_reason"].apply(
        lambda x: "edge_case" if x else "standard"
    )
    benchmark = benchmark[keep_cols]

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(output_path) if output_path is not None else BENCHMARK_FILE
        benchmark.to_csv(out_path, index=False)
        print(f"Saved benchmark to: {out_path}")

    return benchmark


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample n rows with roughly equal coverage across question_type values.
    Falls back to random if any type doesn't have enough rows.
    """
    types = df["question_type"].value_counts()
    per_type = max(1, n // len(types))

    sampled = []
    for qtype, count in types.items():
        take = min(per_type, count)
        sampled.append(df[df["question_type"] == qtype].sample(take, random_state=42))

    result = pd.concat(sampled, ignore_index=True)

    # Top up if we're short
    if len(result) < n:
        already_selected = result.index
        remaining = df[~df.index.isin(already_selected)]
        shortfall = n - len(result)
        if len(remaining) >= shortfall:
            result = pd.concat(
                [result, remaining.sample(shortfall, random_state=42)],
                ignore_index=True
            )

    return result.head(n)


# ---------------------------------------------------------------------------
# Loading the frozen benchmark
# ---------------------------------------------------------------------------

def load_benchmark(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the frozen benchmark file. Run build_curated_benchmark() first."""
    path = Path(csv_path) if csv_path is not None else BENCHMARK_FILE

    if not path.exists():
        raise DatasetError(
            f"Benchmark not found at: {path}\n"
            f"Run build_curated_benchmark() to create it first."
        )

    df = pd.read_csv(path)

    required = ["sample_id", "question", "reference_answer", "context"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(f"Benchmark missing columns: {missing}")

    for col in ["question", "reference_answer", "context"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Quick preview
# ---------------------------------------------------------------------------

def preview(df: pd.DataFrame, n: int = 3) -> None:
    print(f"{len(df)} rows | columns: {list(df.columns)}")
    for _, row in df.head(n).iterrows():
        print(f"\n[{row.get('sample_id', '?')}] {row['question'][:80]}")
        print(f"  answer:  {row['reference_answer'][:80]}")
        print(f"  context: {row['context'][:100]}")


if __name__ == "__main__":
    print("Building curated benchmark...")
    benchmark = build_curated_benchmark(save=True)

    print(f"\nBenchmark breakdown:")
    print(benchmark["subset_type"].value_counts().to_string())
    print()
    print(benchmark["question_type"].value_counts().to_string())

    print()
    preview(benchmark)