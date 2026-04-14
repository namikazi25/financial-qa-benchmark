"""Dataset loading and benchmark construction for the financial QA pipeline.

Builds a frozen 50-row benchmark from the raw Financial-QA-10k dataset with
stratified sampling (40 standard rows) and heuristic edge-case selection
(10 rows with weak or insufficient context).
"""

import re
from pathlib import Path

import pandas as pd

from config import CONFIG, PROJECT_ROOT
from logger import get_logger

logger = get_logger(__name__)

_cfg = CONFIG.dataset
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_FILE = PROJECT_ROOT / _cfg.raw_data_file
BENCHMARK_FILE = PROJECT_ROOT / _cfg.benchmark_file
STANDARD_COUNT = _cfg.standard_count
EDGE_CASE_COUNT = _cfg.edge_case_count

# The Kaggle file uses "answer" instead of "reference_answer" in some versions
COLUMN_ALIASES = {
    "answer": "reference_answer",
    "reference": "reference_answer",
}

# Heuristic thresholds for question-type detection and edge-case flagging
_ENTITY_MAX_WORDS = 6
_LIST_MIN_ITEMS = 3
_LOW_OVERLAP_THRESHOLD = 0.15
_VERY_SHORT_CONTEXT_LEN = 20
_SHORT_CONTEXT_FOR_RICH_ANSWER = 80
_RICH_ANSWER_MIN_LEN = 120
_WEAKNESS_OVERLAP_WEIGHT = 0.6
_WEAKNESS_CONTEXT_WEIGHT = 0.4


class DatasetError(Exception):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the raw Financial-QA-10k CSV.

    Args:
        csv_path: Path to the raw CSV. Defaults to the configured raw_data_file.

    Returns:
        Cleaned DataFrame with question, reference_answer, context, and metadata.

    Raises:
        DatasetError: If the file is missing or lacks required columns.
    """
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
    """Rename known column aliases and build metadata from ticker/filing fields."""
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
            " | ".join(p for p in [t, f] if p) or None for t, f in zip(ticker, filing)
        ]

    return df


# ---------------------------------------------------------------------------
# Curation helpers
# ---------------------------------------------------------------------------


def detect_question_type(question: str, answer: str) -> str:
    """Classify a question as numeric, list, entity_or_short_fact, or descriptive.

    Args:
        question: The question text.
        answer: The reference answer text.

    Returns:
        One of "numeric", "list", "entity_or_short_fact", or "descriptive".
    """
    # numeric: answer contains a number, dollar amount, percentage, etc.
    if re.search(r"\b\d[\d,.\-%$]*\b", answer):
        return "numeric"

    # list: answer has 3+ comma/and-separated items
    items = [x.strip() for x in re.split(r",| and ", answer) if x.strip()]
    if len(items) >= _LIST_MIN_ITEMS:
        return "list"

    # short fact: six words or fewer
    if len(answer.split()) <= _ENTITY_MAX_WORDS:
        return "entity_or_short_fact"

    return "descriptive"


def _token_overlap(answer: str, context: str) -> float:
    """Compute fraction of answer tokens that appear in the context."""

    def tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9$%.-]+", text.lower()))

    a_tokens = tokens(answer)
    if not a_tokens:
        return 0.0
    return len(a_tokens & tokens(context)) / len(a_tokens)


def _looks_truncated(text: str) -> bool:
    """Check whether text appears to be truncated or cut off mid-sentence."""
    s = text.strip()
    if not s:
        return True
    return bool(re.search(r"(\.\.\.|[\"'(\[{]|[:;,|-])$", s))


def _sparse_table(text: str) -> bool:
    """Detect sparse pipe-delimited table fragments with minimal content."""
    if "|" not in text:
        return False
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return len(parts) <= 4 and len(text.split()) <= 12


def flag_edge_case(question: str, answer: str, context: str) -> str:
    """Flag a row as an edge case based on context quality heuristics.

    Args:
        question: The question text.
        answer: The reference answer text.
        context: The source context from the filing.

    Returns:
        Edge case reason string, or empty string if not an edge case.
    """
    ctx_len = len(context)
    overlap = _token_overlap(answer, context)

    if ctx_len == 0:
        return "empty_context"
    if ctx_len < _VERY_SHORT_CONTEXT_LEN:
        return "very_short_context"
    if _looks_truncated(context):
        return "truncated_context"
    if _sparse_table(context):
        return "sparse_table_fragment"
    if len(answer) > _RICH_ANSWER_MIN_LEN and ctx_len < _SHORT_CONTEXT_FOR_RICH_ANSWER:
        return "answer_richer_than_context"
    if overlap < _LOW_OVERLAP_THRESHOLD and len(answer.split()) >= 5:
        return "low_overlap"

    return ""


# ---------------------------------------------------------------------------
# Benchmark creation
# ---------------------------------------------------------------------------


def build_curated_benchmark(
    csv_path: Path | str | None = None,
    n_standard: int = STANDARD_COUNT,
    n_edge: int = EDGE_CASE_COUNT,
    save: bool = True,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build the frozen 50-row benchmark.

    Selection is heuristic-based, not manually labelled:
    - 40 standard rows: clearly answerable from context, stratified by question type
    - 10 edge-case rows: weak context flagged by overlap, length, and truncation heuristics

    Args:
        csv_path: Path to raw CSV. Defaults to configured raw_data_file.
        n_standard: Number of standard rows to include.
        n_edge: Number of edge-case rows to include.
        save: Whether to write the benchmark to disk.
        output_path: Output CSV path. Defaults to configured benchmark_file.

    Returns:
        The curated benchmark DataFrame.
    """
    df = load_dataset(csv_path).reset_index(drop=True)

    df["question_type"] = df.apply(
        lambda r: detect_question_type(r["question"], r["reference_answer"]), axis=1
    )
    df["edge_case_reason"] = df.apply(
        lambda r: flag_edge_case(r["question"], r["reference_answer"], r["context"]), axis=1
    )
    df["overlap"] = df.apply(lambda r: _token_overlap(r["reference_answer"], r["context"]), axis=1)

    edge_pool = df[df["edge_case_reason"] != ""].copy()
    standard_pool = df[df["edge_case_reason"] == ""].copy()

    if len(edge_pool) < n_edge:
        raise DatasetError(
            f"Only found {len(edge_pool)} edge-case rows, need {n_edge}. "
            "Try loosening the flag thresholds."
        )

    standard_rows = _stratified_sample(standard_pool, n_standard)

    # Pick edge cases with weakest contexts (lowest overlap, shortest context)
    edge_pool = edge_pool.copy()
    edge_pool["weakness_score"] = (1 - edge_pool["overlap"]) * _WEAKNESS_OVERLAP_WEIGHT + (
        1 / (edge_pool["context"].str.len().clip(lower=1))
    ) * _WEAKNESS_CONTEXT_WEIGHT
    edge_rows = edge_pool.nlargest(n_edge, "weakness_score")

    benchmark = pd.concat([standard_rows, edge_rows], ignore_index=True)
    benchmark = benchmark.sample(frac=1, random_state=42).reset_index(drop=True)
    benchmark.insert(0, "sample_id", range(len(benchmark)))

    benchmark["subset_type"] = benchmark["edge_case_reason"].apply(
        lambda x: "edge_case" if x else "standard"
    )

    keep_cols = [
        "sample_id",
        "question",
        "reference_answer",
        "context",
        "metadata",
        "subset_type",
        "question_type",
    ]
    benchmark = benchmark[keep_cols]

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(output_path) if output_path is not None else BENCHMARK_FILE
        benchmark.to_csv(out_path, index=False)
        logger.info("Saved benchmark to: %s", out_path)

    return benchmark


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows with roughly equal coverage across question_type values.

    Falls back to random sampling to top up if any type doesn't have enough rows.

    Args:
        df: DataFrame with a question_type column.
        n: Target number of rows.

    Returns:
        Sampled DataFrame with up to n rows.
    """
    types = df["question_type"].value_counts()
    per_type = max(1, n // len(types))

    sampled: list[pd.DataFrame] = []
    selected_indices: list[int] = []

    for qtype, count in types.items():
        take = min(per_type, count)
        rows = df[df["question_type"] == qtype].sample(take, random_state=42)
        sampled.append(rows)
        selected_indices.extend(rows.index.tolist())

    result = pd.concat(sampled, ignore_index=True)

    # Top up if short — use original indices to avoid resampling already selected rows
    if len(result) < n:
        remaining = df[~df.index.isin(selected_indices)]
        shortfall = n - len(result)
        if len(remaining) >= shortfall:
            result = pd.concat(
                [result, remaining.sample(shortfall, random_state=42)],
                ignore_index=True,
            )

    return result.head(n)


# ---------------------------------------------------------------------------
# Loading the frozen benchmark
# ---------------------------------------------------------------------------


def load_benchmark(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the frozen benchmark file.

    Args:
        csv_path: Path to the benchmark CSV. Defaults to configured benchmark_file.

    Returns:
        Validated benchmark DataFrame.

    Raises:
        DatasetError: If the file is missing or lacks required columns.
    """
    path = Path(csv_path) if csv_path is not None else BENCHMARK_FILE

    if not path.exists():
        raise DatasetError(
            f"Benchmark not found at: {path}\nRun build_curated_benchmark() to create it first."
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
    """Log a quick preview of the first n rows of a DataFrame.

    Args:
        df: DataFrame to preview.
        n: Number of rows to show.
    """
    logger.info("%d rows | columns: %s", len(df), list(df.columns))
    for _, row in df.head(n).iterrows():
        logger.info("[%s] %s", row.get("sample_id", "?"), row["question"][:80])
        logger.info("  answer:  %s", row["reference_answer"][:80])
        logger.info("  context: %s", row["context"][:100])


if __name__ == "__main__":
    logger.info("Building curated benchmark...")
    benchmark = build_curated_benchmark(save=True)

    logger.info("Benchmark breakdown:")
    logger.info("\n%s", benchmark["subset_type"].value_counts().to_string())
    logger.info("\n%s", benchmark["question_type"].value_counts().to_string())

    preview(benchmark)
