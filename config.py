"""Configuration loader for the financial QA pipeline.

Reads config.yaml and exposes a validated AppConfig dataclass singleton.
All tunable parameters live in config.yaml; secrets stay in .env.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings: level, output directory, and log filename."""

    level: str = "INFO"
    log_dir: str = "outputs/logs"
    log_file: str = "pipeline.log"


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and sizes for benchmark construction."""

    raw_data_file: str = "data/Financial-QA-10k.csv"
    benchmark_file: str = "data/final_benchmark_50.csv"
    standard_count: int = 40
    edge_case_count: int = 10


@dataclass(frozen=True)
class ModelsConfig:
    """Model identifiers, API settings, and fallback pricing."""

    gpt4o_mini: str = "openai/gpt-4o-mini"
    gemini_25_flash_lite: str = "google/gemini-2.5-flash-lite"
    claude_sonnet_45: str = "anthropic/claude-sonnet-4-5"
    api_base_url: str = "https://openrouter.ai/api/v1"
    default_temperature: float = 0.0
    default_max_tokens: int = 512
    default_retries: int = 3
    cost_per_1m: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        }
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Answer generation pipeline settings."""

    outputs_dir: str = "outputs"
    delay_between_calls: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    """LLM-as-judge evaluation settings."""

    delay_between_calls: float = 1.5
    judge_correct_threshold: int = 7
    summary_file: str = "outputs/comparison_summary.csv"


@dataclass(frozen=True)
class BertScoreConfig:
    """BERTScore computation settings."""

    model: str = "distilbert-base-uncased"
    output_file: str = "outputs/bertscore_results.csv"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration composing all sections."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    bertscore: BertScoreConfig = field(default_factory=BertScoreConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to config.yaml. Defaults to config.yaml in project root.

    Returns:
        Populated AppConfig instance. Falls back to dataclass defaults if the
        YAML file is missing.
    """
    config_path = path or PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        return AppConfig()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        logging=LoggingConfig(**raw.get("logging", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        models=ModelsConfig(**raw.get("models", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        bertscore=BertScoreConfig(**raw.get("bertscore", {})),
    )


CONFIG = load_config()
