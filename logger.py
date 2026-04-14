"""Shared logging configuration for the financial QA pipeline.

Provides console and file logging with a consistent format across all modules.
Import ``get_logger`` and call it with ``__name__`` to get a module-scoped logger.
"""

import logging

from config import CONFIG, PROJECT_ROOT


def setup_logging(level: str, log_dir: str, log_file: str) -> None:
    """Configure the root logger with console and file handlers.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files, relative to project root.
        log_file: Log filename.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers on repeated imports
    if root.handlers:
        return

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)


setup_logging(
    level=CONFIG.logging.level,
    log_dir=CONFIG.logging.log_dir,
    log_file=CONFIG.logging.log_file,
)
