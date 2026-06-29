"""Logging utilities for experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default centralized log directory (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


def get_log_path(stage_name: str, log_dir: Optional[Path] = None) -> Path:
    """Get the path for a stage's log file in the centralized log directory.

    Args:
        stage_name: Name of the stage (e.g., "5_clustering", "7c1_steering_baseline")
        log_dir: Optional custom log directory. If None, uses DEFAULT_LOG_DIR.

    Returns:
        Path to the log file
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stage_name}.log"


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
