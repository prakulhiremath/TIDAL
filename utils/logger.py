"""
utils/logger.py
───────────────
Structured logging configuration for TIDAL experiments.
Uses loguru for rich console output and file logging.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: Optional[str] = None,
    experiment_name: str = "tidal",
    level: str = "INFO",
    rotation: str = "10 MB",
) -> None:
    """
    Configure loguru logger with console and optional file sinks.

    Args:
        log_dir: Directory to write log files. None = console only.
        experiment_name: Name prefix for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        rotation: File rotation policy.
    """
    # Remove default handler
    logger.remove()

    # Rich console handler
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=console_format, level=level, colorize=True)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        log_file = log_path / f"{experiment_name}.log"
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            compression="zip",
        )
        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Logger initialized [level={level}]")


def log_config(config: dict) -> None:
    """
    Pretty-print experiment configuration to log.

    Args:
        config: Configuration dictionary.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    _log_dict(config, indent=0)
    logger.info("=" * 60)


def _log_dict(d: dict, indent: int = 0) -> None:
    """Recursively log dictionary contents."""
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{key}:")
            _log_dict(value, indent + 1)
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_metrics(metrics: dict, step: Optional[int] = None, prefix: str = "") -> None:
    """
    Log a dictionary of metrics.

    Args:
        metrics: Metric name → value mapping.
        step: Optional step/epoch number.
        prefix: Prefix string (e.g. 'train', 'val').
    """
    step_str = f" [step={step}]" if step is not None else ""
    prefix_str = f"[{prefix}] " if prefix else ""
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    logger.info(f"{prefix_str}{step_str} {' | '.join(parts)}")
