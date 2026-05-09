"""
utils/logger.py
---------------
Logging utilities for TIDAL experiments.

Provides a pre-configured loguru logger with optional Rich console output
and file sink. All experiment scripts import ``get_logger`` and use it
instead of print statements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# Default format strings
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}"
)


def setup_logger(
    log_dir: Optional[str | Path] = None,
    experiment_name: str = "tidal",
    level: str = "INFO",
    rich_console: bool = True,
) -> None:
    """Configure the global loguru logger.

    Should be called once at the start of each experiment script. Subsequent
    calls to ``get_logger()`` anywhere in the codebase will use this config.

    Parameters
    ----------
    log_dir:
        Directory to write the log file. If None, file logging is disabled.
    experiment_name:
        Used as the log file name stem: ``<log_dir>/<experiment_name>.log``.
    level:
        Minimum log level: "DEBUG" | "INFO" | "WARNING" | "ERROR".
    rich_console:
        If True, attempts to use Rich for colourised console output.
    """
    logger.remove()  # Remove default handler

    if rich_console:
        try:
            from rich.logging import RichHandler
            import logging

            logging.basicConfig(
                level=level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, markup=True)],
            )
            # Bridge loguru → stdlib so Rich picks it up
            logger.add(
                lambda msg: logging.getLogger("tidal").info(msg, stacklevel=8),
                format="{message}",
                level=level,
                colorize=False,
            )
        except ImportError:
            rich_console = False

    if not rich_console:
        logger.add(
            sys.stderr,
            format=_CONSOLE_FORMAT,
            level=level,
            colorize=True,
        )

    if log_dir is not None:
        log_path = Path(log_dir) / f"{experiment_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=_FILE_FORMAT,
            level=level,
            rotation="50 MB",
            retention="30 days",
            encoding="utf-8",
        )


def get_logger(name: str = "tidal") -> "logger":  # type: ignore[return]
    """Return a loguru logger bound to a module name.

    Usage::

        log = get_logger(__name__)
        log.info("Training started")
    """
    return logger.bind(name=name)


def log_config(cfg: object, log: Optional[object] = None) -> None:
    """Pretty-print an OmegaConf config to the logger.

    Parameters
    ----------
    cfg:
        OmegaConf DictConfig or any object with a sensible ``str()`` repr.
    log:
        Logger instance. If None, uses the module-level logger.
    """
    from omegaconf import OmegaConf

    _log = log or logger
    try:
        cfg_str = OmegaConf.to_yaml(cfg)
    except Exception:
        cfg_str = str(cfg)
    _log.info("Experiment configuration:\n" + cfg_str)
