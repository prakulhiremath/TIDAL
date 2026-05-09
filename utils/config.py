"""
utils/config.py
───────────────
Configuration management for TIDAL experiments.
Supports YAML loading, CLI overrides, and nested dot-access.
"""

import yaml
import argparse
from pathlib import Path
from typing import Any, Optional
from loguru import logger


class TIDALConfig(dict):
    """
    Extended dictionary supporting dot-notation access and YAML serialization.

    Example:
        cfg = TIDALConfig.from_yaml("configs/default.yaml")
        cfg.model.hidden_size = 256
        print(cfg.training.learning_rate)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = TIDALConfig(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    @classmethod
    def from_yaml(cls, path: str) -> "TIDALConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            TIDALConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return cls(raw or {})

    def merge(self, other: dict) -> "TIDALConfig":
        """
        Deep-merge another config dict into this one.

        Args:
            other: Dictionary of overrides.

        Returns:
            Updated config.
        """
        return TIDALConfig(_deep_merge(dict(self), other))

    def override_from_args(self, args: argparse.Namespace) -> "TIDALConfig":
        """
        Apply CLI argument overrides using dot-notation keys.
        e.g. --override training.learning_rate=0.0001

        Args:
            args: Parsed argparse namespace. Expects args.override as list.

        Returns:
            Updated config.
        """
        overrides = getattr(args, "override", None) or []
        for item in overrides:
            key_path, _, value = item.partition("=")
            keys = key_path.strip().split(".")
            _set_nested(self, keys, _parse_value(value))
            logger.debug(f"Config override: {key_path} = {value}")
        return self

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False)
        logger.info(f"Config saved to {path}")

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve nested value using dot-notation.

        Args:
            key_path: Dot-separated key path e.g. "model.hidden_size".
            default: Default if key not found.

        Returns:
            Value at path or default.
        """
        keys = key_path.split(".")
        node = self
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node


def load_config(config_path: str, overrides: Optional[list] = None) -> TIDALConfig:
    """
    Convenience function: load config and apply optional overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: List of "key.path=value" strings.

    Returns:
        TIDALConfig instance.
    """
    cfg = TIDALConfig.from_yaml(config_path)
    if overrides:
        for item in overrides:
            key_path, _, value = item.partition("=")
            keys = key_path.strip().split(".")
            _set_nested(cfg, keys, _parse_value(value))
    return cfg


def get_default_arg_parser(description: str = "TIDAL Experiment") -> argparse.ArgumentParser:
    """
    Return a standardized argument parser for experiment scripts.

    Args:
        description: Parser description string.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key.path=value")
    parser.add_argument("--device", type=str, default=None,
                        help="Override compute device (cpu/cuda)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name for logging")
    return parser


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, keys: list, value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, TIDALConfig())
    d[keys[-1]] = value


def _parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    # Bool
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    # None
    if value_str.lower() == "none":
        return None
    # Int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # List (comma-separated)
    if "," in value_str:
        return [_parse_value(v.strip()) for v in value_str.split(",")]
    # String
    return value_str
