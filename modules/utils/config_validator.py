"""config_validator.py — Minimal config field extractor with strict validation."""

from __future__ import annotations

import logging
import traceback

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and validate a YAML config file. Raises on any failure."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s\n%s", config_path, traceback.format_exc())
        raise
    except yaml.YAMLError:
        logger.error("Invalid YAML in: %s\n%s", config_path, traceback.format_exc())
        raise

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or not a mapping: {config_path}")

    return config


def require_field(config: dict, *keys: str, config_file: str = "config.yml"):
    """Extract a nested field from config. Raises ValueError if missing/None.

    Usage:
        model_id = require_field(config, "model", "model_id")
        batch_size = require_field(config, "model", "training", "per_device_train_batch_size")
    """
    path = ".".join(keys)
    current = config

    try:
        for key in keys:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Expected dict at '{path}', got {type(current).__name__}"
                )
            if key not in current:
                raise KeyError(f"Key '{key}' not found")
            current = current[key]
    except (KeyError, TypeError) as e:
        msg = (
            f"CONFIG ERROR [{config_file}] -> {path}\n"
            f"  Error: {e}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        logger.error(msg)
        raise ValueError(msg) from e

    if current is None:
        msg = (
            f"CONFIG ERROR [{config_file}] -> {path}\n"
            f"  Field is None (empty). A value is required.\n"
        )
        logger.error(msg)
        raise ValueError(msg)

    return current
