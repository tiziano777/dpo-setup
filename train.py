"""train.py — DPO training entry point.

Loads prepared data from .cache/, validates configuration,
and runs a pre-flight check before invoking DPOTrainer.
Designed to catch config/package issues early.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from modules.utils.config_validator import load_config, require_field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_cache(path: Path) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from cache: %s", len(records), path)
    return Dataset.from_list(records)


def preflight(config_path: str) -> dict:
    """Validate everything before training starts."""
    config = load_config(config_path)

    # Model fields
    model_id = require_field(config, "model", "model_id", config_file=config_path)
    cache_dir = require_field(config, "model", "cache_dir", config_file=config_path)
    cache_file = require_field(config, "model", "cache_file", config_file=config_path)
    require_field(config, "model", "torch_dtype", config_file=config_path)
    require_field(config, "model", "device_map", config_file=config_path)

    # Training fields
    require_field(config, "model", "training", "per_device_train_batch_size", config_file=config_path)
    require_field(config, "model", "training", "gradient_accumulation_steps", config_file=config_path)
    require_field(config, "model", "training", "num_train_epochs", config_file=config_path)
    require_field(config, "model", "training", "logging_steps", config_file=config_path)
    require_field(config, "model", "training", "bf16", config_file=config_path)
    require_field(config, "model", "training", "remove_unused_columns", config_file=config_path)

    # Output
    require_field(config, "output", "output_dir", config_file=config_path)

    # Cache existence
    cache_path = Path(cache_dir) / cache_file
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}. Run prepare.py first.")

    # CUDA info
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device: %s (%d GB)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory // (1024**3))

    # Dataset validation
    ds = load_cache(cache_path)
    required_cols = {"prompt", "chosen", "rejected"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    logger.info("Dataset OK: %d rows, columns: %s", len(ds), ds.column_names)

    return {"model_id": model_id, "config": config, "dataset": ds}


def train(config_path: str = "config.yml", dry_run: bool = False):
    ctx = preflight(config_path)

    if dry_run:
        logger.info("Dry run complete — all checks passed.")
        return

    config = ctx["config"]
    model_id = ctx["model_id"]
    ds = ctx["dataset"]

    output_dir = require_field(config, "output", "output_dir", config_file=config_path)
    torch_dtype_str = require_field(config, "model", "torch_dtype", config_file=config_path)
    device_map = require_field(config, "model", "device_map", config_file=config_path)

    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, torch_dtype_str),
        device_map=device_map,
    )

    training_cfg = require_field(config, "model", "training", config_file=config_path)
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        logging_steps=training_cfg["logging_steps"],
        bf16=training_cfg["bf16"],
        remove_unused_columns=training_cfg["remove_unused_columns"],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("Training complete. Saving to %s", output_dir)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    cfg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "config.yml"
    train(cfg, dry_run=dry)
