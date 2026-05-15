"""prepare.py — Build DPO training cache from recipe.

Pipeline:
  1. Load recipe config (entries with dist_uri, replica, system_prompt, chat_type)
  2. For each entry: load data -> replicate -> assign system prompts -> apply template
  3. Save processed dataset to .cache/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from modules.recipe.recipe_loader import RecipeLoader
from modules.loader.data_loader import DataLoader
from modules.system_prompt.assigner import SystemPromptAssigner, PromptAssignmentStrategy
from modules.templates.chat_type_registry import ChatTypeRegistry
from modules.utils.config_validator import load_config, require_field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare(
    config_path: str,
    strategy: PromptAssignmentStrategy = PromptAssignmentStrategy.ALL,
) -> Path:
    config = load_config(config_path)

    cache_dir = Path(require_field(config, "model", "cache_dir", config_file=config_path))
    cache_file = require_field(config, "model", "cache_file", config_file=config_path)
    templates_mapping = require_field(config, "model", "templates_mapping", config_file=config_path)
    temperature = require_field(config, "model", "temperature", config_file=config_path)

    recipe = RecipeLoader.load(config_path)
    logger.info("Loaded recipe: %d entries", len(recipe.entries))

    registry = ChatTypeRegistry(templates_mapping)
    assigner = SystemPromptAssigner(strategy)

    all_samples: list[dict] = []

    for uri, entry in recipe.entries.items():
        logger.info("Processing: %s (replica=%d, chat_type=%s)", uri, entry.replica, entry.chat_type)

        raw_data = DataLoader.load(entry.dist_uri)
        logger.info("  Loaded %d raw samples", len(raw_data))

        template_fn = registry.get_template_fn(entry.chat_type)
        prompts = entry.system_prompt or []
        prompt_names = entry.system_prompt_name or []

        for rep in range(entry.replica):
            for row_idx, sample in enumerate(raw_data):
                assigned = assigner.assign(sample, prompts, prompt_names, row_idx)
                for sample_copy, prompt_content, prompt_id in assigned:
                    try:
                        processed = template_fn(sample_copy, prompt_content, temperature=temperature)
                        processed["_source_uri"] = uri
                        processed["_replica"] = rep
                        processed["_system_prompt_id"] = prompt_id
                        # Include _id_hash from the original sample when available
                        processed["_id_hash"] = sample_copy.get("_id_hash", processed.get("_id_hash"))
                        all_samples.append(processed)
                    except (ValueError, KeyError) as e:
                        logger.warning("  Skipping sample: %s", e)

    cache_dir.mkdir(exist_ok=True)
    output_path = cache_dir / cache_file
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Saved %d processed samples to %s", len(all_samples), output_path)
    return output_path


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    prepare(config)
