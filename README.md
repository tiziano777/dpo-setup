# DPO Setup

Direct Preference Optimization (DPO) training pipeline for LLMs.

## What Is This

Data preparation → cache generation → DPO training. Recipe-driven system: define data sources, system prompts, replication strategy → generates training JSONL → feeds DPOTrainer (TRL).

Key: **modular prompt assignment** (ALL/ROUND_ROBIN/RANDOM), **pluggable chat templates**, **strict validation**.

## Quick Start

### 1. Configure

Edit `config.yml`:
- `model.model_id` — base model (e.g., `meta-llama/Llama-2-7b-hf`)
- `model.cache_dir` — where to save prepared data
- `recipe` — recipe file path
- `output.output_dir` — training checkpoint output

### 2. Prepare Data

```bash
python prepare.py
```

Pipeline:
1. Load recipe YAML → extract entries (dist_uri, chat_type, system_prompt, replica)
2. For each entry: load data → replicate → assign system prompts → apply template
3. Write `.cache/dpo_train_data.jsonl`

### 3. Train

```bash
python train.py
```

Validates config + CUDA + cache, then runs DPOTrainer.

## Data Pipeline

```
recipe.yml
    ↓
RecipeLoader → RecipeConfig
    ↓
For each entry:
  • DataLoader.load(dist_uri) → parquet/jsonl.gz/jsonl auto-detect
  • Replicate N times
  • SystemPromptAssigner → ALL/ROUND_ROBIN/RANDOM strategies
  • ChatTypeRegistry.get_template_fn() → dynamic template import
  • Template: apply_chat_template(sample, prompt, temperature)
  • Output: {prompt, chosen, rejected, _source_uri, _replica, _system_prompt_id, _id_hash}
    ↓
.cache/dpo_train_data.jsonl
    ↓
train.py → DPOTrainer
```

## Config Structure

### model
- `model_id` — HuggingFace model ID
- `cache_dir` — prep output location
- `cache_file` — JSONL filename (e.g., `dpo_train_data.jsonl`)
- `torch_dtype` — `bfloat16`, `float16`, `float32`
- `device_map` — `auto`, `cpu`, `cuda:0`
- `temperature` — used for template inference_params selection
- `templates_mapping` — path to `chat_type_mapping.yml`
- `training.*` — batch_size, gradient_accumulation, epochs, logging_steps, bf16

### recipe
Tree of entries by `dist_uri`:
- `chat_type` — template key (e.g., `train_dpo`)
- `dist_id`, `dist_name`, `dist_uri` — data source metadata
- `replica` — replication factor
- `samples`, `tokens`, `words` — stats
- `system_prompt`, `system_prompt_name` — prompt list + IDs

### output
- `output_dir` — trainer checkpoint path
- `metrics_uri` — eval metrics location

### hardware
- `gpu.type`, `gpu.count`
- `cpu.type`, `cpu.count`
- `memory.size`

## Recipe Format

YAML with top-level metadata + entries dict:

```yaml
recipe_id: v1-alpha
recipe_name: Base DPO Mix
description: Multi-domain DPO training data
scope: dpo
tasks: [instruction-following, reasoning]
tags: [v1, production]
entries:
  /path/to/dist1:
    chat_type: train_dpo
    dist_id: dist_001
    dist_name: Instruct Dataset
    dist_uri: /data/instruct/
    replica: 2
    samples: 5000
    tokens: 1000000
    words: 150000
    system_prompt: ["You are helpful assistant", "You are an AI"]
    system_prompt_name: ["helpful", "neutral"]
```

## Chat Templates

Dynamic template loading via YAML mapping.

`modules/templates/chat_type_mapping.yml`:
```yaml
train_dpo:
  template_fn: ./dpo/template_functions/instruct_dpo_apply_chat_template.py
  schema: ./dpo/input_schema_templates/input_schema.json
```

Template file must export `apply_chat_template(sample, system_prompt, temperature=0.7) → dict`.

### train_dpo Template

Input sample:
```json
{
  "_id_hash": "abc123",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is 2+2?"},
    {
      "role": "assistant",
      "positives": [
        {"inference_params": {"temperature": 0.7}, "content": "The answer is 4"},
        {"inference_params": {"temperature": 1.0}, "content": "2+2=4"}
      ],
      "negatives": [
        {"inference_params": {"temperature": 0.7}, "content": "I don't know"},
        {"inference_params": {"temperature": 1.0}, "content": "2+2=5"}
      ]
    }
  ]
}
```

Output (DPOTrainer format):
```json
{
  "prompt": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "chosen": [{"role": "assistant", "content": "The answer is 4"}],
  "rejected": [{"role": "assistant", "content": "I don't know"}],
  "_source_uri": "...",
  "_replica": 0,
  "_system_prompt_id": "helpful",
  "_id_hash": "abc123"
}
```

Selection: matches positives/negatives by `temperature` param (fallback: first item).

## Prompt Assignment Strategies

### ALL (Cartesian)
Every sample paired with every prompt. Maximum dataset size, best for diverse supervision.

### ROUND_ROBIN
Sample at index i gets prompt i % len(prompts). Balanced, reproducible.

### RANDOM
Seeded random selection (FIXED_SEED=42). Single prompt per sample.

Config in `prepare.py`: `strategy: PromptAssignmentStrategy.ALL` (default).

## Validation

Strict nested key validation via `require_field()`:
- Missing or None values → early error with full path
- Cache: required columns `[prompt, chosen, rejected]`
- Sample: required fields `[messages]` with role/content pairs

## Dependencies

- `torch>=2.4.0`
- `transformers>=4.36.0`
- `trl>=0.7.0` (DPOTrainer)
- `accelerate>=0.25.0`
- `deepspeed>=0.12.0`
- `datasets>=2.14.0`
- `pydantic>=2.0.0`
- `pandas>=2.0.0`

## File Layout

```
dpo-setup/
├── config.yml                          # Main config
├── prepare.py                          # Data prep pipeline
├── train.py                            # Training entry point
├── requirements.txt
├── modules/
│   ├── loader/data_loader.py           # Auto-detect parquet/jsonl/jsonl.gz
│   ├── recipe/
│   │   ├── recipe_loader.py            # YAML → RecipeConfig via Pydantic
│   │   └── recipe_config.py            # Pydantic models
│   ├── system_prompt/assigner.py       # ALL/ROUND_ROBIN/RANDOM
│   ├── templates/
│   │   ├── chat_type_registry.py       # Dynamic template loading
│   │   ├── chat_type_mapping.yml
│   │   └── dpo/template_functions/
│   │       └── instruct_dpo_apply_chat_template.py
│   └── utils/config_validator.py       # load_config(), require_field()
└── .cache/                             # prepare.py output
    └── dpo_train_data.jsonl
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `RecipeLoader` | `modules/recipe/` | Parse YAML → RecipeConfig |
| `ChatTypeRegistry` | `modules/templates/` | Map chat_type → template function |
| `SystemPromptAssigner` | `modules/system_prompt/` | ALL/ROUND_ROBIN/RANDOM assignment |
| `DataLoader` | `modules/loader/` | Auto-detect file format, concatenate |

## API

### RecipeLoader.load(path: str | Path) → RecipeConfig
Parse recipe YAML. Validates entries via Pydantic.

### ChatTypeRegistry(mapping_path: str | Path)
```python
registry = ChatTypeRegistry("modules/templates/chat_type_mapping.yml")
template_fn = registry.get_template_fn("train_dpo")
result = template_fn(sample, system_prompt, temperature=0.7)
```

### SystemPromptAssigner(strategy: PromptAssignmentStrategy)
```python
assigner = SystemPromptAssigner(PromptAssignmentStrategy.ROUND_ROBIN)
tuples = assigner.assign(sample, prompts, prompt_names, row_idx=0)
# → [(sample, prompt_content, prompt_id), ...]
```

### DataLoader.load(dist_uri: str) → list[dict]
Auto-detects and loads all files from directory. Priority: parquet > jsonl.gz > jsonl.

## Exit Codes

- `0` — Success
- `1` — Config error, missing cache, invalid data, CUDA issue
- `2` — Missing dependency

## Future Work

- Pluggable sampling strategies (beyond replica)
- Multi-GPU distributed preparation
- Streaming mode for very large datasets
- Checkpoint resume support
