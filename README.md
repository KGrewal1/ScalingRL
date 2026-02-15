# ScalingRL

GRPO training framework for investigating LoRA rank sensitivity across model families on math reasoning.

**Experiment grid**: 5 model families (7-9B base) x 6 LoRA ranks x AdamW = 30 experiments
**Training data**: GSM8K (openai/gsm8k)
**Evaluation**: GSM8K test (pass@1), AIME, data contamination audit

| Family  | Model ID                    |
| ------- | --------------------------- |
| qwen2.5 | `Qwen/Qwen2.5-7B`           |
| qwen3   | `Qwen/Qwen3-8B`             |
| olmo3   | `allenai/OLMo-3-1025-7B`    |
| mistral | `mistralai/Mistral-7B-v0.3` |
| gemma2  | `google/gemma-2-9b`         |

## Setup

```bash
uv sync
wandb login  # optional, or use --no-wandb
```

## Training

```bash
# Smoke test
bash smoke_test.sh

# Single run
python -m scripts.train --model-family qwen2.5 --lora-rank 8 --no-wandb

# Full sweep (30 experiments) — dry-run first
python -m scripts.sweep.py --phase phase1 --dry-run
python -m scripts.sweep.py --phase phase1

# Custom subset
python -m scripts.sweep.py --phase custom --model-families qwen2.5 mistral --lora-ranks 4 8 --dry-run
```

## Evaluation

```bash
# GSM8K pass@1
python -m scripts.evaluate.py --checkpoint ./outputs/qwen2.5_lora8 --datasets gsm8k

# AIME
python -m scripts.evaluate.py --checkpoint ./outputs/qwen2.5_lora8 --datasets aime2025

# Data contamination (completion @ 60%, per "Reasoning or Memorization?" paper)
python -m scripts.evaluate_contamination.py --model-name Qwen/Qwen2.5-7B
python -m scripts.evaluate_contamination.py --all-families
python -m scripts.evaluate_contamination.py --all-families --output-json contamination.json
```

## Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_data.py -v          # single file
uv run pytest tests/test_data.py::test_load_gsm8k_dataset -v  # single test
```

## Configuration

All defaults are in `scalingrl/config.py`. Override via CLI args — see `python scripts/train.py --help`.
