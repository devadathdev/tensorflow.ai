#!/usr/bin/env python3
"""DEVA CPU-first LLM training CLI. Existing TensorVision train.py is unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Enforce the requested CPU-only run before importing torch/transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml
from src.deva.data import validate_jsonl


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DEVA causal LLM adapters")
    parser.add_argument("--config", default="configs/deva/cpu-0.6b.yaml")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-data", action="store_true")
    actions.add_argument("--evaluate-baseline", action="store_true")
    actions.add_argument("--train", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--export", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.validate_data:
        result = {
            "train": validate_jsonl(config["data"]["train"]),
            "validation": validate_jsonl(config["data"]["validation"]),
        }
        print(json.dumps(result, indent=2))
        return 0

    from src.deva.trainer import train_lora, evaluate_model, export_merged
    if args.evaluate_baseline:
        result = evaluate_model(config, adapter=False)
    elif args.train:
        result = train_lora(config)
    elif args.evaluate:
        result = evaluate_model(config, adapter=True)
    else:
        result = {"merged_model": export_merged(config)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
