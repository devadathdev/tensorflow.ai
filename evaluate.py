#!/usr/bin/env python3
"""Evaluation script for TensorVision AI."""

import os
import sys
import argparse
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.config import get_config
from src.data.dataset import create_dataset_from_config
from src.inference.engine import create_inference_engine
from src.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate TensorVision AI model")
    parser.add_argument('--model', '-m', type=str, help='Model path (default: latest in models/)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', help='Dataset split')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Top K accuracy')
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    dataset = create_dataset_from_config()
    stats = dataset.get_statistics()
    class_names = stats['class_names']
    
    try:
        if args.model:
            engine = create_inference_engine(args.model, class_names)
        else:
            engine = create_inference_engine(
                str(sorted(Path(config.output.get('model_dir', 'models')).glob('*'))[-1]),
                class_names
            )
    except (FileNotFoundError, IndexError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Model: {engine.model_version}")
    print(f"Classes: {engine.num_classes}")
    print()
    
    if args.split == 'val':
        _, val_ds, _ = dataset.get_datasets()
        eval_dataset = val_ds
    else:
        _, _, test_ds = dataset.get_datasets()
        if test_ds is None:
            print("Test dataset not available")
            sys.exit(1)
        eval_dataset = test_ds
    
    evaluator = ModelEvaluator(engine.model, class_names)
    evaluator.config['top_k'] = args.top_k
    
    print(f"Evaluating on {args.split} set...")
    results = evaluator.generate_report(eval_dataset, output_path=args.output)
    
    print(f"\nEvaluation complete!")
    if args.output:
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()