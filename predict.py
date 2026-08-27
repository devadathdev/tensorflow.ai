#!/usr/bin/env python3
"""Inference script for TensorVision AI."""

import os
import sys
import argparse
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.config import get_config
from src.inference.predictor import ImagePredictor, predict_image


def main():
    parser = argparse.ArgumentParser(description="Run inference with TensorVision AI")
    parser.add_argument('input', type=str, help='Input image file or directory')
    parser.add_argument('--model', '-m', type=str, help='Model path (default: latest in models/)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Top K predictions')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmark')
    parser.add_argument('--runs', type=int, default=100, help='Benchmark runs')
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
    
    try:
        if args.model:
            predictor = ImagePredictor(args.model)
        else:
            predictor = ImagePredictor.from_latest_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    model_info = predictor.get_model_info()
    print(f"Model: {model_info['model_version']}")
    print(f"Classes: {model_info['num_classes']}")
    print(f"Input shape: {model_info['input_shape']}")
    print()
    
    if args.benchmark:
        if input_path.is_file():
            print(f"Benchmarking on {input_path} ({args.runs} runs)...")
            results = predictor.benchmark(input_path, num_runs=args.runs)
            print(f"\nBenchmark Results:")
            print(f"  Mean:    {results['mean_ms']:.2f} ms")
            print(f"  Std:     {results['std_ms']:.2f} ms")
            print(f"  Min:     {results['min_ms']:.2f} ms")
            print(f"  Max:     {results['max_ms']:.2f} ms")
            print(f"  Median:  {results['median_ms']:.2f} ms")
            print(f"  P95:     {results['p95_ms']:.2f} ms")
            print(f"  P99:     {results['p99_ms']:.2f} ms")
            print(f"  Throughput: {results['throughput_fps']:.2f} FPS")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
        else:
            print("Benchmark requires a single image file")
        return
    
    if input_path.is_file():
        result = predictor.predict(input_path, top_k=args.top_k, confidence_threshold=args.threshold)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model_version']}")
        
        if result.get('top_k'):
            print(f"\nTop {args.top_k} predictions:")
            for i, pred in enumerate(result['top_k'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    else:
        print(f"Processing directory: {input_path}")
        results = predictor.predict_directory(input_path)
        
        for result in results:
            if 'error' in result:
                print(f"  {result['file_name']}: ERROR - {result['error']}")
            else:
                print(f"  {result['file_name']}: {result['prediction']} ({result['confidence']:.4f})")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()