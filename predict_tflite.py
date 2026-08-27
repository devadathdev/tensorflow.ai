#!/usr/bin/env python3
"""TFLite inference script for TensorVision AI."""

import os
import sys
import argparse
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from src.config import get_config
from src.inference.tflite_engine import create_tflite_engine, benchmark_tflite


def main():
    parser = argparse.ArgumentParser(description="Run TFLite inference with TensorVision AI")
    parser.add_argument('input', type=str, help='Input image file or directory')
    parser.add_argument('--model', '-m', type=str, required=True, help='TFLite model path (.tflite)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Top K predictions')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmark')
    parser.add_argument('--runs', type=int, default=100, help='Benchmark runs')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Load class names from metadata
    metadata_path = model_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        class_names = metadata.get('class_names', [f'class_{i}' for i in range(1000)])
    else:
        print("Error: No metadata.json found. Please provide class names.")
        sys.exit(1)
    
    try:
        engine = create_tflite_engine(model_path, class_names, num_threads=args.threads)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model_info = engine.get_model_info()
    print(f"Model: {model_info['model_version']}")
    print(f"Classes: {model_info['num_classes']}")
    print(f"Input shape: {model_info['input_shape']}")
    print(f"Quantized: {model_info['is_quantized']}")
    print()
    
    if args.benchmark:
        if input_path.is_file():
            print(f"Benchmarking on {input_path} ({args.runs} runs)...")
            
            # Load and preprocess image once
            from PIL import Image
            img = Image.open(input_path).convert('RGB')
            
            results = benchmark_tflite(model_path, class_names, img, num_runs=args.runs)
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
        result = engine.predict(input_path)
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
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        for img_path in image_files:
            try:
                result = engine.predict(img_path)
                print(f"  {img_path.name}: {result['prediction']} ({result['confidence']:.4f})")
            except Exception as e:
                print(f"  {img_path.name}: ERROR - {e}")


if __name__ == '__main__':
    main()