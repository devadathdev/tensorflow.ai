#!/usr/bin/env python3
"""TensorFlow Lite conversion script for TensorVision AI."""

import os
import sys
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from src.config import get_config
from src.inference.engine import load_model


def convert_to_tflite(
    model_path: str,
    output_path: str,
    optimization: str = 'default',
    quantize: bool = False,
    representative_dataset: np.ndarray = None,
    input_shape: tuple = None
):
    """Convert Keras model to TensorFlow Lite format."""
    
    # Load model
    model, metadata = load_model(model_path)
    class_names = metadata.get('class_names', [f'class_{i}' for i in range(model.output_shape[-1])])
    
    print(f"Loaded model: {model_path}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Classes: {class_names}")
    print(f"Total params: {model.count_params():,}")
    print()
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization
    if optimization == 'default':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif optimization == 'size':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    elif optimization == 'latency':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif optimization == 'none':
        converter.optimizations = []
    
    # Quantization
    if quantize:
        if representative_dataset is not None:
            # Full integer quantization with representative dataset
            def representative_data_gen():
                for i in range(min(100, len(representative_dataset))):
                    yield [representative_dataset[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("Using full integer quantization (INT8)")
        else:
            # Dynamic range quantization (weights only)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("Using dynamic range quantization")
    else:
        print("No quantization (float32)")
    
    # Convert
    print("\nConverting model...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    
    # Verify
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nTFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Save metadata
    import json
    metadata_path = output_path.with_suffix('.json')
    tflite_metadata = {
        'model_version': metadata.get('version', 'unknown'),
        'class_names': class_names,
        'input_shape': input_details[0]['shape'].tolist(),
        'output_shape': output_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype']),
        'quantization': quantize,
        'optimization': optimization,
    }
    metadata_path.write_text(json.dumps(tflite_metadata, indent=2))
    print(f"Metadata saved to: {metadata_path}")
    
    return tflite_model


def test_tflite_model(tflite_path: str, test_image: np.ndarray = None):
    """Test TFLite model with sample input."""
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Generate test input if not provided
    if test_image is None:
        input_shape = input_details[0]['shape']
        if input_shape[0] == -1:  # Dynamic batch
            input_shape = [1] + input_shape[1:]
        test_image = np.random.random(input_shape).astype(input_details[0]['dtype'])
    
    # Handle quantized input
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        test_image = (test_image / scale + zero_point).astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTFLite Inference Test:")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return output


def get_representative_dataset(data_dir: str, num_samples: int = 100, image_size: tuple = (224, 224)):
    """Get representative dataset for quantization calibration."""
    
    from src.data.dataset import ImageDataset
    
    dataset = ImageDataset(
        train_dir=data_dir,
        val_dir=data_dir
    )
    
    train_ds, _, _ = dataset.get_datasets(augment_train=False)
    
    samples = []
    for images, _ in train_ds.take(num_samples):
        samples.append(images.numpy())
    
    if samples:
        return np.concatenate(samples, axis=0)[:num_samples]
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert TensorVision model to TensorFlow Lite")
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to Keras model (.keras or .h5)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output TFLite model path')
    parser.add_argument('--optimization', type=str, 
                       choices=['default', 'size', 'latency', 'none'], 
                       default='default', help='Optimization strategy')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--full-int-quant', action='store_true', 
                       help='Full integer quantization (requires --data-dir)')
    parser.add_argument('--data-dir', type=str, help='Data directory for representative dataset')
    parser.add_argument('--test', action='store_true', help='Test converted model')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    # Get representative dataset if needed
    representative_data = None
    if args.full_int_quant:
        if not args.data_dir:
            print("Error: --data-dir required for full integer quantization")
            sys.exit(1)
        image_size = tuple(config.dataset.get('image_size', [224, 224]))
        representative_data = get_representative_dataset(args.data_dir, image_size=image_size)
        if representative_data is None:
            print("Warning: Could not load representative dataset, using dynamic quantization")
    
    # Convert
    convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        optimization=args.optimization,
        quantize=args.quantize or args.full_int_quant,
        representative_dataset=representative_data,
        input_shape=tuple(config.dataset.get('image_size', [224, 224])) + (3,)
    )
    
    # Test
    if args.test:
        test_tflite_model(args.output)


if __name__ == '__main__':
    main()