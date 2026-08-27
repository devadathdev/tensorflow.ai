#!/usr/bin/env python3
"""Training script for TensorVision AI."""

import os
import sys
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.config import get_config
from src.data.dataset import ImageDataset, create_dataset_from_config
from src.data.augmentation import create_augmentation_pipeline
from src.models.factory import create_model, compile_model, get_model_info
from src.training.trainer import Trainer, train_model
from src.training.callbacks import get_callbacks


def setup_gpu(config):
    """Configure GPU settings."""
    hardware_config = config.hardware
    
    if hardware_config.get('use_gpu', True):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found, using CPU")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled, using CPU")
    
    if hardware_config.get('mixed_precision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    
    if hardware_config.get('xla_compile', False):
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled")


def main():
    parser = argparse.ArgumentParser(description="Train TensorVision AI model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'transfer_learning'], help='Model type')
    parser.add_argument('--architecture', type=str, help='Model architecture')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--model-dir', type=str, help='Model output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('dataset.batch_size', args.batch_size)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    if args.model_type:
        config.set('model.type', args.model_type)
    if args.architecture:
        config.set('model.architecture', args.architecture)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        config.set('dataset.train_dir', str(data_dir / 'train'))
        config.set('dataset.val_dir', str(data_dir / 'val'))
        config.set('dataset.test_dir', str(data_dir / 'test'))
    if args.model_dir:
        config.set('output.model_dir', args.model_dir)
    
    setup_gpu(config)
    
    print("=" * 60)
    print("TensorVision AI - Training")
    print("=" * 60)
    
    dataset = create_dataset_from_config()
    
    validation = dataset.validate_dataset()
    if not validation['valid']:
        print("Dataset validation failed:")
        for error in validation['errors']:
            print(f"  ERROR: {error}")
        sys.exit(1)
    
    for warning in validation['warnings']:
        print(f"  WARNING: {warning}")
    
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Class names: {stats['class_names']}")
    print(f"  Train images: {stats['train']['total']}")
    print(f"  Val images: {stats['val']['total']}")
    if stats['test']:
        print(f"  Test images: {stats['test']['total']}")
    
    num_classes = stats['num_classes']
    class_names = stats['class_names']
    
    model = create_model(num_classes)
    print(f"\nModel: {model.name}")
    model.summary()
    
    model_info = get_model_info(model)
    print(f"\nModel Info:")
    print(f"  Total params: {model_info['total_params']:,}")
    print(f"  Trainable params: {model_info['trainable_params']:,}")
    
    augmentation = create_augmentation_pipeline()
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = tf.keras.models.load_model(args.resume)
    else:
        print("\nStarting training...")
        trainer = train_model(
            model=model,
            dataset=dataset,
            augmentation_pipeline=augmentation.get_layer() if augmentation.enabled else None
        )
    
    print("\nTraining complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()