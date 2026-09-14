#!/usr/bin/env python3
"""Training CLI for TensorVision AI."""

import os
import sys
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.config import get_config
from src.data.dataset import create_dataset_from_config
from src.data.augmentation import create_augmentation_pipeline
from src.models.factory import create_model, get_model_info
from src.training.trainer import train_model

SUPPORTED_TASKS = ('cnn', 'transfer_learning', 'detection', 'segmentation', 'ocr')


def setup_gpu(config):
    """Configure GPU, mixed precision, and XLA settings."""
    hardware_config = config.hardware
    if hardware_config.get('use_gpu', True):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as exc:
                print(f"GPU setup error: {exc}")
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
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--model-type', choices=SUPPORTED_TASKS, help='Task/model type')
    parser.add_argument('--architecture', help='Model architecture')
    parser.add_argument('--data-dir', help='Dataset root containing train/val/test')
    parser.add_argument('--model-dir', help='Model output directory')
    parser.add_argument('--resume', help='Checkpoint/model path to resume from')
    args = parser.parse_args()

    config = get_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0: parser.error('--epochs must be greater than 0')
        config.set('training.epochs', args.epochs)
    if args.batch_size is not None:
        if args.batch_size <= 0: parser.error('--batch-size must be greater than 0')
        config.set('dataset.batch_size', args.batch_size)
    if args.lr is not None:
        if args.lr <= 0: parser.error('--lr must be greater than 0')
        config.set('training.learning_rate', args.lr)
    if args.model_type: config.set('model.type', args.model_type)
    if args.architecture: config.set('model.architecture', args.architecture)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        config.set('dataset.train_dir', str(data_dir / 'train'))
        config.set('dataset.val_dir', str(data_dir / 'val'))
        config.set('dataset.test_dir', str(data_dir / 'test'))
    if args.model_dir: config.set('output.model_dir', args.model_dir)

    setup_gpu(config)
    print('=' * 60)
    print('TensorVision AI - Training')
    print('=' * 60)
    print(f"Task: {config.get('model.type', 'cnn')}")

    dataset = create_dataset_from_config(config)
    validation = dataset.validate_dataset()
    if not validation['valid']:
        print('Dataset validation failed:')
        for error in validation['errors']: print(f'  ERROR: {error}')
        return 1
    for warning in validation['warnings']: print(f'  WARNING: {warning}')

    stats = dataset.get_statistics()
    print('\nDataset Statistics:')
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Class names: {stats['class_names']}")
    print(f"  Train images: {stats['train']['total']}")
    print(f"  Val images: {stats['val']['total']}")
    if stats['test']: print(f"  Test images: {stats['test']['total']}")

    model = create_model(stats['num_classes'], config.model)
    print(f"\nModel: {model.name}")
    model.summary()
    info = get_model_info(model)
    print(f"\nModel Info:\n  Total params: {info['total_params']:,}\n  Trainable params: {info['trainable_params']:,}")

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = tf.keras.models.load_model(args.resume)

    augmentation = create_augmentation_pipeline(config.augmentation)
    print('\nStarting training...')
    train_model(
        model=model,
        dataset=dataset,
        config=config.training,
        augmentation_pipeline=augmentation.get_layer() if augmentation.enabled else None,
    )
    print('\nTraining complete!')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
