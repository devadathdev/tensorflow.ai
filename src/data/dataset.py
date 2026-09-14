"""Dataset loading for classification and structured vision tasks."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import get_config

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class ImageDataset:
    """Directory-based image classification dataset."""

    def __init__(self, train_dir, val_dir, test_dir=None, image_size=(224, 224), batch_size=32,
                 shuffle_buffer=1000, cache_dataset=False, prefetch_buffer=32):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir) if test_dir else None
        self.image_size = tuple(image_size)
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.cache_dataset = cache_dataset
        self.prefetch_buffer = prefetch_buffer
        self.class_names = self.discover_classes()
        self.num_classes = len(self.class_names)

    def discover_classes(self):
        if not self.train_dir.exists():
            return []
        return sorted(p.name for p in self.train_dir.iterdir() if p.is_dir())

    def count_images(self, split='train'):
        root = {'train': self.train_dir, 'val': self.val_dir, 'test': self.test_dir}.get(split)
        if root is None or not root.exists():
            return {c: 0 for c in self.class_names}
        return {c: sum(1 for p in (root / c).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS) for c in self.class_names}

    def _make(self, root, shuffle=False, augmentation_pipeline=None, augment=False):
        if root is None or not root.exists():
            return None
        ds = tf.keras.utils.image_dataset_from_directory(
            root, labels='inferred', label_mode='int', class_names=self.class_names,
            image_size=self.image_size, batch_size=self.batch_size, shuffle=shuffle,
            seed=42)
        if augmentation_pipeline is not None and augment:
            ds = ds.map(lambda x, y: (augmentation_pipeline(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache_dataset:
            ds = ds.cache()
        return ds.prefetch(self.prefetch_buffer)

    def get_datasets(self, augmentation_pipeline=None, augment_train=True):
        return (self._make(self.train_dir, True, augmentation_pipeline, augment_train),
                self._make(self.val_dir), self._make(self.test_dir))

    def get_class_weights(self):
        counts = self.count_images('train')
        total = sum(counts.values())
        if not total or not self.num_classes:
            return None
        return {i: total / (self.num_classes * max(counts[c], 1)) for i, c in enumerate(self.class_names)}

    def validate_dataset(self):
        errors, warnings = [], []
        if not self.train_dir.exists(): errors.append(f'Train directory does not exist: {self.train_dir}')
        if not self.val_dir.exists(): errors.append(f'Validation directory does not exist: {self.val_dir}')
        if not self.class_names: errors.append('No classes found in the training directory')
        if self.test_dir and not self.test_dir.exists(): warnings.append(f'Test directory does not exist: {self.test_dir}')
        return {'valid': not errors, 'errors': errors, 'warnings': warnings}

    def get_statistics(self):
        def split_stats(split):
            counts = self.count_images(split)
            return {'total': sum(counts.values()), 'per_class': counts}
        return {'num_classes': self.num_classes, 'class_names': self.class_names,
                'train': split_stats('train'), 'val': split_stats('val'),
                'test': split_stats('test') if self.test_dir else None}


def create_dataset_from_config(config=None):
    cfg = config or get_config()
    d = cfg.dataset if hasattr(cfg, 'dataset') else config.get('dataset', {})
    return ImageDataset(d.get('train_dir', 'data/train'), d.get('val_dir', 'data/val'), d.get('test_dir', 'data/test'),
                        image_size=tuple(d.get('image_size', [224, 224])), batch_size=d.get('batch_size', 32),
                        shuffle_buffer=d.get('shuffle_buffer', 1000), cache_dataset=d.get('cache_dataset', False),
                        prefetch_buffer=d.get('prefetch_buffer', 32))


class StructuredDatasetError(ValueError):
    pass


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_segmentation_dataset(image_dir, mask_dir, image_size=(256, 256), batch_size=8, num_classes=None, shuffle=False):
    """Create an image/mask segmentation dataset. Masks are integer class IDs."""
    image_paths = sorted(str(p) for p in Path(image_dir).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    pairs = []
    for image in image_paths:
        mask = Path(mask_dir) / Path(image).name
        if mask.exists(): pairs.append((image, str(mask)))
    if not pairs: raise StructuredDatasetError('No image/mask pairs found')

    def load(image_path, mask_path):
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
        image = tf.image.resize(tf.cast(image, tf.float32), image_size) / 255.0
        mask = tf.image.resize(tf.cast(mask, tf.float32), image_size, method='nearest')
        mask = tf.cast(tf.squeeze(mask, -1), tf.int32)
        if num_classes is not None:
            tf.debugging.assert_less(mask, num_classes)
        return image, mask
    ds = tf.data.Dataset.from_tensor_slices(tuple(zip(*pairs))).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(max(len(pairs), 1), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_ocr_dataset(annotation_file, image_root=None, image_size=(32, 256), batch_size=16, charset=None, max_length=32, shuffle=False):
    """Create CRNN/CTC dataset from JSON/JSONL annotations: [{image, text}, ...]."""
    path = Path(annotation_file)
    rows = []
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = _load_json(path)
    if isinstance(rows, dict): rows = rows.get('annotations', rows.get('items', []))
    root = Path(image_root) if image_root else path.parent
    charset = list(charset or sorted(set(''.join(str(r['text']) for r in rows))))
    lookup = {ch: i + 1 for i, ch in enumerate(charset)}  # 0 is CTC blank
    if any(len(str(r['text'])) > max_length for r in rows):
        raise StructuredDatasetError(f'OCR label exceeds max_length={max_length}')
    images, labels = [], []
    for r in rows:
        images.append(str(root / r['image']))
        encoded = [lookup[ch] for ch in str(r['text'])]
        labels.append(encoded + [-1] * (max_length - len(encoded)))

    def load(image_path, label):
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
        image = tf.image.resize(tf.cast(image, tf.float32), image_size) / 255.0
        return image, tf.cast(label, tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(max(len(images), 1), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), charset
