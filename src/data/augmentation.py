"""Configurable image augmentation."""

import tensorflow as tf


class AugmentationPipeline:
    """Keras augmentation stack driven by configuration."""

    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        seed = self.config.get('seed', 42)
        layers = []
        if self.config.get('horizontal_flip', True):
            layers.append(tf.keras.layers.RandomFlip('horizontal', seed=seed))
        if self.config.get('vertical_flip', False):
            layers.append(tf.keras.layers.RandomFlip('vertical', seed=seed))
        if self.config.get('rotation_factor', 0):
            layers.append(tf.keras.layers.RandomRotation(self.config['rotation_factor'], seed=seed))
        if self.config.get('zoom_factor', 0):
            layers.append(tf.keras.layers.RandomZoom(self.config['zoom_factor'], seed=seed))
        if self.config.get('width_shift_factor', 0) or self.config.get('height_shift_factor', 0):
            layers.append(tf.keras.layers.RandomTranslation(
                self.config.get('height_shift_factor', 0),
                self.config.get('width_shift_factor', 0), seed=seed))
        if self.config.get('brightness_factor', 0):
            layers.append(tf.keras.layers.RandomBrightness(self.config['brightness_factor'], seed=seed))
        if self.config.get('contrast_factor', 0):
            layers.append(tf.keras.layers.RandomContrast(self.config['contrast_factor'], seed=seed))
        self.layer = tf.keras.Sequential(layers, name='augmentation') if layers else tf.keras.layers.Identity()

    def __call__(self, images, training=True):
        if not self.enabled:
            return images
        return self.layer(images, training=training)

    def get_layer(self):
        return self.layer


def create_augmentation_pipeline(config=None):
    from src.config import get_config
    return AugmentationPipeline(config or get_config().augmentation)
