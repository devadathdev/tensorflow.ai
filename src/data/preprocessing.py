"""Image preprocessing utilities."""

from typing import Tuple
import tensorflow as tf


class PreprocessingPipeline:
    """Consistent image preprocessing for training and inference."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224), rescale: float = 1.0 / 255.0):
        self.image_size = tuple(image_size)
        self.rescale = rescale

    def preprocess(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.image_size, method='bilinear')
        if self.rescale is not None:
            image = image * self.rescale
        return tf.clip_by_value(image, 0.0, 1.0)

    def preprocess_batch(self, images: tf.Tensor, labels=None):
        images = self.preprocess(images)
        return (images, labels) if labels is not None else images

    def get_preprocessing_layer(self):
        return tf.keras.Sequential([
            tf.keras.layers.Resizing(*self.image_size),
            tf.keras.layers.Rescaling(self.rescale),
        ], name='preprocessing')
