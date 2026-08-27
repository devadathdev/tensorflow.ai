"""Custom CNN model for TensorVision AI."""

import tensorflow as tf
from typing import List, Optional, Dict, Any
from src.config import get_config


class CustomCNN(tf.keras.Model):
    """Customizable CNN for image classification."""
    
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (224, 224, 3),
        conv_blocks: Optional[List[Dict]] = None,
        dense_units: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        l2_regularization: float = 1e-4,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.config = config or get_config().model
        
        custom_cnn_config = self.config.get('custom_cnn', {})
        self.conv_blocks = conv_blocks or custom_cnn_config.get('conv_blocks', [
            {'filters': 32, 'kernel_size': 3, 'pool_size': 2},
            {'filters': 64, 'kernel_size': 3, 'pool_size': 2},
            {'filters': 128, 'kernel_size': 3, 'pool_size': 2},
            {'filters': 256, 'kernel_size': 3, 'pool_size': 2},
        ])
        self.dense_units = dense_units or custom_cnn_config.get('dense_units', [512, 256])
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the CNN architecture."""
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        x = inputs
        
        for i, block in enumerate(self.conv_blocks):
            x = self._conv_block(x, block, f'conv_block_{i+1}')
        
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        for i, units in enumerate(self.dense_units):
            x = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
                name=f'dense_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = tf.keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
    
    def _conv_block(self, x: tf.Tensor, block: Dict, name: str) -> tf.Tensor:
        """Create a convolutional block."""
        filters = block.get('filters', 32)
        kernel_size = block.get('kernel_size', 3)
        pool_size = block.get('pool_size', 2)
        
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name=f'{name}_conv'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(x)
        
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name=f'{name}_conv2'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        if pool_size and pool_size > 1:
            x = tf.keras.layers.MaxPooling2D(pool_size, name=f'{name}_pool')(x)
        
        return x
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
    
    def get_config(self) -> Dict:
        """Return model configuration for serialization."""
        return {
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'conv_blocks': self.conv_blocks,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_regularization,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'CustomCNN':
        """Create model from configuration."""
        return cls(**config)
    
    def summary(self, *args, **kwargs):
        """Print model summary."""
        return self.model.summary(*args, **kwargs)


def create_custom_cnn(num_classes: int, config: Optional[Dict] = None) -> CustomCNN:
    """Factory function to create CustomCNN from config."""
    cfg = config or get_config().model
    return CustomCNN(
        num_classes=num_classes,
        input_shape=tuple(cfg.get('input_shape', [224, 224, 3])),
        dropout_rate=cfg.get('dropout_rate', 0.3),
        l2_regularization=cfg.get('l2_regularization', 1e-4),
        config=config
    )