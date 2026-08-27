"""Transfer learning models for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any, List
from src.config import get_config


BACKBONE_MODELS = {
    'mobilenetv2': tf.keras.applications.MobileNetV2,
    'mobilenetv3_small': tf.keras.applications.MobileNetV3Small,
    'mobilenetv3_large': tf.keras.applications.MobileNetV3Large,
    'efficientnetb0': tf.keras.applications.EfficientNetB0,
    'efficientnetb1': tf.keras.applications.EfficientNetB1,
    'efficientnetb2': tf.keras.applications.EfficientNetB2,
    'efficientnetb3': tf.keras.applications.EfficientNetB3,
    'efficientnetv2b0': tf.keras.applications.EfficientNetV2B0,
    'efficientnetv2b1': tf.keras.applications.EfficientNetV2B1,
    'efficientnetv2b2': tf.keras.applications.EfficientNetV2B2,
    'efficientnetv2b3': tf.keras.applications.EfficientNetV2B3,
    'resnet50': tf.keras.applications.ResNet50,
    'resnet101': tf.keras.applications.ResNet101,
    'resnet152': tf.keras.applications.ResNet152,
    'resnet50v2': tf.keras.applications.ResNet50V2,
    'inceptionv3': tf.keras.applications.InceptionV3,
    'xception': tf.keras.applications.Xception,
    'densenet121': tf.keras.applications.DenseNet121,
    'densenet169': tf.keras.applications.DenseNet169,
    'densenet201': tf.keras.applications.DenseNet201,
}


class TransferLearningModel(tf.keras.Model):
    """Transfer learning model with pretrained backbones."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'efficientnetb0',
        input_shape: tuple = (224, 224, 3),
        backbone_trainable: bool = False,
        fine_tune_at: Optional[int] = None,
        pooling: str = 'avg',
        dropout_rate: float = 0.3,
        dense_units: Optional[List[int]] = None,
        l2_regularization: float = 1e-4,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.input_shape = input_shape
        self.backbone_trainable = backbone_trainable
        self.fine_tune_at = fine_tune_at
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.l2_regularization = l2_regularization
        self.config = config or get_config().model
        
        if self.backbone_name not in BACKBONE_MODELS:
            raise ValueError(f"Unsupported backbone: {backbone}. Available: {list(BACKBONE_MODELS.keys())}")
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build transfer learning model."""
        backbone_class = BACKBONE_MODELS[self.backbone_name]
        
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        backbone = backbone_class(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=self.pooling
        )
        
        backbone.trainable = self.backbone_trainable
        
        if self.fine_tune_at is not None and self.backbone_trainable:
            for layer in backbone.layers[:self.fine_tune_at]:
                layer.trainable = False
            for layer in backbone.layers[self.fine_tune_at:]:
                layer.trainable = True
        
        x = backbone.output
        
        if self.pooling is None:
            x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        
        if self.dense_units:
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
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'TL_{self.backbone_name}')
        self.backbone = backbone
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
    
    def set_backbone_trainable(self, trainable: bool, fine_tune_at: Optional[int] = None) -> None:
        """Set backbone trainability for fine-tuning."""
        self.backbone_trainable = trainable
        self.fine_tune_at = fine_tune_at
        
        self.backbone.trainable = trainable
        
        if trainable and fine_tune_at is not None:
            for layer in self.backbone.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in self.backbone.layers[fine_tune_at:]:
                layer.trainable = True
        
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
    
    def unfreeze_top_layers(self, num_layers: int = 10) -> None:
        """Unfreeze top N layers of backbone for fine-tuning."""
        if not self.backbone_trainable:
            self.backbone_trainable = True
            self.backbone.trainable = True
        
        for layer in self.backbone.layers[-num_layers:]:
            layer.trainable = True
        
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
    
    def get_config(self) -> Dict:
        """Return model configuration for serialization."""
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'input_shape': self.input_shape,
            'backbone_trainable': self.backbone_trainable,
            'fine_tune_at': self.fine_tune_at,
            'pooling': self.pooling,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'l2_regularization': self.l2_regularization,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TransferLearningModel':
        """Create model from configuration."""
        return cls(**config)
    
    def summary(self, *args, **kwargs):
        """Print model summary."""
        return self.model.summary(*args, **kwargs)


def create_transfer_learning_model(
    num_classes: int,
    config: Optional[Dict] = None
) -> TransferLearningModel:
    """Factory function to create TransferLearningModel from config."""
    cfg = config or get_config().model
    tl_config = cfg.get('transfer_learning', {})
    
    return TransferLearningModel(
        num_classes=num_classes,
        backbone=cfg.get('architecture', 'efficientnetb0'),
        input_shape=tuple(cfg.get('input_shape', [224, 224, 3])),
        backbone_trainable=tl_config.get('backbone_trainable', False),
        fine_tune_at=tl_config.get('fine_tune_at'),
        pooling=tl_config.get('pooling', 'avg'),
        dropout_rate=cfg.get('dropout_rate', 0.3),
        l2_regularization=cfg.get('l2_regularization', 1e-4),
        config=config
    )