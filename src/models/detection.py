"""Object detection models for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any, List, Tuple
from src.config import get_config


class ObjectDetectionModel(tf.keras.Model):
    """Object detection model with configurable backbones."""
    
    SUPPORTED_BACKBONES = {
        'mobilenetv2': tf.keras.applications.MobileNetV2,
        'mobilenetv3_small': tf.keras.applications.MobileNetV3Small,
        'mobilenetv3_large': tf.keras.applications.MobileNetV3Large,
        'efficientnetb0': tf.keras.applications.EfficientNetB0,
        'efficientnetb1': tf.keras.applications.EfficientNetB1,
        'efficientnetb2': tf.keras.applications.EfficientNetB2,
        'efficientnetb3': tf.keras.applications.EfficientNetB3,
        'resnet50': tf.keras.applications.ResNet50,
        'resnet101': tf.keras.applications.ResNet101,
        'resnet50v2': tf.keras.applications.ResNet50V2,
    }
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'mobilenetv2',
        input_shape: tuple = (320, 320, 3),
        backbone_trainable: bool = False,
        num_anchors: int = 9,
        max_detections: int = 100,
        nms_iou_threshold: float = 0.5,
        score_threshold: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.input_shape = input_shape
        self.backbone_trainable = backbone_trainable
        self.num_anchors = num_anchors
        self.max_detections = max_detections
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.config = config or get_config().model
        
        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Available: {list(self.SUPPORTED_BACKBONES.keys())}")
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build object detection model (SSD-style)."""
        backbone_class = self.SUPPORTED_BACKBONES[self.backbone_name]
        
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        # Backbone
        backbone = backbone_class(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        backbone.trainable = self.backbone_trainable
        
        # Feature pyramid - extract features at multiple scales
        # Get intermediate layer outputs for multi-scale detection
        feature_layers = self._get_feature_layers(backbone)
        features = [backbone.get_layer(name).output for name in feature_layers]
        
        # Detection heads for each feature level
        box_outputs = []
        class_outputs = []
        
        for i, feature in enumerate(features):
            # Box regression head
            x_box = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name=f'box_head_{i}_conv1')(feature)
            x_box = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name=f'box_head_{i}_conv2')(x_box)
            box_pred = tf.keras.layers.Conv2D(
                self.num_anchors * 4, 3, padding='same', name=f'box_pred_{i}'
            )(x_box)
            box_outputs.append(box_pred)
            
            # Class prediction head
            x_class = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name=f'class_head_{i}_conv1')(feature)
            x_class = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name=f'class_head_{i}_conv2')(x_class)
            class_pred = tf.keras.layers.Conv2D(
                self.num_anchors * self.num_classes, 3, padding='same', name=f'class_pred_{i}'
            )(x_class)
            class_outputs.append(class_pred)
        
        # Reshape outputs
        box_outputs_reshaped = []
        class_outputs_reshaped = []
        
        for box, cls in zip(box_outputs, class_outputs):
            box_outputs_reshaped.append(tf.keras.layers.Reshape((-1, 4), name=f'box_reshape')(box))
            class_outputs_reshaped.append(tf.keras.layers.Reshape((-1, self.num_classes), name=f'class_reshape')(cls))
        
        boxes = tf.keras.layers.Concatenate(axis=1, name='boxes')(box_outputs_reshaped)
        classes = tf.keras.layers.Concatenate(axis=1, name='classes')(class_outputs_reshaped)
        classes = tf.keras.layers.Activation('sigmoid', name='class_probs')(classes)
        
        # Apply NMS in post-processing
        self.model = tf.keras.Model(inputs=inputs, outputs=[boxes, classes], name=f'OD_{self.backbone_name}')
        self.backbone = backbone
    
    def _get_feature_layers(self, backbone: tf.keras.Model) -> List[str]:
        """Get feature layer names for the backbone."""
        layer_names = [layer.name for layer in backbone.layers]
        
        if 'mobilenetv2' in self.backbone_name:
            return ['block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu']
        elif 'mobilenetv3' in self.backbone_name:
            return ['expanded_conv_15', 'expanded_conv_10', 'expanded_conv_5']
        elif 'efficientnet' in self.backbone_name:
            # EfficientNet feature layers
            return [n for n in layer_names if 'add' in n.lower()][-3:]
        elif 'resnet' in self.backbone_name:
            return ['conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out']
        else:
            # Default: use last 3 convolutional blocks
            conv_layers = [n for n in layer_names if 'conv' in n.lower() and 'bn' not in n.lower()]
            return conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
    
    def get_config(self) -> Dict:
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'input_shape': self.input_shape,
            'backbone_trainable': self.backbone_trainable,
            'num_anchors': self.num_anchors,
            'max_detections': self.max_detections,
            'nms_iou_threshold': self.nms_iou_threshold,
            'score_threshold': self.score_threshold,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ObjectDetectionModel':
        return cls(**config)


def create_object_detection_model(
    num_classes: int,
    config: Optional[Dict] = None
) -> ObjectDetectionModel:
    """Factory function to create ObjectDetectionModel from config."""
    cfg = config or get_config().model
    det_config = cfg.get('detection', {})
    
    return ObjectDetectionModel(
        num_classes=num_classes,
        backbone=cfg.get('architecture', 'mobilenetv2'),
        input_shape=tuple(cfg.get('input_shape', [320, 320, 3])),
        backbone_trainable=det_config.get('backbone_trainable', False),
        num_anchors=det_config.get('num_anchors', 9),
        max_detections=det_config.get('max_detections', 100),
        nms_iou_threshold=det_config.get('nms_iou_threshold', 0.5),
        score_threshold=det_config.get('score_threshold', 0.3),
        config=config
    )