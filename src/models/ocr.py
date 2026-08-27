"""OCR (Optical Character Recognition) models for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any, List, Tuple
from src.config import get_config


class OCRModel(tf.keras.Model):
    """OCR model with CRNN (CNN + RNN + CTC) architecture."""
    
    SUPPORTED_BACKBONES = {
        'mobilenetv2': tf.keras.applications.MobileNetV2,
        'mobilenetv3_small': tf.keras.applications.MobileNetV3Small,
        'mobilenetv3_large': tf.keras.applications.MobileNetV3Large,
        'efficientnetb0': tf.keras.applications.EfficientNetB0,
        'efficientnetb1': tf.keras.applications.EfficientNetB1,
        'resnet34': 'custom',  # Custom ResNet34 for OCR
    }
    
    def __init__(
        self,
        num_classes: int,  # Number of characters + 1 (blank for CTC)
        backbone: str = 'mobilenetv3_small',
        input_shape: tuple = (32, 256, 3),  # Height, Width, Channels
        backbone_trainable: bool = False,
        rnn_layers: int = 2,
        rnn_units: int = 256,
        rnn_type: str = 'lstm',  # 'lstm' or 'gru'
        bidirectional: bool = True,
        ctc_merge_repeated: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.input_shape = input_shape
        self.backbone_trainable = backbone_trainable
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.ctc_merge_repeated = ctc_merge_repeated
        self.config = config or get_config().model
        
        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Available: {list(self.SUPPORTED_BACKBONES.keys())}")
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build CRNN model for OCR."""
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        # CNN Backbone (feature extractor)
        if self.backbone_name == 'resnet34':
            x = self._build_resnet34_backbone(inputs)
        else:
            x = self._build_pretrained_backbone(inputs)
        
        # Reshape for RNN: (batch, width, features)
        # After CNN: (batch, height, width, channels) -> (batch, width, height*channels)
        x = tf.keras.layers.Permute((2, 1, 3), name='permute')(x)
        x = tf.keras.layers.Reshape((-1, x.shape[2] * x.shape[3]), name='reshape')(x)
        
        # RNN layers
        for i in range(self.rnn_layers):
            return_sequences = (i < self.rnn_layers - 1)
            
            if self.rnn_type == 'lstm':
                rnn_layer = tf.keras.layers.LSTM(
                    self.rnn_units,
                    return_sequences=return_sequences,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    name=f'lstm_{i}'
                )
            else:  # gru
                rnn_layer = tf.keras.layers.GRU(
                    self.rnn_units,
                    return_sequences=return_sequences,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    name=f'gru_{i}'
                )
            
            if self.bidirectional:
                x = tf.keras.layers.Bidirectional(rnn_layer, name=f'bidirectional_{i}')(x)
            else:
                x = rnn_layer(x)
        
        # Dense layer for character classification
        x = tf.keras.layers.Dense(
            self.num_classes,
            activation=None,  # No activation - CTC loss expects logits
            name='character_logits'
        )(x)
        
        # For inference: apply softmax
        probs = tf.keras.layers.Activation('softmax', name='character_probs')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=probs, name=f'OCR_{self.backbone_name}')
    
    def _build_pretrained_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """Build backbone from pretrained model."""
        backbone_class = self.SUPPORTED_BACKBONES[self.backbone_name]
        
        backbone = backbone_class(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        backbone.trainable = self.backbone_trainable
        
        # Remove final pooling, keep spatial dimensions
        # Get feature map before global pooling
        if 'mobilenetv2' in self.backbone_name:
            x = backbone.get_layer('block_13_expand_relu').output
        elif 'mobilenetv3' in self.backbone_name:
            x = backbone.get_layer('expanded_conv_15').output
        elif 'efficientnet' in self.backbone_name:
            add_layers = [n for n in [l.name for l in backbone.layers] if 'add' in n.lower()]
            x = backbone.get_layer(add_layers[-1]).output
        else:
            x = backbone.output
        
        return x
    
    def _build_resnet34_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """Build custom ResNet34 backbone optimized for OCR."""
        def residual_block(x, filters, stride=1, name=''):
            shortcut = x
            if stride != 1 or x.shape[-1] != filters:
                shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', name=f'{name}_shortcut')(shortcut)
                shortcut = tf.keras.layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
            
            x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, name=f'{name}_conv1')(x)
            x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
            x = tf.keras.layers.Activation('relu', name=f'{name}_relu1')(x)
            
            x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv2')(x)
            x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
            
            x = tf.keras.layers.Add(name=f'{name}_add')([x, shortcut])
            x = tf.keras.layers.Activation('relu', name=f'{name}_relu2')(x)
            return x
        
        # Initial conv
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.Activation('relu', name='relu1')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='maxpool')(x)
        
        # ResNet34 layers: [3, 4, 6, 3] blocks
        x = residual_block(x, 64, name='layer1_0')
        x = residual_block(x, 64, name='layer1_1')
        x = residual_block(x, 64, name='layer1_2')
        
        x = residual_block(x, 128, stride=2, name='layer2_0')
        for i in range(3):
            x = residual_block(x, 128, name=f'layer2_{i+1}')
        
        x = residual_block(x, 256, stride=2, name='layer3_0')
        for i in range(5):
            x = residual_block(x, 256, name=f'layer3_{i+1}')
        
        x = residual_block(x, 512, stride=2, name='layer4_0')
        for i in range(2):
            x = residual_block(x, 512, name=f'layer4_{i+1}')
        
        return x
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
    
    def ctc_loss(self, y_true, y_pred):
        """CTC loss function for OCR training."""
        # y_true: (batch, max_label_length) - sparse labels
        # y_pred: (batch, timesteps, num_classes) - logits/probs
        
        input_len = tf.ones(tf.shape(y_pred)[0]) * tf.shape(y_pred)[1]
        label_len = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1)
        
        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len
        )
        return loss
    
    def decode_predictions(self, predictions, greedy=True, beam_width=10):
        """Decode CTC predictions to text."""
        if greedy:
            decoded, _ = tf.keras.backend.ctc_decode(
                predictions,
                input_length=tf.ones(tf.shape(predictions)[0]) * tf.shape(predictions)[1],
                greedy=True
            )
        else:
            decoded, _ = tf.keras.backend.ctc_decode(
                predictions,
                input_length=tf.ones(tf.shape(predictions)[0]) * tf.shape(predictions)[1],
                greedy=False,
                beam_width=beam_width
            )
        return decoded
    
    def get_config(self) -> Dict:
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'input_shape': self.input_shape,
            'backbone_trainable': self.backbone_trainable,
            'rnn_layers': self.rnn_layers,
            'rnn_units': self.rnn_units,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'ctc_merge_repeated': self.ctc_merge_repeated,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'OCRModel':
        return cls(**config)


class OCRDetector(tf.keras.Model):
    """Text detection model (EAST-style or DBNet-style)."""
    
    def __init__(
        self,
        input_shape: tuple = (512, 512, 3),
        backbone: str = 'mobilenetv3_small',
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.input_shape = input_shape
        self.backbone_name = backbone
        self.config = config or get_config().model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build text detection model (DBNet-style)."""
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        # Backbone
        if 'mobilenetv3' in self.backbone_name:
            backbone_class = tf.keras.applications.MobileNetV3Small
        elif 'mobilenetv2' in self.backbone_name:
            backbone_class = tf.keras.applications.MobileNetV2
        else:
            backbone_class = tf.keras.applications.EfficientNetB0
        
        backbone = backbone_class(include_top=False, weights='imagenet', input_tensor=inputs)
        backbone.trainable = False
        
        # Feature pyramid
        feature_layers = ['expanded_conv_5', 'expanded_conv_10', 'expanded_conv_15'] if 'mobilenetv3' in self.backbone_name else \
                        ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu']
        
        features = [backbone.get_layer(name).output for name in feature_layers]
        
        # DBNet head
        # Probability map (text/non-text)
        prob_map = self._dbnet_head(features[-1], name='prob')
        prob_map = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear', name='prob_upsample')(prob_map)
        prob_map = tf.keras.layers.Activation('sigmoid', name='prob_map')(prob_map)
        
        # Threshold map
        thresh_map = self._dbnet_head(features[-1], name='thresh')
        thresh_map = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear', name='thresh_upsample')(thresh_map)
        thresh_map = tf.keras.layers.Activation('sigmoid', name='thresh_map')(thresh_map)
        
        # Binary map (differentiable binarization)
        binary_map = self._differentiable_binarization(prob_map, thresh_map)
        
        outputs = {
            'probability_map': prob_map,
            'threshold_map': thresh_map,
            'binary_map': binary_map
        }
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'OCR_Detector_{self.backbone_name}')
    
    def _dbnet_head(self, x, name=''):
        """DBNet detection head."""
        x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False, name=f'{name}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_relu1')(x)
        
        x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False, name=f'{name}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_relu2')(x)
        
        x = tf.keras.layers.Conv2D(1, 1, padding='same', name=f'{name}_out')(x)
        return x
    
    def _differentiable_binarization(self, prob_map, thresh_map, k=50):
        """Differentiable binarization: 1 / (1 + exp(-k * (prob - thresh)))"""
        return tf.keras.layers.Lambda(
            lambda x: 1.0 / (1.0 + tf.exp(-k * (x[0] - x[1]))),
            name='binary_map'
        )([prob_map, thresh_map])
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


def create_ocr_model(
    num_classes: int,
    config: Optional[Dict] = None
) -> OCRModel:
    """Factory function to create OCRModel from config."""
    cfg = config or get_config().model
    ocr_config = cfg.get('ocr', {})
    
    return OCRModel(
        num_classes=num_classes,
        backbone=cfg.get('architecture', 'mobilenetv3_small'),
        input_shape=tuple(cfg.get('input_shape', [32, 256, 3])),
        backbone_trainable=ocr_config.get('backbone_trainable', False),
        rnn_layers=ocr_config.get('rnn_layers', 2),
        rnn_units=ocr_config.get('rnn_units', 256),
        rnn_type=ocr_config.get('rnn_type', 'lstm'),
        bidirectional=ocr_config.get('bidirectional', True),
        ctc_merge_repeated=ocr_config.get('ctc_merge_repeated', True),
        config=config
    )


def create_ocr_detector(
    config: Optional[Dict] = None
) -> OCRDetector:
    """Factory function to create OCRDetector from config."""
    cfg = config or get_config().model
    ocr_config = cfg.get('ocr', {})
    
    return OCRDetector(
        input_shape=tuple(cfg.get('input_shape', [512, 512, 3])),
        backbone=ocr_config.get('detector_backbone', 'mobilenetv3_small'),
        config=config
    )