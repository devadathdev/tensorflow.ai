"""Segmentation models for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any, List
from src.config import get_config


class SegmentationModel(tf.keras.Model):
    """Semantic segmentation model with U-Net/DeepLab-style architecture."""
    
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
    
    SUPPORTED_DECODERS = ['unet', 'deeplabv3plus', 'fpn']
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'mobilenetv2',
        decoder: str = 'unet',
        input_shape: tuple = (256, 256, 3),
        backbone_trainable: bool = False,
        output_stride: int = 16,
        atrous_rates: List[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.decoder_type = decoder.lower()
        self.input_shape = input_shape
        self.backbone_trainable = backbone_trainable
        self.output_stride = output_stride
        self.atrous_rates = atrous_rates or [6, 12, 18]
        self.config = config or get_config().model
        
        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Available: {list(self.SUPPORTED_BACKBONES.keys())}")
        if self.decoder_type not in self.SUPPORTED_DECODERS:
            raise ValueError(f"Unsupported decoder: {decoder}. Available: {self.SUPPORTED_DECODERS}")
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build segmentation model."""
        backbone_class = self.SUPPORTED_BACKBONES[self.backbone_name]
        
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        # Backbone with appropriate output stride
        backbone = backbone_class(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        backbone.trainable = self.backbone_trainable
        
        # Get encoder features at different scales
        encoder_features = self._get_encoder_features(backbone)
        
        # Build decoder
        if self.decoder_type == 'unet':
            x = self._build_unet_decoder(encoder_features)
        elif self.decoder_type == 'deeplabv3plus':
            x = self._build_deeplabv3plus_decoder(backbone, encoder_features)
        elif self.decoder_type == 'fpn':
            x = self._build_fpn_decoder(encoder_features)
        
        # Final classification layer
        outputs = tf.keras.layers.Conv2D(
            self.num_classes, 1, padding='same', name='segmentation_output'
        )(x)
        
        # Upsample to input resolution if needed
        if outputs.shape[1] != self.input_shape[0] or outputs.shape[2] != self.input_shape[1]:
            outputs = tf.keras.layers.UpSampling2D(
                size=(self.input_shape[0] // outputs.shape[1], self.input_shape[1] // outputs.shape[2]),
                interpolation='bilinear',
                name='final_upsample'
            )(outputs)
        
        # Softmax for multi-class segmentation
        outputs = tf.keras.layers.Activation('softmax', name='segmentation_probs')(outputs)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'SEG_{self.backbone_name}_{self.decoder_type}')
        self.backbone = backbone
    
    def _get_encoder_features(self, backbone: tf.keras.Model) -> Dict[str, tf.keras.layers.Layer]:
        """Extract encoder features at different scales."""
        layer_names = [layer.name for layer in backbone.layers]
        features = {}
        
        if 'mobilenetv2' in self.backbone_name:
            features['low'] = backbone.get_layer('block_3_expand_relu').output    # 1/8
            features['mid'] = backbone.get_layer('block_6_expand_relu').output    # 1/16
            features['high'] = backbone.get_layer('block_13_expand_relu').output  # 1/32
        elif 'mobilenetv3' in self.backbone_name:
            features['low'] = backbone.get_layer('expanded_conv_5').output
            features['mid'] = backbone.get_layer('expanded_conv_10').output
            features['high'] = backbone.get_layer('expanded_conv_15').output
        elif 'efficientnet' in self.backbone_name:
            add_layers = [n for n in layer_names if 'add' in n.lower()]
            features['low'] = backbone.get_layer(add_layers[1]).output if len(add_layers) > 1 else backbone.get_layer(add_layers[0]).output
            features['mid'] = backbone.get_layer(add_layers[len(add_layers)//2]).output
            features['high'] = backbone.get_layer(add_layers[-1]).output
        elif 'resnet' in self.backbone_name:
            features['low'] = backbone.get_layer('conv2_block3_out').output   # 1/4
            features['mid'] = backbone.get_layer('conv3_block4_out').output   # 1/8
            features['high'] = backbone.get_layer('conv4_block6_out').output  # 1/16
            if len(backbone.layers) > 150:
                features['very_high'] = backbone.get_layer('conv5_block3_out').output  # 1/32
        else:
            # Generic: use spatial reductions
            conv_layers = [n for n in layer_names if 'conv' in n.lower() and 'bn' not in n.lower()]
            if len(conv_layers) >= 3:
                features['low'] = backbone.get_layer(conv_layers[len(conv_layers)//3]).output
                features['mid'] = backbone.get_layer(conv_layers[2*len(conv_layers)//3]).output
                features['high'] = backbone.get_layer(conv_layers[-1]).output
        
        return features
    
    def _build_unet_decoder(self, encoder_features: Dict) -> tf.Tensor:
        """Build U-Net style decoder with skip connections."""
        x = encoder_features['high']
        
        # Decoder blocks with skip connections
        for i, (name, feature) in enumerate(reversed(list(encoder_features.items()))):
            if i == 0:
                continue  # Skip the highest level (already at bottleneck)
            
            # Upsample
            x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', name=f'upsample_{i}')(x)
            
            # Concatenate with encoder feature
            x = tf.keras.layers.Concatenate(name=f'concat_{i}')([x, feature])
            
            # Convolution block
            x = self._conv_block(x, 256 // (2**i), name=f'decoder_block_{i}')
        
        # Final upsample to low resolution
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', name='upsample_final')(x)
        x = self._conv_block(x, 64, name='decoder_final')
        
        return x
    
    def _build_deeplabv3plus_decoder(self, backbone: tf.keras.Model, encoder_features: Dict) -> tf.Tensor:
        """Build DeepLabV3+ decoder with ASPP."""
        # ASPP on high-level features
        x = encoder_features['high']
        x = self._aspp_module(x, name='aspp')
        
        # Decoder: upsample and fuse with low-level features
        x = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear', name='aspp_upsample')(x)
        
        # Low-level feature projection
        low_level = encoder_features['low']
        low_level = tf.keras.layers.Conv2D(48, 1, padding='same', name='low_level_proj')(low_level)
        low_level = tf.keras.layers.BatchNormalization(name='low_level_bn')(low_level)
        low_level = tf.keras.layers.Activation('relu', name='low_level_relu')(low_level)
        
        # Concatenate and process
        x = tf.keras.layers.Concatenate(name='decoder_concat')([x, low_level])
        x = self._conv_block(x, 256, name='decoder_conv1')
        x = self._conv_block(x, 256, name='decoder_conv2')
        
        return x
    
    def _build_fpn_decoder(self, encoder_features: Dict) -> tf.Tensor:
        """Build Feature Pyramid Network decoder."""
        # Lateral connections
        laterals = {}
        for i, (name, feature) in enumerate(encoder_features.items()):
            laterals[name] = tf.keras.layers.Conv2D(256, 1, padding='same', name=f'lateral_{name}')(feature)
        
        # Top-down pathway
        outputs = {}
        prev = None
        for name in reversed(list(encoder_features.keys())):
            if prev is not None:
                prev = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', name=f'fpn_upsample_{name}')(prev)
                outputs[name] = tf.keras.layers.Add(name=f'fpn_add_{name}')([laterals[name], prev])
            else:
                outputs[name] = laterals[name]
            prev = outputs[name]
        
        # Use the lowest resolution output
        return outputs['low']
    
    def _aspp_module(self, x: tf.Tensor, name: str) -> tf.Tensor:
        """Atrous Spatial Pyramid Pooling module."""
        # 1x1 conv
        aspp_1 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False, name=f'{name}_1x1')(x)
        aspp_1 = tf.keras.layers.BatchNormalization(name=f'{name}_1x1_bn')(aspp_1)
        aspp_1 = tf.keras.layers.Activation('relu', name=f'{name}_1x1_relu')(aspp_1)
        
        # Atrous convolutions
        aspp_atrous = []
        for i, rate in enumerate(self.atrous_rates):
            aspp = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=rate, use_bias=False, name=f'{name}_atrous_{rate}')(x)
            aspp = tf.keras.layers.BatchNormalization(name=f'{name}_atrous_{rate}_bn')(aspp)
            aspp = tf.keras.layers.Activation('relu', name=f'{name}_atrous_{rate}_relu')(aspp)
            aspp_atrous.append(aspp)
        
        # Image pooling
        aspp_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_pool')(x)
        aspp_pool = tf.keras.layers.Reshape((1, 1, x.shape[-1]), name=f'{name}_pool_reshape')(aspp_pool)
        aspp_pool = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False, name=f'{name}_pool_conv')(aspp_pool)
        aspp_pool = tf.keras.layers.BatchNormalization(name=f'{name}_pool_bn')(aspp_pool)
        aspp_pool = tf.keras.layers.Activation('relu', name=f'{name}_pool_relu')(aspp_pool)
        aspp_pool = tf.keras.layers.UpSampling2D(
            size=(x.shape[1], x.shape[2]), interpolation='bilinear', name=f'{name}_pool_upsample'
        )(aspp_pool)
        
        # Concatenate all
        x = tf.keras.layers.Concatenate(name=f'{name}_concat')([aspp_1] + aspp_atrous + [aspp_pool])
        x = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False, name=f'{name}_project')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_project_bn')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_project_relu')(x)
        
        return x
    
    def _conv_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_relu1')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_relu2')(x)
        return x
    
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
    
    def get_config(self) -> Dict:
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'decoder': self.decoder_type,
            'input_shape': self.input_shape,
            'backbone_trainable': self.backbone_trainable,
            'output_stride': self.output_stride,
            'atrous_rates': self.atrous_rates,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'SegmentationModel':
        return cls(**config)


def create_segmentation_model(
    num_classes: int,
    config: Optional[Dict] = None
) -> SegmentationModel:
    """Factory function to create SegmentationModel from config."""
    cfg = config or get_config().model
    seg_config = cfg.get('segmentation', {})
    
    return SegmentationModel(
        num_classes=num_classes,
        backbone=cfg.get('architecture', 'mobilenetv2'),
        decoder=seg_config.get('decoder', 'unet'),
        input_shape=tuple(cfg.get('input_shape', [256, 256, 3])),
        backbone_trainable=seg_config.get('backbone_trainable', False),
        output_stride=seg_config.get('output_stride', 16),
        atrous_rates=seg_config.get('atrous_rates', [6, 12, 18]),
        config=config
    )