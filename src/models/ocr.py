"""OCR (Optical Character Recognition) models for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any, List
from src.config import get_config


class OCRModel(tf.keras.Model):
    """CRNN recognizer with CTC-compatible output."""
    SUPPORTED_BACKBONES={'mobilenetv2':tf.keras.applications.MobileNetV2,'mobilenetv3_small':tf.keras.applications.MobileNetV3Small,'mobilenetv3_large':tf.keras.applications.MobileNetV3Large,'efficientnetb0':tf.keras.applications.EfficientNetB0,'efficientnetb1':tf.keras.applications.EfficientNetB1,'resnet34':'custom'}
    def __init__(self,num_classes:int,backbone='mobilenetv3_small',input_shape=(32,256,3),backbone_trainable=False,rnn_layers=2,rnn_units=256,rnn_type='lstm',bidirectional=True,ctc_merge_repeated=True,config=None):
        super().__init__(); self.num_classes=num_classes; self.backbone_name=backbone.lower(); self.input_shape=input_shape; self.backbone_trainable=backbone_trainable; self.rnn_layers=rnn_layers; self.rnn_units=rnn_units; self.rnn_type=rnn_type.lower(); self.bidirectional=bidirectional; self.ctc_merge_repeated=ctc_merge_repeated; self.config=config or get_config().model
        if self.backbone_name not in self.SUPPORTED_BACKBONES: raise ValueError(f'Unsupported backbone: {backbone}. Available: {list(self.SUPPORTED_BACKBONES)}')
        self._build_model()
    def _build_model(self):
        inputs=tf.keras.Input(shape=self.input_shape,name='input_image'); x=self._build_resnet34_backbone(inputs) if self.backbone_name=='resnet34' else self._build_pretrained_backbone(inputs)
        x=tf.keras.layers.Permute((2,1,3),name='permute')(x); x=tf.keras.layers.Reshape((-1,x.shape[2]*x.shape[3]),name='reshape')(x)
        for i in range(self.rnn_layers):
            seq=i<self.rnn_layers-1; r=tf.keras.layers.LSTM(self.rnn_units,return_sequences=seq,dropout=.2,recurrent_dropout=.2,name=f'lstm_{i}') if self.rnn_type=='lstm' else tf.keras.layers.GRU(self.rnn_units,return_sequences=seq,dropout=.2,recurrent_dropout=.2,name=f'gru_{i}')
            x=tf.keras.layers.Bidirectional(r,name=f'bidirectional_{i}')(x) if self.bidirectional else r(x)
        x=tf.keras.layers.Dense(self.num_classes,name='character_logits')(x); self.model=tf.keras.Model(inputs,tf.keras.layers.Activation('softmax',name='character_probs')(x),name=f'OCR_{self.backbone_name}')
    def _build_pretrained_backbone(self,inputs):
        b=self.SUPPORTED_BACKBONES[self.backbone_name](include_top=False,weights='imagenet',input_tensor=inputs); b.trainable=self.backbone_trainable
        if 'mobilenetv2' in self.backbone_name:return b.get_layer('block_13_expand_relu').output
        if 'mobilenetv3' in self.backbone_name:return b.get_layer('expanded_conv_15').output
        if 'efficientnet' in self.backbone_name:
            adds=[l.name for l in b.layers if 'add' in l.name.lower()]; return b.get_layer(adds[-1]).output
        return b.output
    def _build_resnet34_backbone(self,inputs):
        def block(x,f,s=1,n=''):
            sc=x
            if s!=1 or x.shape[-1]!=f: sc=tf.keras.layers.BatchNormalization(name=f'{n}_scbn')(tf.keras.layers.Conv2D(f,1,strides=s,padding='same',name=f'{n}_sc')(sc))
            x=tf.keras.layers.Activation('relu')(tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(f,3,strides=s,padding='same',use_bias=False)(x))); x=tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(f,3,padding='same',use_bias=False)(x)); return tf.keras.layers.Activation('relu')(tf.keras.layers.Add()([x,sc]))
        x=tf.keras.layers.MaxPooling2D(3,2,padding='same')(tf.keras.layers.Activation('relu')(tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(64,7,strides=2,padding='same',use_bias=False)(inputs))))
        for i in range(3): x=block(x,64,n=f'l1_{i}')
        x=block(x,128,2,'l2_0');
        for i in range(3): x=block(x,128,n=f'l2_{i+1}')
        x=block(x,256,2,'l3_0');
        for i in range(5): x=block(x,256,n=f'l3_{i+1}')
        x=block(x,512,2,'l4_0');
        for i in range(2): x=block(x,512,n=f'l4_{i+1}')
        return x
    def call(self,inputs,training=None,mask=None): return self.model(inputs,training=training)
    def ctc_loss(self,y_true,y_pred):
        input_len=tf.fill([tf.shape(y_pred)[0]],tf.shape(y_pred)[1]); label_len=tf.reduce_sum(tf.cast(tf.not_equal(y_true,0),tf.int32),axis=1)
        return tf.keras.backend.ctc_batch_cost(y_true,y_pred,tf.cast(input_len,tf.float32),tf.cast(label_len,tf.float32))
    def decode_predictions(self,predictions,greedy=True,beam_width=10):
        return tf.keras.backend.ctc_decode(predictions,input_length=tf.fill([tf.shape(predictions)[0]],tf.shape(predictions)[1]),greedy=greedy,beam_width=beam_width)[0]
    def get_config(self): return {'num_classes':self.num_classes,'backbone':self.backbone_name,'input_shape':self.input_shape,'backbone_trainable':self.backbone_trainable,'rnn_layers':self.rnn_layers,'rnn_units':self.rnn_units,'rnn_type':self.rnn_type,'bidirectional':self.bidirectional,'ctc_merge_repeated':self.ctc_merge_repeated}
    @classmethod
    def from_config(cls,config): return cls(**config)

class OCRDetector(tf.keras.Model):
    """DBNet-style text detector placeholder retained for future joint OCR training."""
    def __init__(self,input_shape=(512,512,3),backbone='mobilenetv3_small',config=None):
        super().__init__(); self.input_shape=input_shape; self.backbone_name=backbone; self.config=config or get_config().model; inputs=tf.keras.Input(shape=input_shape); b=tf.keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_tensor=inputs); x=b.output; p=tf.keras.layers.Conv2D(1,1,activation='sigmoid')(tf.keras.layers.UpSampling2D(4)(x)); self.model=tf.keras.Model(inputs,{'probability_map':p,'threshold_map':p,'binary_map':p},name='OCR_Detector')
    def call(self,inputs,training=None,mask=None): return self.model(inputs,training=training)

def create_ocr_model(num_classes,config=None):
    cfg=config or get_config().model; o=cfg.get('ocr',{}); return OCRModel(num_classes=num_classes,backbone=cfg.get('architecture','mobilenetv3_small'),input_shape=tuple(cfg.get('input_shape',[32,256,3])),backbone_trainable=o.get('backbone_trainable',False),rnn_layers=o.get('rnn_layers',2),rnn_units=o.get('rnn_units',256),rnn_type=o.get('rnn_type','lstm'),bidirectional=o.get('bidirectional',True),ctc_merge_repeated=o.get('ctc_merge_repeated',True),config=config)
def create_ocr_detector(config=None):
    cfg=config or get_config().model; return OCRDetector(input_shape=tuple(cfg.get('input_shape',[512,512,3])),backbone=cfg.get('ocr',{}).get('detector_backbone','mobilenetv3_small'),config=config)
