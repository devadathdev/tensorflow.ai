"""Model factory for TensorVision AI."""
import tensorflow as tf
from typing import Optional, Dict, Any
from src.models.cnn import CustomCNN, create_custom_cnn
from src.models.transfer_learning import TransferLearningModel, create_transfer_learning_model
from src.models.detection import ObjectDetectionModel, create_object_detection_model
from src.models.segmentation import SegmentationModel, create_segmentation_model
from src.models.ocr import OCRModel, OCRDetector, create_ocr_model, create_ocr_detector

def create_model(num_classes:int,config:Optional[Dict[str,Any]]=None)->tf.keras.Model:
    cfg=config or {}; model_type=cfg.get('type','cnn'); architecture=cfg.get('architecture','custom_cnn')
    if model_type=='detection': return create_object_detection_model(num_classes,config)
    if model_type=='segmentation': return create_segmentation_model(num_classes,config)
    if model_type=='ocr': return create_ocr_model(num_classes,config)
    if model_type=='transfer_learning' or architecture in ['mobilenetv2','mobilenetv3_small','mobilenetv3_large','efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetv2b0','efficientnetv2b1','efficientnetv2b2','efficientnetv2b3','resnet50','resnet101','resnet152','resnet50v2','inceptionv3','xception','densenet121','densenet169','densenet201']: return create_transfer_learning_model(num_classes,config)
    return create_custom_cnn(num_classes,config)
def create_detection_model(num_classes,config=None): return create_object_detection_model(num_classes,config)
def create_segmentation_model_fn(num_classes,config=None): return create_segmentation_model(num_classes,config)
def create_ocr_model_fn(num_classes,config=None): return create_ocr_model(num_classes,config)
def create_ocr_detector_fn(config=None): return create_ocr_detector(config)
def _optimizer(cfg):
    lr=cfg.get('learning_rate',1e-3); name=cfg.get('optimizer','adam').lower()
    return tf.keras.optimizers.SGD(lr,momentum=.9) if name=='sgd' else tf.keras.optimizers.RMSprop(lr) if name=='rmsprop' else tf.keras.optimizers.Adam(lr)
def compile_model(model,config=None):
    cfg=config or {}; model.compile(optimizer=_optimizer(cfg),loss=cfg.get('loss','sparse_categorical_crossentropy'),metrics=cfg.get('metrics',['accuracy'])); return model
def compile_detection_model(model,config=None):
    cfg=config or {}; opt=_optimizer(cfg)
    def box_loss(y_true,y_pred):
        d=tf.abs(y_true-y_pred); return tf.reduce_mean(tf.where(d<1,.5*d*d,d-.5))
    def class_loss(y_true,y_pred):
        ce=tf.keras.backend.binary_crossentropy(y_true,y_pred); p=tf.where(tf.equal(y_true,1),y_pred,1-y_pred); return tf.reduce_mean(.25*tf.pow(1-p,2)*ce)
    model.compile(optimizer=opt,loss={'boxes':box_loss,'classes':class_loss},loss_weights={'boxes':1.,'classes':1.},metrics={'classes':['accuracy']}); return model
def compile_segmentation_model(model,config=None):
    cfg=config or {}; opt=_optimizer(cfg)
    def dice_loss(y_true,y_pred,smooth=1e-6):
        yt=tf.one_hot(tf.cast(y_true,tf.int32),model.num_classes); return 1-(2*tf.reduce_sum(yt*y_pred)+smooth)/(tf.reduce_sum(yt)+tf.reduce_sum(y_pred)+smooth)
    def combined(y_true,y_pred): return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)+dice_loss(y_true,y_pred)
    model.compile(optimizer=opt,loss=combined,metrics=['accuracy']); return model
def compile_ocr_model(model,config=None):
    model.compile(optimizer=_optimizer(config or {}),loss=model.ctc_loss); return model
def get_model_info(model):
    total=model.count_params(); trainable=sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    return {'name':model.name,'total_params':total,'trainable_params':trainable,'non_trainable_params':total-trainable,'input_shape':model.input_shape,'output_shape':model.output_shape,'layers':len(model.layers)}
