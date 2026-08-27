"""Model factory for TensorVision AI."""

import tensorflow as tf
from typing import Optional, Dict, Any
from src.config import get_config
from src.models.cnn import CustomCNN, create_custom_cnn
from src.models.transfer_learning import TransferLearningModel, create_transfer_learning_model
from src.models.detection import ObjectDetectionModel, create_object_detection_model
from src.models.segmentation import SegmentationModel, create_segmentation_model
from src.models.ocr import OCRModel, OCRDetector, create_ocr_model, create_ocr_detector


def create_model(
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """Create model based on configuration.
    
    Args:
        num_classes: Number of output classes
        config: Configuration dictionary
        
    Returns:
        Keras model
    """
    cfg = config or get_config().model
    model_type = cfg.get('type', 'cnn')
    architecture = cfg.get('architecture', 'custom_cnn')
    
    if model_type == 'detection':
        return create_object_detection_model(num_classes, config)
    elif model_type == 'segmentation':
        return create_segmentation_model(num_classes, config)
    elif model_type == 'ocr':
        return create_ocr_model(num_classes, config)
    elif model_type == 'transfer_learning' or architecture in [
        'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large',
        'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
        'efficientnetv2b0', 'efficientnetv2b1', 'efficientnetv2b2', 'efficientnetv2b3',
        'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
        'inceptionv3', 'xception',
        'densenet121', 'densenet169', 'densenet201'
    ]:
        return create_transfer_learning_model(num_classes, config)
    else:
        return create_custom_cnn(num_classes, config)


def create_detection_model(
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> ObjectDetectionModel:
    """Create object detection model."""
    return create_object_detection_model(num_classes, config)


def create_segmentation_model_fn(
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> SegmentationModel:
    """Create segmentation model."""
    return create_segmentation_model(num_classes, config)


def create_ocr_model_fn(
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> OCRModel:
    """Create OCR recognition model."""
    return create_ocr_model(num_classes, config)


def create_ocr_detector_fn(
    config: Optional[Dict[str, Any]] = None
) -> OCRDetector:
    """Create OCR text detector model."""
    return create_ocr_detector(config)


def compile_model(
    model: tf.keras.Model,
    config: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """Compile model with training configuration.
    
    Args:
        model: Keras model to compile
        config: Training configuration
        
    Returns:
        Compiled model
    """
    cfg = config or get_config().training
    
    learning_rate = cfg.get('learning_rate', 1e-3)
    optimizer_name = cfg.get('optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    loss = cfg.get('loss', 'sparse_categorical_crossentropy')
    metrics = cfg.get('metrics', ['accuracy'])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def compile_detection_model(
    model: ObjectDetectionModel,
    config: Optional[Dict[str, Any]] = None
) -> ObjectDetectionModel:
    """Compile object detection model with custom losses."""
    cfg = config or get_config().training
    
    learning_rate = cfg.get('learning_rate', 1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Custom losses for detection
    def box_loss(y_true, y_pred):
        # Smooth L1 loss for box regression
        diff = tf.abs(y_true - y_pred)
        loss = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
        return tf.reduce_mean(loss)
    
    def class_loss(y_true, y_pred):
        # Focal loss for class imbalance
        alpha = 0.25
        gamma = 2.0
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow(1 - p_t, gamma) * ce
        return tf.reduce_mean(loss)
    
    model.compile(
        optimizer=optimizer,
        loss={'boxes': box_loss, 'classes': class_loss},
        loss_weights={'boxes': 1.0, 'classes': 1.0},
        metrics={'classes': ['accuracy']}
    )
    return model


def compile_segmentation_model(
    model: SegmentationModel,
    config: Optional[Dict[str, Any]] = None
) -> SegmentationModel:
    """Compile segmentation model with Dice loss."""
    cfg = config or get_config().training
    
    learning_rate = cfg.get('learning_rate', 1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1]))
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    def combined_loss(y_true, y_pred):
        ce = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return ce + dice
    
    def mean_iou(y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        return tf.keras.metrics.MeanIoU(num_classes=model.num_classes)(y_true, y_pred_classes)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['accuracy', mean_iou]
    )
    return model


def compile_ocr_model(
    model: OCRModel,
    config: Optional[Dict[str, Any]] = None
) -> OCRModel:
    """Compile OCR model with CTC loss."""
    cfg = config or get_config().training
    
    learning_rate = cfg.get('learning_rate', 1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=model.ctc_loss,
        metrics=['accuracy']
    )
    return model


def get_model_info(model: tf.keras.Model) -> Dict[str, Any]:
    """Get model information for logging/versioning."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'layers': len(model.layers),
    }