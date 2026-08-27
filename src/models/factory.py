"""Model factory and task-specific compilation helpers."""

import tensorflow as tf
from typing import Optional, Dict, Any
from src.config import get_config
from src.models.cnn import CustomCNN, create_custom_cnn
from src.models.transfer_learning import TransferLearningModel, create_transfer_learning_model
from src.models.detection import ObjectDetectionModel, create_object_detection_model
from src.models.segmentation import SegmentationModel, create_segmentation_model
from src.models.ocr import OCRModel, OCRDetector, create_ocr_model, create_ocr_detector


_TRANSFER_ARCHITECTURES = {
    'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large',
    'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
    'efficientnetv2b0', 'efficientnetv2b1', 'efficientnetv2b2', 'efficientnetv2b3',
    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
    'inceptionv3', 'xception', 'densenet121', 'densenet169', 'densenet201',
}


def _model_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize either a model section or a full config dictionary."""
    if config is None:
        return get_config().model
    return config.get('model', config)


def _training_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize either a training section or a full config dictionary."""
    if config is None:
        return get_config().training
    return config.get('training', config)


def create_model(num_classes: int, config: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
    """Create a model from either a model section or a full config dictionary."""
    cfg = _model_config(config)
    model_type = cfg.get('type', 'cnn')
    architecture = cfg.get('architecture', 'custom_cnn')

    # The task type is authoritative; architecture selects the backbone within it.
    if model_type == 'detection':
        return create_object_detection_model(num_classes, cfg)
    if model_type == 'segmentation':
        return create_segmentation_model(num_classes, cfg)
    if model_type == 'ocr':
        return create_ocr_model(num_classes, cfg)
    if model_type == 'transfer_learning' or architecture in _TRANSFER_ARCHITECTURES:
        return create_transfer_learning_model(num_classes, cfg)
    if model_type != 'cnn':
        raise ValueError(f"Unsupported model type: {model_type}")
    return create_custom_cnn(num_classes, cfg)


def create_detection_model(num_classes: int, config: Optional[Dict[str, Any]] = None) -> ObjectDetectionModel:
    return create_object_detection_model(num_classes, _model_config(config))


def create_segmentation_model_fn(num_classes: int, config: Optional[Dict[str, Any]] = None) -> SegmentationModel:
    return create_segmentation_model(num_classes, _model_config(config))


def create_ocr_model_fn(num_classes: int, config: Optional[Dict[str, Any]] = None) -> OCRModel:
    return create_ocr_model(num_classes, _model_config(config))


def create_ocr_detector_fn(config: Optional[Dict[str, Any]] = None) -> OCRDetector:
    return create_ocr_detector(_model_config(config))


def _optimizer(cfg: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    name = str(cfg.get('optimizer', 'adam')).lower()
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


def compile_model(model: tf.keras.Model, config: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
    """Compile a standard classification model."""
    cfg = _training_config(config)
    model.compile(
        optimizer=_optimizer(cfg),
        loss=cfg.get('loss', 'sparse_categorical_crossentropy'),
        metrics=cfg.get('metrics', ['accuracy']),
    )
    return model


def compile_detection_model(model: ObjectDetectionModel, config: Optional[Dict[str, Any]] = None) -> ObjectDetectionModel:
    """Compile a detector using Smooth-L1 box loss and focal classification loss."""
    cfg = _training_config(config)

    def box_loss(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))

    def class_loss(y_true, y_pred):
        alpha, gamma = 0.25, 2.0
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return tf.reduce_mean(alpha * tf.pow(1 - p_t, gamma) * ce)

    model.compile(
        optimizer=_optimizer(cfg),
        loss={'boxes': box_loss, 'classes': class_loss},
        loss_weights={'boxes': 1.0, 'classes': 1.0},
        metrics={'classes': ['accuracy']},
    )
    return model


def compile_segmentation_model(model: SegmentationModel, config: Optional[Dict[str, Any]] = None) -> SegmentationModel:
    """Compile a multi-class segmentation model with CE + Dice loss."""
    cfg = _training_config(config)

    def dice_loss(y_true, y_pred, smooth=1e-6):
        depth = tf.shape(y_pred)[-1]
        one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=depth)
        y_true_f = tf.reshape(one_hot, [tf.shape(one_hot)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))

    def combined_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    def mean_iou(y_true, y_pred):
        metric = tf.keras.metrics.MeanIoU(num_classes=model.num_classes)
        return metric(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32))

    model.compile(optimizer=_optimizer(cfg), loss=combined_loss, metrics=['accuracy', mean_iou])
    return model


def compile_ocr_model(model: OCRModel, config: Optional[Dict[str, Any]] = None) -> OCRModel:
    """Compile OCR model with CTC loss."""
    cfg = _training_config(config)
    model.compile(optimizer=_optimizer(cfg), loss=model.ctc_loss, metrics=['accuracy'])
    return model


def get_model_info(model: tf.keras.Model) -> Dict[str, Any]:
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'layers': len(model.layers),
    }
