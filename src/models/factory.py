"""Model factory for TensorVision AI."""

from typing import Optional, Dict, Any
from src.config import get_config
from src.models.cnn import CustomCNN, create_custom_cnn
from src.models.transfer_learning import TransferLearningModel, create_transfer_learning_model


def create_model(
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """Create model based on configuration.
    
    Args:
        num_classes: Number of output classes
        config: Configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    
    cfg = config or get_config().model
    model_type = cfg.get('type', 'cnn')
    architecture = cfg.get('architecture', 'custom_cnn')
    
    if model_type == 'transfer_learning' or architecture in [
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