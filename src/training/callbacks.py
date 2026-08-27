"""Training callbacks for TensorVision AI."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import tensorflow as tf
from src.config import get_config


class ModelMetadataCallback(tf.keras.callbacks.Callback):
    """Callback to save model metadata at the end of training."""
    
    def __init__(self, model_dir: str, model_name: str, version: str, metadata: Dict[str, Any]):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.version = version
        self.metadata = metadata
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_end(self, logs=None):
        """Save metadata when training ends."""
        import json
        from datetime import datetime
        
        metadata = {
            **self.metadata,
            'training_completed': datetime.now().isoformat(),
            'final_metrics': logs or {},
            'version': self.version,
        }
        
        metadata_path = self.model_dir / f'{self.model_name}_{self.version}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def get_callbacks(
    config: Optional[Dict[str, Any]] = None,
    model_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[tf.keras.callbacks.Callback]:
    """Get training callbacks based on configuration."""
    cfg = config or get_config().training
    output_cfg = get_config().output
    logging_cfg = get_config().logging
    
    callbacks = []
    
    model_dir = model_dir or output_cfg.get('model_dir', 'models')
    model_name = model_name or output_cfg.get('model_name', 'tensorvision_model')
    version = version or 'v001'
    
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_cfg = cfg.get('model_checkpoint', {})
    if checkpoint_cfg.get('enabled', True):
        filepath = model_dir_path / f'{model_name}_{version}.{{epoch:02d}}-{{val_accuracy:.4f}}.keras'
        if checkpoint_cfg.get('save_best_only', True):
            filepath = model_dir_path / f'{model_name}_{version}_best.keras'
        
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(filepath),
            monitor=checkpoint_cfg.get('monitor', 'val_accuracy'),
            mode=checkpoint_cfg.get('mode', 'max'),
            save_best_only=checkpoint_cfg.get('save_best_only', True),
            save_weights_only=False,
            verbose=1
        ))
    
    early_stopping_cfg = cfg.get('early_stopping', {})
    if early_stopping_cfg.get('enabled', True):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_cfg.get('monitor', 'val_loss'),
            patience=early_stopping_cfg.get('patience', 10),
            mode=early_stopping_cfg.get('mode', 'min'),
            restore_best_weights=early_stopping_cfg.get('restore_best_weights', True),
            verbose=1
        ))
    
    reduce_lr_cfg = cfg.get('reduce_lr_on_plateau', {})
    if reduce_lr_cfg.get('enabled', True):
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_cfg.get('monitor', 'val_loss'),
            factor=reduce_lr_cfg.get('factor', 0.5),
            patience=reduce_lr_cfg.get('patience', 5),
            min_lr=reduce_lr_cfg.get('min_lr', 1e-6),
            mode=reduce_lr_cfg.get('mode', 'min'),
            verbose=1
        ))
    
    tensorboard_cfg = logging_cfg.get('tensorboard', {})
    if tensorboard_cfg.get('enabled', True):
        log_dir = Path(tensorboard_cfg.get('log_dir', 'logs/tensorboard'))
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=tensorboard_cfg.get('histogram_freq', 1),
            write_graph=tensorboard_cfg.get('write_graph', True),
            write_images=tensorboard_cfg.get('write_images', False),
            update_freq='epoch'
        ))
    
    csv_logger_cfg = logging_cfg.get('csv_logger', {})
    if csv_logger_cfg.get('enabled', True):
        log_dir = Path(csv_logger_cfg.get('log_dir', 'logs/csv'))
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(
            filename=str(log_dir / f'training_{model_name}_{version}.csv'),
            append=True
        ))
    
    if metadata is not None:
        callbacks.append(ModelMetadataCallback(
            model_dir=model_dir,
            model_name=model_name,
            version=version,
            metadata=metadata
        ))
    
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    
    return callbacks