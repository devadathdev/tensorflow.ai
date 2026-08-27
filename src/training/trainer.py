"""Trainer for TensorVision AI."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import tensorflow as tf
import numpy as np
from src.config import get_config
from src.models.factory import compile_model, get_model_info
from src.training.callbacks import get_callbacks
from src.data.dataset import ImageDataset


class Trainer:
    """Model trainer with full training pipeline."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: Optional[tf.data.Dataset] = None,
        config: Optional[Dict[str, Any]] = None,
        class_weights: Optional[Dict[int, float]] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config or get_config().training
        self.class_weights = class_weights
        
        self.history = None
        self.output_config = get_config().output
    
    def compile_model(self) -> None:
        """Compile model with training configuration."""
        self.model = compile_model(self.model, self.config)
    
    def train(self, epochs: Optional[int] = None) -> tf.keras.callbacks.History:
        """Train the model."""
        epochs = epochs or self.config.get('epochs', 50)
        
        callbacks = get_callbacks(
            config=self.config,
            metadata={
                'model_info': get_model_info(self.model),
                'training_config': self.config,
                'class_weights': self.class_weights,
            }
        )
        
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """Evaluate model on dataset."""
        eval_dataset = dataset or self.val_dataset
        results = self.model.evaluate(eval_dataset, verbose=1, return_dict=True)
        return results
    
    def evaluate_test(self) -> Optional[Dict[str, float]]:
        """Evaluate on test dataset if available."""
        if self.test_dataset is not None:
            return self.evaluate(self.test_dataset)
        return None
    
    def save_model(
        self,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        save_format: Optional[str] = None
    ) -> str:
        """Save trained model."""
        model_dir = model_dir or self.output_config.get('model_dir', 'models')
        model_name = model_name or self.output_config.get('model_name', 'tensorvision_model')
        version = version or 'v001'
        save_format = save_format or self.output_config.get('save_format', 'tf')
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'h5':
            filepath = model_path / f'{model_name}_{version}.h5'
            self.model.save(str(filepath), save_format='h5')
        else:
            filepath = model_path / f'{model_name}_{version}'
            self.model.save(str(filepath), save_format='tf')
        
        return str(filepath)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history as dictionary."""
        if self.history is None:
            return {}
        return self.history.history


def train_model(
    model: tf.keras.Model,
    dataset: ImageDataset,
    config: Optional[Dict[str, Any]] = None,
    augmentation_pipeline: Optional[tf.keras.layers.Layer] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Complete training pipeline from dataset to trained model."""
    
    train_ds, val_ds, test_ds = dataset.get_datasets(
        augmentation_pipeline=augmentation_pipeline,
        augment_train=True
    )
    
    class_weights = dataset.get_class_weights()
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        config=config,
        class_weights=class_weights
    )
    
    trainer.compile_model()
    trainer.train()
    
    val_results = trainer.evaluate()
    print(f"Validation Results: {val_results}")
    
    test_results = trainer.evaluate_test()
    if test_results:
        print(f"Test Results: {test_results}")
    
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    
    return trainer.model, trainer.history