"""Inference engine for TensorVision AI."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from src.config import get_config
from src.data.preprocessing import PreprocessingPipeline


class InferenceEngine:
    """High-performance inference engine for trained models."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str],
        preprocessing: Optional[PreprocessingPipeline] = None,
        config: Optional[Dict] = None,
        model_version: str = "unknown",
        model_metadata: Optional[Dict] = None
    ):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.config = config or get_config().inference
        self.model_version = model_version
        self.model_metadata = model_metadata or {}
        
        self.preprocessing = preprocessing or PreprocessingPipeline(
            image_size=tuple(get_config().dataset.get('image_size', [224, 224]))
        )
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.top_k = self.config.get('top_k', 5)
        self.batch_size = self.config.get('batch_size', 32)
        
        self._compiled_model = None
    
    def _get_compiled_model(self) -> tf.keras.Model:
        """Get or create compiled model for inference."""
        if self._compiled_model is None:
            self._compiled_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.output
            )
        return self._compiled_model
    
    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image, tf.Tensor]) -> tf.Tensor:
        """Preprocess a single image for inference."""
        if isinstance(image, (str, Path)):
            image = tf.io.read_file(str(image))
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.convert_image_dtype(image, tf.float32)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        elif isinstance(image, np.ndarray):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        image = self.preprocessing.preprocess(image)
        return image
    
    def preprocess_batch(self, images: List[Union[str, Path, np.ndarray, Image.Image]]) -> tf.Tensor:
        """Preprocess a batch of images."""
        processed = []
        for img in images:
            processed.append(self.preprocess_image(img))
        return tf.concat(processed, axis=0)
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image, tf.Tensor],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Run inference on a single image."""
        processed = self.preprocess_image(image)
        predictions = self._get_compiled_model()(processed, training=False)
        probs = predictions.numpy()[0]
        
        return self._format_prediction(probs, return_probabilities)
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of images."""
        processed = self.preprocess_batch(images)
        predictions = self._get_compiled_model()(processed, training=False)
        probs = predictions.numpy()
        
        results = []
        for prob in probs:
            results.append(self._format_prediction(prob, return_probabilities))
        
        return results
    
    def _format_prediction(
        self,
        probs: np.ndarray,
        return_probabilities: bool
    ) -> Dict[str, Any]:
        """Format prediction results."""
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_classes = [self.class_names[i] for i in top_k_indices]
        
        pred_class_idx = top_k_indices[0]
        pred_class = top_k_classes[0]
        confidence = float(top_k_probs[0])
        
        result = {
            'prediction': pred_class,
            'confidence': confidence,
            'model_version': self.model_version,
            'class_index': int(pred_class_idx),
            'above_threshold': confidence >= self.confidence_threshold
        }
        
        if return_probabilities:
            result['top_k'] = [
                {'class': cls, 'confidence': float(prob), 'index': int(idx)}
                for cls, prob, idx in zip(top_k_classes, top_k_probs, top_k_indices)
            ]
            result['all_probabilities'] = {
                self.class_names[i]: float(probs[i]) for i in range(self.num_classes)
            }
        
        return result
    
    def predict_from_dataset(
        self,
        dataset: tf.data.Dataset,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """Run inference on a TensorFlow dataset."""
        all_results = []
        
        for images, _ in dataset:
            predictions = self._get_compiled_model()(images, training=False)
            probs = predictions.numpy()
            
            for prob in probs:
                all_results.append(self._format_prediction(prob, return_probabilities))
        
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_version': self.model_version,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'model_metadata': self.model_metadata,
            'total_params': self.model.count_params(),
        }


def load_model(
    model_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    custom_objects: Optional[Dict] = None
) -> Tuple[tf.keras.Model, Dict]:
    """Load a saved model with metadata."""
    model_path = Path(model_path)
    
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    
    metadata = {}
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    if class_names is None and 'class_names' in metadata:
        class_names = metadata['class_names']
    
    return model, metadata


def create_inference_engine(
    model_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> InferenceEngine:
    """Factory function to create inference engine from saved model."""
    model, metadata = load_model(model_path, class_names)
    
    if class_names is None:
        class_names = metadata.get('class_names', [f'class_{i}' for i in range(model.output_shape[-1])])
    
    model_version = metadata.get('version', model_path.stem)
    
    return InferenceEngine(
        model=model,
        class_names=class_names,
        config=config,
        model_version=model_version,
        model_metadata=metadata
    )