"""TensorFlow Lite inference for TensorVision AI."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import get_config
from src.data.preprocessing import PreprocessingPipeline


class TFLiteInferenceEngine:
    """TensorFlow Lite inference engine."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: List[str],
        config: Optional[Dict] = None,
        model_version: str = "unknown",
        num_threads: int = 4
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.config = config or get_config().inference
        self.model_version = model_version
        self.num_threads = num_threads
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Preprocessing
        self.preprocessing = PreprocessingPipeline(
            image_size=tuple(get_config().dataset.get('image_size', [224, 224]))
        )
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.top_k = self.config.get('top_k', 5)
        
        # Check if quantized
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        if self.is_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
    
    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for TFLite inference."""
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
        image = image.numpy()
        
        # Handle quantized input
        if self.is_quantized:
            image = (image / self.input_scale + self.input_zero_point).astype(np.uint8)
        
        return image
    
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Run inference on a single image."""
        processed = self.preprocess_image(image)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], processed)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output if needed
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        probs = output[0]
        return self._format_prediction(probs)
    
    def predict_batch(self, images: List[Union[str, Path, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Run inference on a batch of images."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results
    
    def _format_prediction(self, probs: np.ndarray) -> Dict[str, Any]:
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
        
        result['top_k'] = [
            {'class': cls, 'confidence': float(prob), 'index': int(idx)}
            for cls, prob, idx in zip(top_k_classes, top_k_probs, top_k_indices)
        ]
        result['all_probabilities'] = {
            self.class_names[i]: float(probs[i]) for i in range(self.num_classes)
        }
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_version': self.model_version,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype']),
            'is_quantized': self.is_quantized,
            'total_params': 'N/A (TFLite)',
        }


def create_tflite_engine(
    model_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    num_threads: int = 4
) -> TFLiteInferenceEngine:
    """Factory function to create TFLite inference engine."""
    
    model_path = Path(model_path)
    
    # Try to load class names from metadata
    if class_names is None:
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                class_names = metadata.get('class_names')
    
    if class_names is None:
        raise ValueError("class_names must be provided or available in metadata JSON")
    
    model_version = model_path.stem
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            model_version = metadata.get('model_version', model_path.stem)
    
    return TFLiteInferenceEngine(
        model_path=model_path,
        class_names=class_names,
        config=config,
        model_version=model_version,
        num_threads=num_threads
    )


def benchmark_tflite(
    model_path: Union[str, Path],
    class_names: List[str],
    image: Union[str, Path, np.ndarray, Image.Image],
    num_runs: int = 100,
    warmup_runs: int = 10,
    num_threads: int = 4
) -> Dict[str, float]:
    """Benchmark TFLite model inference speed."""
    
    import time
    
    engine = create_tflite_engine(model_path, class_names, num_threads=num_threads)
    
    # Warmup
    for _ in range(warmup_runs):
        engine.predict(image)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        engine.predict(image)
        times.append(time.perf_counter() - start)
    
    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
        'p95_ms': float(np.percentile(times, 95) * 1000),
        'p99_ms': float(np.percentile(times, 99) * 1000),
        'throughput_fps': float(1.0 / np.mean(times)),
    }