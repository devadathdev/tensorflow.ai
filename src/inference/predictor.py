"""High-level predictor for TensorVision AI."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import tensorflow as tf
from PIL import Image
from src.inference.engine import InferenceEngine, create_inference_engine
from src.config import get_config


class ImagePredictor:
    """High-level image prediction interface."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        self.model_path = Path(model_path)
        self.config = config or get_config()
        self.engine = create_inference_engine(model_path, class_names, config)
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predict class for a single image."""
        if top_k is not None:
            original_top_k = self.engine.top_k
            self.engine.top_k = top_k
        
        if confidence_threshold is not None:
            original_threshold = self.engine.confidence_threshold
            self.engine.confidence_threshold = confidence_threshold
        
        result = self.engine.predict(image)
        
        if top_k is not None:
            self.engine.top_k = original_top_k
        if confidence_threshold is not None:
            self.engine.confidence_threshold = original_threshold
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Predict classes for multiple images."""
        if top_k is not None:
            original_top_k = self.engine.top_k
            self.engine.top_k = top_k
        
        if confidence_threshold is not None:
            original_threshold = self.engine.confidence_threshold
            self.engine.confidence_threshold = confidence_threshold
        
        results = self.engine.predict_batch(images)
        
        if top_k is not None:
            self.engine.top_k = original_top_k
        if confidence_threshold is not None:
            self.engine.confidence_threshold = original_threshold
        
        return results
    
    def predict_directory(
        self,
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict all images in a directory."""
        directory = Path(directory)
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']
        
        image_files = []
        if recursive:
            for ext in extensions:
                image_files.extend(directory.rglob(f'*{ext}'))
                image_files.extend(directory.rglob(f'*{ext.upper()}'))
        else:
            for ext in extensions:
                image_files.extend(directory.glob(f'*{ext}'))
                image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        results = []
        for img_path in image_files:
            try:
                pred = self.predict(img_path)
                pred['file_path'] = str(img_path)
                pred['file_name'] = img_path.name
                results.append(pred)
            except Exception as e:
                results.append({
                    'file_path': str(img_path),
                    'file_name': img_path.name,
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.engine.get_model_info()
    
    @classmethod
    def from_latest_model(
        cls,
        model_dir: Union[str, Path] = "models",
        pattern: str = "*_best.keras"
    ) -> 'ImagePredictor':
        """Create predictor from latest model in directory."""
        model_dir = Path(model_dir)
        models = list(model_dir.glob(pattern))
        
        if not models:
            models = list(model_dir.glob("*.keras"))
        if not models:
            models = list(model_dir.glob("*.h5"))
        if not models:
            models = list(model_dir.glob("*"))  # Check for SavedModel directories
            models = [m for m in models if m.is_dir()]
        
        if not models:
            raise FileNotFoundError(f"No models found in {model_dir}")
        
        latest_model = max(models, key=lambda p: p.stat().st_mtime)
        return cls(latest_model)
    
    def benchmark(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark inference speed."""
        import time
        
        for _ in range(warmup_runs):
            self.predict(image)
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(image)
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


def predict_image(
    image_path: Union[str, Path],
    model_path: Union[str, Path],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function for single image prediction."""
    predictor = ImagePredictor(model_path, class_names)
    return predictor.predict(image_path)