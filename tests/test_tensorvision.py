"""Tests for TensorVision AI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import os

from src.config import Config, get_config
from src.data.dataset import ImageDataset
from src.data.preprocessing import PreprocessingPipeline
from src.data.augmentation import AugmentationPipeline
from src.models.cnn import CustomCNN
from src.models.transfer_learning import TransferLearningModel
from src.models.factory import create_model, compile_model
from src.training.callbacks import get_callbacks
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.inference.engine import InferenceEngine


class TestConfig:
    """Test configuration management."""
    
    def test_config_load(self):
        config = Config('config.yaml')
        assert config.get('dataset.batch_size') == 32
        assert config.get('training.epochs') == 50
        assert config.get('model.type') == 'cnn'
    
    def test_config_get_section(self):
        config = Config('config.yaml')
        dataset_cfg = config.get_section('dataset')
        assert 'batch_size' in dataset_cfg
        assert 'image_size' in dataset_cfg
    
    def test_config_set(self):
        config = Config('config.yaml')
        config.set('test.value', 42)
        assert config.get('test.value') == 42


class TestPreprocessing:
    """Test preprocessing pipeline."""
    
    def test_preprocessing_pipeline(self):
        pipeline = PreprocessingPipeline(image_size=(224, 224))
        
        image = tf.random.uniform((300, 300, 3), 0, 255, dtype=tf.float32)
        processed = pipeline.preprocess(image)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == tf.float32
    
    def test_preprocessing_batch(self):
        pipeline = PreprocessingPipeline(image_size=(224, 224))
        
        images = tf.random.uniform((4, 300, 300, 3), 0, 255, dtype=tf.float32)
        labels = tf.constant([0, 1, 0, 1])
        processed_images, processed_labels = pipeline.preprocess_batch(images, labels)
        
        assert processed_images.shape == (4, 224, 224, 3)
        assert tf.reduce_all(processed_labels == labels)
    
    def test_preprocessing_layer(self):
        pipeline = PreprocessingPipeline(image_size=(224, 224))
        layer = pipeline.get_preprocessing_layer()
        
        image = tf.random.uniform((1, 300, 300, 3), 0, 255, dtype=tf.float32)
        processed = layer(image)
        
        assert processed.shape == (1, 224, 224, 3)


class TestAugmentation:
    """Test data augmentation."""
    
    def test_augmentation_pipeline(self):
        config = {
            'enabled': True,
            'horizontal_flip': True,
            'rotation_factor': 0.1,
        }
        pipeline = AugmentationPipeline(config)
        
        image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
        augmented = pipeline(image, training=True)
        
        assert augmented.shape == (224, 224, 3)
    
    def test_augmentation_disabled(self):
        config = {'enabled': False}
        pipeline = AugmentationPipeline(config)
        
        image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
        augmented = pipeline(image, training=True)
        
        assert tf.reduce_all(tf.equal(augmented, image))


class TestModels:
    """Test model creation."""
    
    def test_custom_cnn(self):
        model = CustomCNN(num_classes=10, input_shape=(224, 224, 3))
        
        assert model.output_shape == (None, 10)
        assert model.count_params() > 0
    
    def test_transfer_learning_model(self):
        model = TransferLearningModel(
            num_classes=10,
            backbone='mobilenetv2',
            input_shape=(224, 224, 3)
        )
        
        assert model.output_shape == (None, 10)
        assert model.count_params() > 0
    
    def test_create_model_cnn(self):
        config = {'model': {'type': 'cnn', 'architecture': 'custom_cnn'}}
        model = create_model(5, config)
        
        assert isinstance(model, CustomCNN)
        assert model.output_shape == (None, 5)
    
    def test_create_model_transfer(self):
        config = {'model': {'type': 'transfer_learning', 'architecture': 'mobilenetv2'}}
        model = create_model(5, config)
        
        assert isinstance(model, TransferLearningModel)
        assert model.output_shape == (None, 5)
    
    def test_compile_model(self):
        model = CustomCNN(num_classes=3)
        compiled = compile_model(model, {'learning_rate': 1e-3, 'optimizer': 'adam'})
        
        assert compiled.optimizer is not None
        assert compiled.loss is not None


class TestCallbacks:
    """Test training callbacks."""
    
    def test_get_callbacks(self):
        callbacks = get_callbacks()
        assert len(callbacks) > 0
        
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'ModelCheckpoint' in callback_types
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_compute_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
            [0.05, 0.1, 0.85],
        ])
        
        metrics = compute_metrics(y_true, y_pred, y_prob, ['class_0', 'class_1', 'class_2'])
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_metrics' in metrics
        assert metrics['accuracy'] == pytest.approx(0.5, rel=0.1)
    
    def test_evaluation_metrics_class(self):
        evaluator = EvaluationMetrics(['A', 'B', 'C'])
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        
        metrics = evaluator.compute_all(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.6
        assert 'per_class_metrics' in metrics


class TestInferenceEngine:
    """Test inference engine."""
    
    @pytest.fixture
    def dummy_model(self):
        model = CustomCNN(num_classes=3, input_shape=(224, 224, 3))
        model.build((None, 224, 224, 3))
        return model
    
    def test_inference_engine(self, dummy_model):
        engine = InferenceEngine(
            model=dummy_model,
            class_names=['A', 'B', 'C'],
            model_version='test_v1'
        )
        
        image = tf.random.uniform((224, 224, 3), 0, 255, dtype=tf.float32)
        result = engine.predict(image)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'model_version' in result
        assert result['model_version'] == 'test_v1'
        assert result['prediction'] in ['A', 'B', 'C']
        assert 0 <= result['confidence'] <= 1
    
    def test_inference_batch(self, dummy_model):
        engine = InferenceEngine(
            model=dummy_model,
            class_names=['A', 'B', 'C']
        )
        
        images = [tf.random.uniform((224, 224, 3), 0, 255, dtype=tf.float32) for _ in range(3)]
        results = engine.predict_batch(images)
        
        assert len(results) == 3
        for r in results:
            assert r['prediction'] in ['A', 'B', 'C']


class TestDataset:
    """Test dataset handling."""
    
    def test_create_sample_dataset(self, tmp_path):
        train_dir = tmp_path / 'train'
        val_dir = tmp_path / 'val'
        
        for split_dir in [train_dir, val_dir]:
            for class_name in ['cat', 'dog']:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True)
                for i in range(5):
                    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                    img.save(class_dir / f'{class_name}_{i}.jpg')
        
        dataset = ImageDataset(train_dir, val_dir)
        classes = dataset.discover_classes()
        
        assert classes == ['cat', 'dog']
        assert dataset.num_classes == 2
        
        counts = dataset.count_images()
        assert counts['cat'] == 5
        assert counts['dog'] == 5
        
        stats = dataset.get_statistics()
        assert stats['num_classes'] == 2
        assert stats['train']['total'] == 10
        assert stats['val']['total'] == 10


def test_tensorflow_available():
    """Verify TensorFlow is working."""
    assert tf.__version__ >= '2.15.0'
    print(f"TensorFlow version: {tf.__version__}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])