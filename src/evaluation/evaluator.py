"""Model evaluator for TensorVision AI."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.config import get_config


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.class_names = class_names
        self.config = config or get_config().inference
        self.metrics_calculator = EvaluationMetrics(class_names)
    
    def evaluate_dataset(
        self,
        dataset: tf.data.Dataset,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        y_true = []
        y_pred = []
        y_prob = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)
            batch_pred = np.argmax(predictions, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(batch_pred)
            y_prob.extend(predictions)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        results = self.metrics_calculator.compute_all(y_true, y_pred, y_prob)
        
        if return_predictions:
            results['predictions'] = {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist()
            }
        
        return results
    
    def evaluate_single_batch(
        self,
        images: tf.Tensor,
        labels: tf.Tensor
    ) -> Dict[str, Any]:
        """Evaluate on a single batch."""
        predictions = self.model.predict(images, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = labels.numpy()
        
        return compute_metrics(y_true, y_pred, predictions, self.class_names)
    
    def get_confusion_matrix(
        self,
        dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, List[str]]:
        """Get confusion matrix and class names."""
        y_true = []
        y_pred = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)
            batch_pred = np.argmax(predictions, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(batch_pred)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return cm, self.class_names
    
    def get_misclassified(
        self,
        dataset: tf.data.Dataset,
        max_samples: int = 100
    ) -> List[Dict]:
        """Get misclassified samples for analysis."""
        misclassified = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred = np.argmax(predictions, axis=1)
            y_true = labels.numpy()
            y_prob = np.max(predictions, axis=1)
            
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i] and len(misclassified) < max_samples:
                    misclassified.append({
                        'true_class': self.class_names[y_true[i]] if self.class_names else str(y_true[i]),
                        'pred_class': self.class_names[y_pred[i]] if self.class_names else str(y_pred[i]),
                        'confidence': float(y_prob[i]),
                        'true_idx': int(y_true[i]),
                        'pred_idx': int(y_pred[i]),
                        'probabilities': predictions[i].tolist()
                    })
            
            if len(misclassified) >= max_samples:
                break
        
        return misclassified
    
    def evaluate_top_k(
        self,
        dataset: tf.data.Dataset,
        k: int = 5
    ) -> Dict[str, float]:
        """Evaluate top-k accuracy."""
        y_true = []
        top_k_correct = 0
        total = 0
        
        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
            y_true_batch = labels.numpy()
            
            for i, true_label in enumerate(y_true_batch):
                if true_label in top_k_preds[i]:
                    top_k_correct += 1
                total += 1
        
        return {
            f'top_{k}_accuracy': top_k_correct / total if total > 0 else 0.0,
            'total_samples': total
        }
    
    def generate_report(
        self,
        dataset: tf.data.Dataset,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate full evaluation report."""
        results = self.evaluate_dataset(dataset, return_predictions=True)
        
        top_k = self.config.get('top_k', 5)
        top_k_results = self.evaluate_top_k(dataset, k=top_k)
        results.update(top_k_results)
        
        misclassified = self.get_misclassified(dataset)
        results['misclassified_samples'] = misclassified[:20]
        results['num_misclassified'] = len(misclassified)
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        self.metrics_calculator.print_summary(results)
        
        return results


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convenience function to evaluate a model."""
    evaluator = ModelEvaluator(model, class_names, config)
    return evaluator.generate_report(dataset)