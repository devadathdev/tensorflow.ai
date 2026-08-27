"""Evaluation metrics for TensorVision AI."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
from typing import Dict, List, Optional, Tuple, Any
import json


class EvaluationMetrics:
    """Comprehensive evaluation metrics for classification models."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else None
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        results = {}
        
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        results['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        results['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
        results['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        results['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        results['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
        results['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        results['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        results['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        results['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        if self.class_names:
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
            )
            results['per_class_metrics'] = self._compute_per_class_metrics(y_true, y_pred, cm)
        
        if y_prob is not None:
            results['roc_auc'] = self._compute_roc_auc(y_true, y_prob)
            results['roc_curves'] = self._compute_roc_curves(y_true, y_prob)
        
        return results
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cm: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics from confusion matrix."""
        per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            support = cm[i, :].sum()
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity),
                'support': int(support),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
            }
        
        return per_class
    
    def _compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute ROC AUC scores."""
        results = {}
        
        try:
            if self.num_classes == 2:
                results['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob[:, 1]))
                results['roc_auc_ovo'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                results['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
                results['roc_auc_ovo'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro'))
            
            results['roc_auc_per_class'] = {}
            for i in range(self.num_classes):
                if self.num_classes == 2 and i == 1:
                    binary_true = (y_true == i).astype(int)
                    results['roc_auc_per_class'][self.class_names[i] if self.class_names else str(i)] = \
                        float(roc_auc_score(binary_true, y_prob[:, i]))
                elif self.num_classes > 2:
                    binary_true = (y_true == i).astype(int)
                    results['roc_auc_per_class'][self.class_names[i] if self.class_names else str(i)] = \
                        float(roc_auc_score(binary_true, y_prob[:, i]))
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _compute_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Dict[str, List[float]]]:
        """Compute ROC curves for each class."""
        curves = {}
        
        for i in range(self.num_classes):
            binary_true = (y_true == i).astype(int)
            fpr, tpr, thresholds = roc_curve(binary_true, y_prob[:, i])
            curves[self.class_names[i] if self.class_names else str(i)] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(auc(fpr, tpr))
            }
        
        return curves
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print formatted metrics summary."""
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        print(f"Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"Precision (macro):  {metrics.get('precision_macro', 0):.4f}")
        print(f"Recall (macro):     {metrics.get('recall_macro', 0):.4f}")
        print(f"F1 (macro):         {metrics.get('f1_macro', 0):.4f}")
        print(f"Precision (weighted): {metrics.get('precision_weighted', 0):.4f}")
        print(f"Recall (weighted):    {metrics.get('recall_weighted', 0):.4f}")
        print(f"F1 (weighted):        {metrics.get('f1_weighted', 0):.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC (OVR):      {metrics['roc_auc'].get('roc_auc_ovr', 0):.4f}")
            print(f"ROC AUC (OVO):      {metrics['roc_auc'].get('roc_auc_ovo', 0):.4f}")
        
        if 'per_class_metrics' in metrics:
            print("\nPer-Class Metrics:")
            print("-" * 60)
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall:    {class_metrics['recall']:.4f}")
                print(f"    F1:        {class_metrics['f1']:.4f}")
                print(f"    Support:   {class_metrics['support']}")
        
        print("="*60 + "\n")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function to compute all metrics."""
    evaluator = EvaluationMetrics(class_names)
    return evaluator.compute_all(y_true, y_pred, y_prob)