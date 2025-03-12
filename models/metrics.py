# models/metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict

class MetricsTracker:
    """Tracks and computes various metrics during training"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        
    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor):
        """Update metrics with new predictions"""
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
        
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        
        # Convert to numpy arrays
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0
            
        self.reset()  # Reset for next epoch
        return metrics