import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score, 
    confusion_matrix, brier_score_loss
)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Computes a comprehensive suite of evaluation metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total': int(len(y_true))
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
            metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
            metrics['brier_score'] = None
            
    return metrics
