import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes classification metrics for vulnerability detection.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Check if there's only one class in y_true, which happens in small batches
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_probs)
    else:
        auc = 0.5
        
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
