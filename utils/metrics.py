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

def best_f1_threshold(y_true, y_pred_probs):
    """
    Choose the decision threshold that maximises F1 on a *calibration* split
    drawn from the training projects (never the test set). Scans the
    precision-recall curve, so it is exact rather than grid-based.
    """
    from sklearn.metrics import precision_recall_curve
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_pred_probs)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def compute_cp_metrics(y_true, y_pred_probs, source_projects, target_projects, threshold=0.5):
    """
    Computes Cross-Project F1 (CP-F1).
    Evaluates only on functions from projects NOT seen during training.
    """
    # Filter indices where project is in target_projects
    # This requires projects to be tracked in the dataset.
    pass

def compute_paired_accuracy(pairs, threshold=0.5):
    """
    Computes Paired Accuracy (P-C).
    Requires a list of tuples: (prob_vulnerable_version, prob_patched_version).
    Model must rank vulnerable > patched.
    """
    correct = sum(1 for p_vuln, p_patch in pairs if p_vuln > p_patch)
    return correct / max(1, len(pairs))
