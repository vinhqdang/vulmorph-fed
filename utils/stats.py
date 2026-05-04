import numpy as np
from scipy.stats import wilcoxon

def cliffs_delta(lst1, lst2):
    """
    Computes Cliff's delta effect size for two lists of observations.
    Returns delta value between -1 and 1.
    """
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0:
        return 0.0
        
    dom = 0
    for x in lst1:
        for y in lst2:
            if x > y:
                dom += 1
            elif x < y:
                dom -= 1
                
    return dom / (m * n)

def run_statistical_tests(model_a_scores, model_b_scores):
    """
    Runs Wilcoxon signed-rank test and calculates Cliff's delta.
    Args:
        model_a_scores: List of F1 scores from k-fold runs of model A (e.g. VulMorph-Fed)
        model_b_scores: List of F1 scores from k-fold runs of model B (e.g. baseline)
    """
    try:
        stat, p_value = wilcoxon(model_a_scores, model_b_scores)
    except ValueError:
        # Occurs if all differences are zero
        stat, p_value = 0, 1.0
        
    delta = cliffs_delta(model_a_scores, model_b_scores)
    
    return {
        "wilcoxon_p": p_value,
        "cliffs_delta": delta,
        "significant": p_value < 0.05
    }
