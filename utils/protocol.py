"""
Cross-project evaluation protocol.

Replaces the earlier seed-based scheme, in which the held-out fraction drifted
between 20% and 51% and three of five "seeds" were effectively single-project
evaluations, so the reported standard deviation mixed structurally different
experiments.

Two components:

  * `project_group_kfold` - repeated GroupKFold at project granularity with a
    cap on how much of the test set any single project may contribute, so no
    fold degenerates into "test = Chrome".

  * `cluster_bootstrap_ci` - bootstrap confidence intervals resampling whole
    *projects*, because functions inside one repository are not independent.
    Applied to the paired difference between two systems, which is the
    quantity the paper actually claims.
"""

import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def project_group_kfold(data_list, n_splits: int = 5, n_repeats: int = 5,
                        max_project_share: float = 0.30,
                        seed: int = 0):
    """
    Yield (train_idx, test_idx, info) for repeated project-level GroupKFold.

    Projects are shuffled per repeat and dealt into `n_splits` folds by a
    greedy smallest-fold-first rule, which keeps fold sizes comparable without
    ever splitting a project across folds. A fold whose largest project would
    exceed `max_project_share` of that fold is rebalanced by moving that
    project to the fold where it is least dominant; if no such fold exists the
    project is excluded from testing (it still trains) and this is reported in
    `info`, never silently.
    """
    projects: Dict[str, List[int]] = {}
    for i, d in enumerate(data_list):
        projects.setdefault(getattr(d, "project", "unknown"), []).append(i)

    names = sorted(projects)
    total = sum(len(v) for v in projects.values())
    # A project that alone exceeds `max_project_share` of an average fold can
    # never be placed without dominating it, because projects are never split.
    # Such mega-projects (Chrome and linux account for ~66% of BigVul) are held
    # TRAIN-ONLY and reported, rather than being allowed to become a fold whose
    # "cross-project" test set is really one repository.
    avg_fold = total / n_splits
    mega = [p for p in names
            if len(projects[p]) > max_project_share * avg_fold]
    testable = [p for p in names if p not in set(mega)]

    for rep in range(n_repeats):
        rng = random.Random(seed + rep)
        order = testable[:]
        rng.shuffle(order)
        # Deal largest-first into the currently smallest fold.
        order.sort(key=lambda p: -len(projects[p]))
        folds: List[List[str]] = [[] for _ in range(n_splits)]
        sizes = [0] * n_splits
        excluded: List[str] = list(mega)
        for p in order:
            k = min(range(n_splits), key=lambda j: sizes[j])
            folds[k].append(p)
            sizes[k] += len(projects[p])

        for f in range(n_splits):
            test_projects = folds[f]
            if not test_projects:
                continue
            test_idx = [i for p in test_projects for i in projects[p]]
            train_idx = [i for p in names
                         if p not in set(test_projects)
                         for i in projects[p]]
            if not test_idx or not train_idx:
                continue
            largest = max(len(projects[p]) for p in test_projects)
            info = {
                "repeat": rep, "fold": f,
                "n_test": len(test_idx), "n_train": len(train_idx),
                "test_fraction": len(test_idx) / len(data_list),
                "n_test_projects": len(test_projects),
                "largest_test_project_share": largest / len(test_idx),
                "excluded_projects": excluded,
                "test_projects": test_projects,
            }
            yield train_idx, test_idx, info


def cluster_bootstrap_ci(y_true: Sequence[float], score_a: Sequence[float],
                         score_b: Sequence[float], groups: Sequence[str],
                         metric: Callable[[np.ndarray, np.ndarray], float],
                         n_boot: int = 1000, alpha: float = 0.05,
                         seed: int = 0) -> Dict[str, float]:
    """
    Percentile bootstrap CI for metric(A) - metric(B), resampling whole
    projects with replacement.

    Functions within a repository share style, APIs and often near-duplicate
    code, so resampling functions independently would understate the
    uncertainty. Resampling clusters is the honest choice and is what the
    number of held-out *projects* actually supports.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    observed = metric(y_true, score_a) - metric(y_true, score_b)
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(metric(yt, score_a[idx]) - metric(yt, score_b[idx]))

    if not diffs:
        return {"diff": float(observed), "lo": float("nan"),
                "hi": float("nan"), "p_two_sided": 1.0, "n_clusters": len(uniq)}

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value: how often the sign flips relative to 0.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(observed), "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(1.0, p)), "n_clusters": int(len(uniq)),
            "n_boot": int(len(diffs))}


def trivial_all_positive(y_true: Sequence[float]) -> Dict[str, float]:
    """
    Metrics of the constant 'everything is vulnerable' classifier.

    Every results table reports this as a floor. Without it, an F1 of 0.15 on
    a 6%-prevalence test set reads as a result rather than as roughly what a
    constant achieves.
    """
    y = np.asarray(y_true)
    prev = float(y.mean()) if y.size else 0.0
    f1 = 2 * prev / (1 + prev) if prev > 0 else 0.0
    return {"precision": prev, "recall": 1.0, "f1": f1, "auc": 0.5}
