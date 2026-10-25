"""
Function-size baseline: score every function by its number of named AST nodes
(untruncated) and nothing else. Vulnerable functions are longer than benign
ones in BigVul and DiverseVul, so any detector may be partly learning length;
this rule-based 'model' shows how much AUC length alone buys on the same
folds. Stored in the same format as run_representation.py so the same paired
tests apply.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from data.loaders.sampled import load_sampled
from data.loaders.real_datasets import dedupe_functions
from utils.protocol import project_group_kfold, trivial_all_positive


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bigvul")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--sample_seed", type=int, default=0)
    args = p.parse_args()
    data, _, _ = load_sampled(args.dataset, args.n, args.sample_seed, 8)
    data = dedupe_functions(data)
    folds = list(project_group_kfold(data, 5, 2))
    cell = {"auc": [], "auprc": [], "folds": [], "scores": {}}
    res = {"_config": {"dataset": args.dataset, "n": args.n,
                       "sample_seed": args.sample_seed, "taxonomy_size": 8,
                       "edge_set": "all", "rule": "score = untruncated node count"},
           "_folds": [f[2] for f in folds], "size|rule": cell, "_trivial": {}}
    for fi, (tr, te, info) in enumerate(folds):
        y = np.array([float(data[i].y[0]) for i in te])
        s = np.array([float(data[i].n_full) for i in te])
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
        ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0
        cell["auc"].append(float(auc)); cell["auprc"].append(float(ap))
        cell["folds"].append(fi)
        cell["scores"][str(fi)] = {"y": list(y), "s": list(s),
                                   "groups": [data[i].project for i in te]}
        res["_trivial"][str(fi)] = trivial_all_positive(list(y))
    out = Path(__file__).parent / "results" / f"{args.dataset}_size.json"
    json.dump(res, open(out, "w"))
    print(f"{args.dataset}: size-only mean AUC {np.mean(cell['auc']):.3f} "
          f"AUPRC {np.mean(cell['auprc']):.3f}")


if __name__ == "__main__":
    main()
