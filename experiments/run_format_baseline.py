"""
Formatting-only baselines: classifiers that never look at what the code says.

BigVul's benign and vulnerable functions differ in raw formatting (for example,
trailing whitespace is present in most vulnerable functions and almost no
benign ones). Two baselines expose how much a model could gain from formatting
alone, on exactly the same project-level folds:

  trail|rule  score = fraction of lines with trailing whitespace
  format|gbm  gradient boosting over sixteen formatting/size statistics
              (length, lines, tabs, CRs, trailing whitespace, blank lines,
              comment markers, indentation, ...), trained per fold

Results are stored in the format that analyze_representation.py merges.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gzip
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.protocol import trivial_all_positive

NAMES = ["len", "lines", "tab", "cr", "trail", "blank", "comments", "indent",
         "dblnl", "endsnl", "startsws", "dblspace", "maxline", "contline",
         "brace", "hash"]


def feats(s):
    lines = s.split("\n")
    n = max(1, len(s))
    ind = [len(l) - len(l.lstrip(" \t")) for l in lines if l.strip()]
    return [len(s), len(lines), s.count("\t") / n, s.count("\r") / n,
            sum(l != l.rstrip() for l in lines) / len(lines),
            sum(not l.strip() for l in lines) / len(lines),
            s.count("/*") + s.count("//"), np.mean(ind) if ind else 0,
            s.count("\n\n"), int(s.endswith("\n")),
            int(s.startswith(" ") or s.startswith("\t")), s.count("  ") / n,
            max((len(l) for l in lines), default=0), int("\\\n" in s),
            s.count("{") / n, s.count("#") / n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    d = json.load(gzip.open(f"colab_data/{args.dataset}.json.gz", "rt"))
    y = np.asarray(d["y"])
    proj = d["project"]
    X = np.array([feats(s) for s in d["codes"]])
    res = {"_config": {"dataset": args.dataset, "taxonomy_size": 8,
                       "edge_set": "all", "features": NAMES},
           "_folds": [f["info"] for f in d["folds"]], "_trivial": {}}
    cells = {"trail|rule": {"auc": [], "auprc": [], "folds": [], "scores": {}},
             "format|gbm": {"auc": [], "auprc": [], "folds": [], "scores": {}}}
    for fi, f in enumerate(d["folds"]):
        tr, te = f["train"], f["test"]
        gb = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, class_weight="balanced",
            random_state=1000 + fi).fit(X[tr], y[tr])
        out = {"trail|rule": X[te, NAMES.index("trail")],
               "format|gbm": gb.predict_proba(X[te])[:, 1]}
        for key, s in out.items():
            yt = y[te]
            c = cells[key]
            c["auc"].append(float(roc_auc_score(yt, s)))
            c["auprc"].append(float(average_precision_score(yt, s)))
            c["folds"].append(fi)
            c["scores"][str(fi)] = {"y": [float(v) for v in yt],
                                    "s": [round(float(v), 6) for v in s],
                                    "groups": [proj[i] for i in te]}
        res["_trivial"][str(fi)] = trivial_all_positive(list(y[te]))
    res.update(cells)
    out_path = Path(__file__).parent / "results" / f"{args.dataset}_format.json"
    json.dump(res, open(out_path, "w"))
    for k, c in cells.items():
        print(f"{args.dataset} {k}: mean AUC {np.mean(c['auc']):.3f} "
              f"AUPRC {np.mean(c['auprc']):.3f}")


if __name__ == "__main__":
    main()
