"""
RQ4, done as a matched comparison.

The three granularity runs (|T| = 8, 16, 32) were executed with the same
project-level fold assignment: fold i holds out the same repositories in every
run, which we verify here rather than assume. That makes the honest test a
direct paired comparison of one granularity against another on their common
folds -- granularity is then the only quantity that differs.

The earlier framing compared each granularity to its own lexical baseline and
read the trend off three separate differences. Those baselines were themselves
estimated on different numbers of folds, so the trend confounded granularity
with which folds each run happened to complete. This script does not.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.protocol import cluster_bootstrap_ci

RESULTS = Path(__file__).parent / "results"
RUNS = [(8, "core_bigvul"), (16, "taxonomy_t16"), (32, "taxonomy_t32")]
CELL = "op_kind|ggnn"


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5


def _ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0


def load(stem):
    p = RESULTS / f"{stem}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d.get(CELL)


def pooled(cell_a, cell_b, folds, metric_fn, n_boot, n_seeds=10):
    ys, sa, sb, gs = [], [], [], []
    for f in folds:
        fa, fb = cell_a["scores"].get(f), cell_b["scores"].get(f)
        if not fa or not fb:
            continue
        ys += fa["y"]; sa += fa["s"]; sb += fb["s"]
        gs += [f"{f}:{g}" for g in fa["groups"]]
    if not ys:
        return None
    out, excl = None, []
    for s in range(n_seeds):
        c = cluster_bootstrap_ci(ys, sa, sb, gs, metric_fn,
                                 n_boot=n_boot, seed=s)
        if s == 0:
            out = c
        excl.append(1.0 if (c["lo"] > 0 or c["hi"] < 0) else 0.0)
    out["seed_stability"] = float(np.mean(excl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--output", default=str(RESULTS / "granularity_analysis.json"))
    args = ap.parse_args()

    cells = {t: load(stem) for t, stem in RUNS}
    have = {t: c for t, c in cells.items() if c}
    print("granularities with data:", sorted(have))

    base_t = min(have)
    base = have[base_t]
    out = {"reference": base_t, "comparisons": {}, "fold_check": {}}

    for t, cell in sorted(have.items()):
        if t == base_t:
            continue
        common = sorted(set(cell["scores"]) & set(base["scores"]), key=int)
        # The comparison is only meaningful if fold i means the same held-out
        # repositories in both runs. Verify, do not assume.
        mismatched = [f for f in common
                      if set(cell["scores"][f]["groups"])
                      != set(base["scores"][f]["groups"])]
        out["fold_check"][f"{t}_vs_{base_t}"] = {
            "common_folds": common, "mismatched_folds": mismatched}
        if mismatched:
            print(f"  |T|={t}: fold assignment differs on {mismatched}; skipping")
            continue

        entry = {"n_folds": len(common), "folds": common}
        for metric, fn in (("auc", _auc), ("auprc", _ap)):
            a = np.array([cell[metric][cell["folds"].index(int(f))] for f in common])
            b = np.array([base[metric][base["folds"].index(int(f))] for f in common])
            ci = pooled(cell, base, common, fn, args.n_boot)
            crosses = not (ci["lo"] > 0 or ci["hi"] < 0)
            entry[metric] = {
                "per_fold_mean": float((a - b).mean()),
                "wins": int((a - b > 0).sum()), **ci,
                "verdict": ("INCONCLUSIVE (CI crosses 0)" if crosses
                            else ("finer HELPS" if ci["diff"] > 0 else "finer HURTS")),
                f"mean_{metric}_fine": float(a.mean()),
                f"mean_{metric}_coarse": float(b.mean()),
            }
            print(f"  |T|={t} vs |T|={base_t}  [{metric.upper()}]  "
                  f"n={len(common)} folds  "
                  f"d={ci['diff']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}] "
                  f"p={ci['p_two_sided']:.3f}  {entry[metric]['verdict']}")
        out["comparisons"][str(t)] = entry

    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
