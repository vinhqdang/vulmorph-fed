"""
Analysis for the representation x backbone study.

Compares every feature mode against the lexical baseline, holding backbone and
fold fixed, and reports:

  * the paired per-fold difference (mean, std, win rate),
  * a cluster-bootstrap CI on the pooled difference that resamples held-out
    PROJECTS, which is the correct unit given functions inside a repository
    are not independent,
  * the trivial all-positive floor, so an absolute number can be read.

The verdict is deliberately explicit about being inconclusive when the CI
crosses zero. The previous version of this study reported directional
differences as findings; this one does not.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.protocol import cluster_bootstrap_ci

BASELINE_MODE = "lexical"


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5


def _ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0


def analyse(path, metric="auc", n_boot=2000, n_seeds=10):
    data = json.load(open(path))
    cells = {k: v for k, v in data.items() if k.count("|") == 1}
    if not cells:
        print(f"  (no cells in {Path(path).name})")
        return {}

    by_backbone = defaultdict(dict)
    for key, cell in cells.items():
        mode, bb = key.split("|")
        by_backbone[bb][mode] = cell

    metric_fn = _auc if metric == "auc" else _ap
    print(f"\n=== {Path(path).name}  [{metric.upper()}] ===")
    trivial = data.get("_trivial", [])
    if trivial:
        print(f"  trivial all-positive floor: AUC 0.500, "
              f"F1 {np.mean([t['f1'] for t in trivial]):.3f}")

    out = {}
    for bb, modes in sorted(by_backbone.items()):
        base = modes.get(BASELINE_MODE)
        if not base:
            continue
        print(f"\n  backbone = {bb}   (baseline: {BASELINE_MODE}, "
              f"{metric} = {np.mean(base[metric]):.4f})")
        for mode, cell in sorted(modes.items()):
            if mode == BASELINE_MODE:
                continue
            common = sorted(set(cell["folds"]) & set(base["folds"]))
            if not common:
                continue
            a = np.array([cell[metric][cell["folds"].index(f)] for f in common])
            b = np.array([base[metric][base["folds"].index(f)] for f in common])
            d = a - b
            wins = int((d > 0).sum())

            # Pooled cluster bootstrap over held-out projects, using the raw
            # per-fold scores rather than the summarised metric.
            ci = {"lo": float("nan"), "hi": float("nan"), "p_two_sided": 1.0}
            sc_a, sc_b = cell.get("scores", {}), base.get("scores", {})
            ys, sa, sb, gs = [], [], [], []
            for f in common:
                fa, fb = sc_a.get(str(f)), sc_b.get(str(f))
                if not fa or not fb:
                    continue
                ys += fa["y"]; sa += fa["s"]; sb += fb["s"]
                gs += [f"{f}:{g}" for g in fa["groups"]]
            # A bootstrap interval is itself a random quantity. For an effect
            # sitting near zero the verdict can depend on the resampling draw,
            # so we repeat the bootstrap under several seeds and record how
            # often the interval excludes zero. A conclusion that holds for
            # only some seeds is reported as borderline rather than as a
            # finding; the headline interval remains the seed-0 one.
            stability = float("nan")
            if ys:
                excl = []
                for s in range(n_seeds):
                    c = cluster_bootstrap_ci(ys, sa, sb, gs, metric_fn,
                                             n_boot=n_boot, seed=s)
                    if s == 0:
                        ci = c
                    excl.append(1.0 if (c["lo"] > 0 or c["hi"] < 0) else 0.0)
                stability = float(np.mean(excl))

            crosses = not (ci["lo"] > 0 or ci["hi"] < 0)
            if crosses:
                verdict = "INCONCLUSIVE (CI crosses 0)"
            elif stability == stability and stability < 0.9:
                verdict = f"BORDERLINE (excludes 0 in {stability:.0%} of seeds)"
            else:
                verdict = "mode HELPS" if ci["diff"] > 0 else "mode HURTS"
            print(f"    {mode:<10} d={d.mean():+.4f}±{d.std():.4f} "
                  f"wins {wins}/{len(d)}  "
                  f"pooled d={ci['diff']:+.4f} "
                  f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] "
                  f"p={ci['p_two_sided']:.3f}  stab={stability:.2f}  {verdict}")
            out[f"{bb}|{mode}"] = {
                "per_fold_mean": float(d.mean()), "per_fold_std": float(d.std()),
                "wins": wins, "n_folds": len(d), **ci,
                "seed_stability": stability, "n_seeds": n_seeds,
                "verdict": verdict}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=10)
    args = p.parse_args()
    paths = args.paths or sorted(
        glob.glob(str(Path(__file__).parent / "results" / "*representation*.json"))
        + glob.glob(str(Path(__file__).parent / "results" / "core_*.json")))
    summary = {}
    for path in paths:
        for m in ("auc", "auprc"):
            r = analyse(path, m, args.n_boot, args.n_seeds)
            if r:
                summary[f"{Path(path).stem}:{m}"] = r
    out = Path(__file__).parent / "results" / "representation_analysis.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
