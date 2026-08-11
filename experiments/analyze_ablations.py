"""
Paired analysis of the ablation study.

Each ablation variant is run on the *same* seeds as the full model, and each
seed draws the same project split for every variant, so the per-seed F1 values
are genuinely paired. This script therefore compares variants with a paired
test rather than by eyeballing means whose standard deviations overlap.

For every variant we report:
  * the mean +- std difference from the full model,
  * how many seeds the full model wins,
  * a paired Wilcoxon signed-rank p-value and the matched-pairs rank-biserial
    correlation (the appropriate paired effect size),
  * a verdict that is explicit about being inconclusive when it is.

Usage:
    python experiments/analyze_ablations.py [results/ablations5_*.json ...]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import json

import numpy as np
from scipy.stats import wilcoxon

FULL = "Full VulMorph-Fed"


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation: (R+ - R-) / (R+ + R-)."""
    d = np.asarray([x for x in diffs if x != 0.0])
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total else 0.0


def analyse(path, metric="f1"):
    data = json.load(open(path))
    if FULL not in data:
        print(f"  (no '{FULL}' entry in {Path(path).name}; skipping)")
        return {}

    full = data[FULL][metric]["values"]
    n = len(full)
    print(f"\n=== {Path(path).name}  [{metric.upper()}, {n} seeds, paired] ===")
    print(f"  {FULL}: {np.mean(full):.4f} +- {np.std(full):.4f}")
    print(f"  {'variant':<32} {'delta(full-variant)':>20} {'full wins':>10} "
          f"{'p':>8} {'effect':>8}  verdict")

    out = {}
    for name, entry in data.items():
        if name == FULL or not isinstance(entry, dict):
            continue
        if metric not in entry or "values" not in entry[metric]:
            continue
        vals = entry[metric]["values"]
        if len(vals) != n:
            continue
        diffs = [f - v for f, v in zip(full, vals)]
        wins = sum(1 for d in diffs if d > 0)
        try:
            _, p = wilcoxon(full, vals)
        except ValueError:
            p = 1.0
        eff = rank_biserial(diffs)

        if p < 0.05 and np.mean(diffs) > 0:
            verdict = "component HELPS (significant)"
        elif p < 0.05 and np.mean(diffs) < 0:
            verdict = "component HURTS (significant)"
        elif abs(np.mean(diffs)) < 0.005:
            verdict = "no effect"
        else:
            verdict = "INCONCLUSIVE (underpowered)"

        print(f"  {name:<32} {np.mean(diffs):>+10.4f} +- {np.std(diffs):.4f} "
              f"{wins:>6}/{n} {p:>8.3f} {eff:>+8.2f}  {verdict}")
        out[name] = {"delta_mean": float(np.mean(diffs)),
                     "delta_std": float(np.std(diffs)),
                     "full_wins": wins, "n": n,
                     "wilcoxon_p": float(p), "rank_biserial": eff,
                     "verdict": verdict}
    return out


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(str(Path(__file__).parent / "results" / "ablations*.json")))
    summary = {}
    for p in paths:
        for metric in ("f1", "auc"):
            res = analyse(p, metric)
            if res:
                summary[f"{Path(p).stem}:{metric}"] = res

    out = Path(__file__).parent / "results" / "ablation_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    print("\nNOTE: with 5 paired seeds the smallest attainable two-sided "
          "Wilcoxon p-value is 0.0625, so no single-dataset comparison can "
          "reach p < 0.05. Treat per-dataset results as directional and pool "
          "across datasets for the significance claim.")


if __name__ == "__main__":
    main()
