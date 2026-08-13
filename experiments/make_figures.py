"""
Figures for the representation study, generated from the v2 result files.

Two figures, both chosen because they carry information a table does not:

  fig:forest    the paired difference of every representation against the
                lexical baseline, with cluster-bootstrap intervals, across all
                corpus x backbone cells. This is the paper's central claim in
                one view, and it shows the intervals rather than only the point
                estimates.

  fig:redundancy  the conditional distribution of the operation label given the
                grammar kind, which is the evidence for the redundancy that
                explains the null taxonomy effect.

Nothing is drawn from a file that does not exist: missing cells are skipped and
reported, never imputed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent.parent / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.autolayout": True,
})

MODE_LABEL = {"op": "operation type only",
              "kind": "grammar kind only",
              "op_kind": "operation + kind"}
COLOUR = {"op": "#b0b0b0", "kind": "#2c7fb8", "op_kind": "#41ab5d"}


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def forest():
    """Paired difference vs lexical, with bootstrap CIs, per cell."""
    analysis = load("representation_analysis.json")
    if not analysis:
        print("  no representation_analysis.json; run analyze_representation.py")
        return

    rows = []
    pretty = {"core_bigvul": "BigVul", "core_diversevul": "DiverseVul",
              "backbone_bigvul": "BigVul"}
    for key, cells in analysis.items():
        stem, metric = key.rsplit(":", 1)
        if metric != "auc":
            continue
        corpus = pretty.get(stem, stem)
        for cell, s in cells.items():
            backbone, mode = cell.split("|")
            if mode not in MODE_LABEL:
                continue
            rows.append((f"{corpus} / {backbone.upper()}", mode,
                         s["diff"], s["lo"], s["hi"], s["p_two_sided"]))
    if not rows:
        print("  no AUC cells found")
        return

    # Group by cell label, order modes consistently.
    order = ["op", "kind", "op_kind"]
    labels = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.4))
    y = 0
    yticks, yticklabels = [], []
    for lab in labels:
        for mode in order:
            m = [r for r in rows if r[0] == lab and r[1] == mode]
            if not m:
                continue
            _, _, d, lo, hi, p = m[0]
            ax.plot([lo, hi], [y, y], color=COLOUR[mode], lw=2.2,
                    solid_capstyle="round")
            ax.plot([d], [y], "o", color=COLOUR[mode], ms=6,
                    markeredgecolor="white", markeredgewidth=0.8)
            sig = "" if lo > 0 or hi < 0 else "  (n.s.)"
            ax.text(hi + 0.004, y, f"{d:+.3f}{sig}", va="center", fontsize=8,
                    color="#333333")
            yticks.append(y)
            yticklabels.append(f"{lab} — {MODE_LABEL[mode]}")
            y += 1
        y += 0.4

    ax.axvline(0, color="#666666", lw=1, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUC vs lexical node features (95% cluster-bootstrap CI)")
    ax.set_title("Representation effect across corpora and backbones")
    ax.grid(axis="x", ls=":", alpha=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    out = FIGURES / "fig_representation_forest.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out.name} ({len(yticks)} cells)")


def redundancy():
    """Why the taxonomy adds nothing: phi is nearly determined by kind."""
    info = load("information_analysis.json")
    if not info:
        print("  no information_analysis.json; run analyze_information.py")
        return

    names, hcond, det = [], [], []
    for cache, r in info.items():
        label = cache.split("_")[0].replace("hf", "").strip("_").capitalize()
        if label in names:
            continue
        names.append(label)
        hcond.append(r["H_phi_given_kind"])
        det.append(100 * r["determinism"])
    if not names:
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(names))

    a1.bar(x, hcond, 0.5, color="#2c7fb8", edgecolor="black", lw=0.6)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("bits")
    a1.set_title(r"$H(\phi \mid \kappa)$: information in the"
                 "\ntaxonomy beyond the grammar kind")
    for xi, v in zip(x, hcond):
        a1.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a1.set_ylim(0, max(hcond) * 1.6)

    a2.bar(x, det, 0.5, color="#41ab5d", edgecolor="black", lw=0.6)
    a2.axhline(100, color="#666666", ls="--", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("% of nodes")
    a2.set_title("Operation label predictable\nfrom grammar kind alone")
    for xi, v in zip(x, det):
        a2.text(xi, v - 6, f"{v:.1f}%", ha="center", fontsize=9, color="white")
    a2.set_ylim(0, 108)

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.5)

    out = FIGURES / "fig_redundancy.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    print("Generating figures from v2 results:")
    forest()
    redundancy()
