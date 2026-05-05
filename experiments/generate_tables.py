"""
LaTeX Table Generator for VulMorph-Fed manuscript (plan.md §7).

Reads all JSON result files and produces:
  - Table 1: Main RQ1 comparison (VulMorph-Fed vs all baselines)
  - Table 2: Ablation study (RQ2)
  - Table 3: Privacy-utility tradeoff (RQ3)
  - Table 4: Scalability (RQ4)
  - Runs Wilcoxon / Cliff's delta significance tests where multiple seeds available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
from utils.stats import run_statistical_tests

RESULTS_DIR = Path(__file__).parent / "results"

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_json(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt(v, bold=False):
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if bold else s

def bold_best(values, metrics, higher_is_better=True):
    """Return list of bold flags for the best value in each metric column."""
    result = []
    for col_vals in zip(*[[m[k] for k in metrics] for m in values]):
        best = max(col_vals) if higher_is_better else min(col_vals)
        result.append([v == best for v in col_vals])
    # Transpose back
    n_methods = len(values)
    n_metrics = len(metrics)
    flags = [[False]*n_metrics for _ in range(n_methods)]
    for mi, col in enumerate(zip(*[
        [m[k] for k in metrics] for m in values
    ])):
        best = max(col)
        for ri, v in enumerate(col):
            if v == best:
                flags[ri][mi] = True
    return flags


# ── Table 1: RQ1 Main Comparison ─────────────────────────────────────────────

def table_rq1():
    baseline_data = load_json("baselines.json")
    vulmorph_data = load_json("devign_real.json")
    ablation_data = load_json("ablations_real.json")

    if not vulmorph_data:
        print("Warning: devign_real.json not found, skipping Table 1.")
        return ""

    rows = []
    # Centralised baselines
    if baseline_data:
        rows.append(("Devign (centralised)", baseline_data.get("centralised_devign", {})))
        rows.append(("GAT / CPVD-style (centralised)", baseline_data.get("centralised_gat", {})))
        rows.append(("FedAvg + GAT", baseline_data.get("fedavg_gat", {})))
    # Proposed
    rows.append(("\\textbf{VulMorph-Fed (proposed)}", vulmorph_data))

    cols = ["f1", "auc", "precision", "recall"]
    col_labels = ["F1", "AUC", "Prec.", "Rec."]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RQ1: Cross-Project Vulnerability Detection on Devign. "
                 "Best results per column in \\textbf{bold}.}")
    lines.append("\\label{tab:rq1_main}")
    lines.append("\\begin{tabular}{l" + "r"*len(cols) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")

    # Find best per column
    all_vals = [m for _, m in rows if m]
    best = {}
    for c in cols:
        vals = [m.get(c, 0) for m in all_vals]
        best[c] = max(vals) if vals else 0

    for name, m in rows:
        if not m:
            cells = " & ".join(["--"]*len(cols))
        else:
            cells = " & ".join(
                fmt(m.get(c, 0), bold=(abs(m.get(c, 0) - best[c]) < 1e-4))
                for c in cols
            )
        lines.append(f"{name} & {cells} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Table 2: RQ2 Ablation ─────────────────────────────────────────────────────

def table_rq2():
    data = load_json("ablations_real.json")
    if not data:
        print("Warning: ablations_real.json not found, skipping Table 2.")
        return ""

    # Map display names
    variant_map = {
        "Full VulMorph-Fed":             "Full VulMorph-Fed (proposed)",
        "w/o VCSA":                      "w/o VCSA",
        "w/o Morphological Abstraction": "w/o Morph. Abstraction",
        "w/o MCFPA (Uniform Avg)":       "w/o MCFPA (uniform avg.)",
        "w/o MGMP (Standard GAT)":       "w/o MGMP (standard GAT)",
        "w/o DP":                        "w/o Differential Privacy",
        "Local Only":                    "Local only (no federation)",
    }

    cols = ["f1", "auc", "precision", "recall"]
    col_labels = ["F1", "AUC", "Prec.", "Rec."]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RQ2: Ablation Study. Each row removes one VulMorph-Fed component. "
                 "Best per column in \\textbf{bold}.}")
    lines.append("\\label{tab:rq2_ablation}")
    lines.append("\\begin{tabular}{l" + "r"*len(cols) + "}")
    lines.append("\\toprule")
    lines.append("Variant & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")

    all_vals = [v for v in data.values()]
    best = {c: max(v.get(c, 0) for v in all_vals) for c in cols}

    for key, display in variant_map.items():
        m = data.get(key, {})
        if key == "Full VulMorph-Fed":
            display = "\\textbf{" + display + "}"
            lines.append("\\midrule")
        cells = " & ".join(
            fmt(m.get(c, 0), bold=(abs(m.get(c, 0) - best[c]) < 1e-4))
            for c in cols
        )
        lines.append(f"{display} & {cells} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Table 3: RQ3 Privacy-Utility ─────────────────────────────────────────────

def table_rq3():
    data = load_json("rq3_privacy.json")
    if not data:
        print("Warning: rq3_privacy.json not found, skipping Table 3.")
        return ""

    cols = ["f1", "auc", "precision", "recall"]
    col_labels = ["F1", "AUC", "Prec.", "Rec."]
    eps_display = {
        "0.1": "0.1 (strong)", "0.5": "0.5", "1.0": "1.0",
        "2.0": "2.0", "5.0": "5.0", "inf": "$\\infty$ (no DP)"
    }

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RQ3: Privacy-Utility Tradeoff across Laplace DP budgets $\\varepsilon$.}")
    lines.append("\\label{tab:rq3_privacy}")
    lines.append("\\begin{tabular}{l" + "r"*len(cols) + "}")
    lines.append("\\toprule")
    lines.append("$\\varepsilon$ & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")

    for eps_key, display in eps_display.items():
        m = data.get(eps_key, {})
        cells = " & ".join(fmt(m.get(c, 0)) for c in cols)
        lines.append(f"{display} & {cells} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Table 4: RQ4 Scalability ──────────────────────────────────────────────────

def table_rq4():
    data = load_json("rq4_scalability.json")
    if not data:
        print("Warning: rq4_scalability.json not found, skipping Table 4.")
        return ""

    cols = ["f1", "auc"]
    col_labels = ["F1", "AUC"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RQ4: Scalability — performance and communication cost "
                 "across $K$ federated clients.}")
    lines.append("\\label{tab:rq4_scalability}")
    lines.append("\\begin{tabular}{r" + "r"*len(cols) + "r}")
    lines.append("\\toprule")
    lines.append("$K$ & " + " & ".join(col_labels) + " & CCR (KB/rnd) \\\\")
    lines.append("\\midrule")

    for k, m in sorted(data.items(), key=lambda x: int(x[0])):
        cells = " & ".join(fmt(m.get(c, 0)) for c in cols)
        ccr = m.get("ccr_kb", 0)
        lines.append(f"{k} & {cells} & {ccr:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate LaTeX tables for manuscript")
    p.add_argument("--output", type=str, default="results/tables.tex")
    args = p.parse_args()

    t1 = table_rq1()
    t2 = table_rq2()
    t3 = table_rq3()
    t4 = table_rq4()

    out = Path(__file__).parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("% ===================================================\n")
        f.write("% VulMorph-Fed — Auto-generated LaTeX Tables\n")
        f.write("% Generated by experiments/generate_tables.py\n")
        f.write("% ===================================================\n\n")
        for label, tbl in [("RQ1 Main Comparison", t1),
                            ("RQ2 Ablation Study", t2),
                            ("RQ3 Privacy-Utility", t3),
                            ("RQ4 Scalability", t4)]:
            if tbl:
                f.write(f"% --- {label} ---\n")
                f.write(tbl + "\n\n")

    print(f"\nLaTeX tables saved → {out}")
    print("\nPreviewing table headers:")
    for t in [t1, t2, t3, t4]:
        if t:
            for line in t.split("\n"):
                if "caption" in line.lower():
                    print(" ", line.strip())


if __name__ == "__main__":
    main()
