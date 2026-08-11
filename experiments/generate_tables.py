"""
LaTeX Table Generator for the VulMorph-Fed manuscript.

Reads the multi-seed result JSONs produced by the experiment runners
(schema: {metric: {"mean": .., "std": .., "values": [..]}}) and produces
tables reporting mean ± std. Missing metrics are rendered as "--", never
as fabricated zeros. Statistical tests (Wilcoxon signed-rank + Cliff's
delta) pair per-dataset, per-seed F1 values of VulMorph-Fed against each
baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from utils.stats import run_statistical_tests

RESULTS_DIR = Path(__file__).parent / "results"

DATASETS = ["devign", "bigvul", "diversevul", "primevul"]
DATASET_LABELS = {"devign": "Devign", "bigvul": "BigVul",
                  "diversevul": "DiverseVul", "primevul": "PrimeVul"}

# RQ1 rows are grouped by privacy class: methods with a formal per-round
# ε-DP guarantee are the peers of VulMorph-Fed; everything that pools raw
# code or transmits parameters in the clear is a non-private reference.
NONPRIVATE_LABELS = {
    "centralised_ggnn":        "Centralised GGNN (lexical graphs)",
    "centralised_gat":         "Centralised GAT (lexical graphs)",
    "centralised_transformer": "Centralised Transformer (sequence)",
    "centralised_ggnn_morph":  "Centralised GGNN (VCSA graphs)",
    "centralised_vulmorph":    "Centralised VulMorph (oracle)",
    "fedavg_gat":              "FedAvg + GAT (lexical graphs)",
    "fedavg_ggnn_morph":       "FedAvg + GGNN (VCSA graphs)",
    "fedavg_transformer":      "FedAvg + Transformer (sequence)",
}
PRIVATE_LABELS = {
    "dp_fedavg_gat":           "DP-FedAvg + GAT (lexical graphs)",
    "dp_fedavg_ggnn_morph":    "DP-FedAvg + GGNN (VCSA graphs)",
}
BASELINE_LABELS = {**NONPRIVATE_LABELS, **PRIVATE_LABELS}


def load_json(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt_ms(entry, metric, bold=False, prec=3):
    """Format a metric as mean±std; '--' when absent."""
    m = entry.get(metric) if entry else None
    if not isinstance(m, dict) or "mean" not in m:
        return "--"
    s = f"{m['mean']:.{prec}f}$\\pm${m['std']:.{prec}f}"
    return f"\\textbf{{{s}}}" if bold else s


def mean_of(entry, metric):
    m = entry.get(metric) if entry else None
    return m["mean"] if isinstance(m, dict) and "mean" in m else None


def values_of(entry, metric):
    m = entry.get(metric) if entry else None
    return m.get("values", []) if isinstance(m, dict) else []


# ── Table 1: RQ1 Main Comparison ─────────────────────────────────────────────

def table_rq1():
    per_ds = {}
    for ds in DATASETS:
        base = load_json(f"baselines_{ds}.json")
        ours = load_json(f"vulmorph_{ds}.json")
        if base or ours:
            per_ds[ds] = (base or {}, ours)

    if not per_ds:
        print("Warning: no RQ1 results found, skipping Table 1.")
        return ""

    ds_used = list(per_ds.keys())

    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{RQ1: Cross-project vulnerability detection F1-score "
        "(mean $\\pm$ std over " + _seed_note(per_ds, ds_used) + " seeds). "
        "Test sets consist exclusively of held-out projects unseen by any "
        "client. The upper block gives non-private references (raw code "
        "pooled, or parameters exchanged in the clear); the lower block "
        "compares methods with a formal per-round $\\varepsilon$-DP "
        "guarantee at the same budget ($\\varepsilon = 2$/round). Best "
        "result within the privacy-preserving block in bold.}",
        "\\label{tab:rq1_main}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(ds_used) + "}",
        "\\toprule",
        "Model & " + " & ".join(DATASET_LABELS[d] for d in ds_used) + " \\\\",
        "        & " + " & ".join("F1 / AUC" for _ in ds_used) + " \\\\",
        "\\midrule",
        "\\multicolumn{" + str(len(ds_used) + 1) +
        "}{l}{\\emph{Non-private references}} \\\\",
    ]

    for m in NONPRIVATE_LABELS:
        if not any(m in per_ds[ds][0] for ds in ds_used):
            continue
        row = [NONPRIVATE_LABELS[m]]
        for ds in ds_used:
            e = per_ds[ds][0].get(m)
            row.append(fmt_ms(e, "f1") + " / " + fmt_ms(e, "auc"))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(ds_used) + 1) +
                 "}{l}{\\emph{Formally private ($\\varepsilon$-DP per round)}} \\\\")

    for m in PRIVATE_LABELS:
        if not any(m in per_ds[ds][0] for ds in ds_used):
            continue
        row = [PRIVATE_LABELS[m]]
        for ds in ds_used:
            e = per_ds[ds][0].get(m)
            row.append(fmt_ms(e, "f1") + " / " + fmt_ms(e, "auc"))
        lines.append(" & ".join(row) + " \\\\")

    # Bold only where our method actually leads its privacy class.
    row = ["\\textbf{VulMorph-Fed (ours)}"]
    for ds in ds_used:
        ours = per_ds[ds][1]
        peers = [per_ds[ds][0].get(k) for k in PRIVATE_LABELS
                 if per_ds[ds][0].get(k)]
        best_peer_f1 = max([mean_of(p, "f1") or 0 for p in peers], default=0)
        best_peer_auc = max([mean_of(p, "auc") or 0 for p in peers], default=0)
        win_f1 = (mean_of(ours, "f1") or 0) >= best_peer_f1
        win_auc = (mean_of(ours, "auc") or 0) >= best_peer_auc
        row.append(fmt_ms(ours, "f1", bold=win_f1) + " / "
                   + fmt_ms(ours, "auc", bold=win_auc))
    lines.append(" & ".join(row) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"]
    return "\n".join(lines)


def _seed_note(per_ds, ds_used):
    for ds in ds_used:
        ours = per_ds[ds][1]
        if ours and "num_seeds" in ours:
            return str(ours["num_seeds"])
    return "?"


def significance_summary():
    """Pairs per-dataset per-seed F1 of VulMorph-Fed vs each baseline."""
    ours_vals, base_vals = {}, {}
    for ds in DATASETS:
        ours = load_json(f"vulmorph_{ds}.json")
        base = load_json(f"baselines_{ds}.json")
        if not ours or not base:
            continue
        ov = values_of(ours, "f1")
        for m, entry in base.items():
            bv = values_of(entry, "f1")
            if len(bv) == len(ov) and ov:
                ours_vals.setdefault(m, []).extend(ov)
                base_vals.setdefault(m, []).extend(bv)

    lines = ["% Statistical comparison: VulMorph-Fed vs each baseline",
             "% (Wilcoxon signed-rank over paired per-dataset, per-seed F1)"]
    stats = {}
    for m in ours_vals:
        r = run_statistical_tests(ours_vals[m], base_vals[m])
        stats[m] = {**r, "n_pairs": len(ours_vals[m])}
        lines.append(f"%   vs {m}: n={len(ours_vals[m])}, "
                     f"p={r['wilcoxon_p']:.4f}, delta={r['cliffs_delta']:.3f}")

    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump(stats, f, indent=2)
    return "\n".join(lines)


# ── Table 2: RQ2 Ablation ─────────────────────────────────────────────────────

VARIANT_ORDER = [
    ("Full VulMorph-Fed",             "\\textbf{Full VulMorph-Fed (proposed)}"),
    ("w/o VCSA",                      "w/o VCSA"),
    ("w/o Morphological Abstraction", "w/o Morph. Abstraction"),
    ("w/o MCFPA (Uniform Avg)",       "w/o MCFPA (uniform avg.)"),
    ("w/o MGMP (Standard GAT)",       "w/o MGMP (standard GAT)"),
    ("w/o DP",                        "w/o Differential Privacy"),
    ("Local Only",                    "Local only (no federation)"),
]

METRIC_COLS = [("f1", "F1"), ("auc", "AUC"),
               ("precision", "Prec."), ("recall", "Rec.")]


def _metric_table(data, order, caption, label, first_col="Variant"):
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(METRIC_COLS) + "}",
        "\\toprule",
        first_col + " & " + " & ".join(l for _, l in METRIC_COLS) + " \\\\",
        "\\midrule",
    ]
    for key, display in order:
        entry = data.get(key)
        if entry is None:
            continue
        cells = " & ".join(fmt_ms(entry, c) for c, _ in METRIC_COLS)
        lines.append(f"{display} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def table_rq2():
    data = load_json("ablations.json")
    if not data:
        print("Warning: ablations.json not found, skipping Table 2.")
        return ""
    return _metric_table(
        data, VARIANT_ORDER,
        "RQ2: Ablation study on BigVul (mean $\\pm$ std over seeds). "
        "Each row removes one VulMorph-Fed component.",
        "tab:rq2_ablation")


def table_rq2b():
    data = load_json("taxonomy_size.json")
    if not data:
        print("Warning: taxonomy_size.json not found, skipping Table 2b.")
        return ""
    order = [(k, f"$|\\mathcal{{T}}| = {k}$") for k in ["8", "16", "32"]]
    return _metric_table(
        data, order,
        "RQ2b: Sensitivity to the morphological taxonomy size "
        "$|\\mathcal{T}|$ (mean $\\pm$ std over seeds).",
        "tab:rq2b_taxonomy", first_col="Taxonomy")


# ── Table 3: RQ3 Privacy-Utility ─────────────────────────────────────────────

def table_rq3():
    data = load_json("rq3_privacy.json")
    if not data:
        print("Warning: rq3_privacy.json not found, skipping Table 3.")
        return ""

    eps_display = {
        "0.1": "0.1 (strong)", "0.5": "0.5", "1.0": "1.0",
        "2.0": "2.0", "5.0": "5.0", "inf": "$\\infty$ (no DP)",
    }
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{RQ3: Privacy-utility trade-off across per-round Laplace "
        "DP budgets $\\varepsilon$ (mean $\\pm$ std over seeds). "
        "$\\varepsilon_{\\mathrm{tot}}$ is the end-to-end budget after "
        "sequential composition over $T$ rounds.}",
        "\\label{tab:rq3_privacy}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "$\\varepsilon$/round & $\\varepsilon_{\\mathrm{tot}}$ & "
        + " & ".join(l for _, l in METRIC_COLS) + " \\\\",
        "\\midrule",
    ]
    for eps_key, display in eps_display.items():
        entry = data.get(eps_key)
        if entry is None:
            continue
        comp = entry.get("epsilon_composed", "--")
        comp_s = "$\\infty$" if comp == "inf" else f"{comp:g}"
        cells = " & ".join(fmt_ms(entry, c) for c, _ in METRIC_COLS)
        lines.append(f"{display} & {comp_s} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


# ── Table 4: RQ4 Scalability ──────────────────────────────────────────────────

def table_rq4():
    data = load_json("rq4_scalability.json")
    if not data:
        print("Warning: rq4_scalability.json not found, skipping Table 4.")
        return ""

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{RQ4: Scalability across $K$ federated clients "
        "(mean $\\pm$ std over seeds). Per-client communication cost per "
        "round (upload + download of one $|\\mathcal{C}| \\times d$ "
        "prototype bank) is constant in $K$; total server traffic grows "
        "linearly.}",
        "\\label{tab:rq4_scalability}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$K$ & F1 & AUC & Client CCR (KB) & Server total (KB) \\\\",
        "\\midrule",
    ]
    for k in sorted(data.keys(), key=int):
        m = data[k]
        lines.append(
            f"{k} & {fmt_ms(m, 'f1')} & {fmt_ms(m, 'auc')} & "
            f"{m.get('ccr_client_kb', '--')} & "
            f"{m.get('ccr_server_total_kb', '--')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


# ── Table 5: RQ1b Public-only vs Public+Federated ────────────────────────────

def table_rq1b():
    data = load_json("public_vs_fed.json")
    if not data:
        print("Warning: public_vs_fed.json not found, skipping Table 5.")
        return ""
    rows = [
        ("public_only_f1", "Centralised, public data only"),
        ("public_fed_f1",  "Public + DP-federated private clients"),
        ("delta_f1",       "$\\Delta$ (value of federation)"),
    ]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{RQ1b: What do private federated clients add on top of a "
        "public corpus? Cross-project F1 (mean $\\pm$ std over seeds).}",
        "\\label{tab:rq1b_public}",
        "\\begin{tabular}{lr}",
        "\\toprule", "Configuration & F1 \\\\", "\\midrule",
    ]
    for key, display in rows:
        m = data.get(key)
        if not isinstance(m, dict):
            continue
        lines.append(f"{display} & {m['mean']:.3f}$\\pm${m['std']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate LaTeX tables for manuscript")
    p.add_argument("--output", type=str, default="results/tables.tex")
    args = p.parse_args()

    sections = [
        ("RQ1 Main Comparison", table_rq1()),
        ("RQ1 Significance", significance_summary()),
        ("RQ1b Public vs Federated", table_rq1b()),
        ("RQ2 Ablation Study", table_rq2()),
        ("RQ2b Taxonomy Size", table_rq2b()),
        ("RQ3 Privacy-Utility", table_rq3()),
        ("RQ4 Scalability", table_rq4()),
    ]

    out = Path(__file__).parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("% ===================================================\n")
        f.write("% VulMorph-Fed — Auto-generated LaTeX Tables\n")
        f.write("% Generated by experiments/generate_tables.py\n")
        f.write("% All values are computed from the JSON files in\n")
        f.write("% experiments/results/ produced by the experiment runners.\n")
        f.write("% ===================================================\n\n")
        for label, tbl in sections:
            if tbl:
                f.write(f"% --- {label} ---\n")
                f.write(tbl + "\n\n")

    print(f"\nLaTeX tables saved → {out}")


if __name__ == "__main__":
    main()
