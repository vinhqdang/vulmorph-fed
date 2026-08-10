"""
Figure generator for the VulMorph-Fed manuscript.

Consumes the multi-seed result JSONs (schema: {metric: {mean, std, values}})
and renders figures with error bars. Figures are only produced when the
underlying result files exist — nothing is fabricated.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'figure.autolayout': True,
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["devign", "bigvul", "diversevul", "primevul"]
DATASET_LABELS = {"devign": "Devign", "bigvul": "BigVul",
                  "diversevul": "DiverseVul", "primevul": "PrimeVul"}


def load_json(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def ms(entry, metric):
    m = entry.get(metric) if entry else None
    if isinstance(m, dict) and "mean" in m:
        return m["mean"], m["std"]
    return None, None


def plot_rq1():
    methods = [
        ("centralised_ggnn",        "Cent. GGNN"),
        ("centralised_gat",         "Cent. GAT"),
        ("centralised_transformer", "Cent. Transf."),
        ("fedavg_gat",              "FedAvg+GAT"),
        ("fedavg_ggnn_morph",       "FedAvg+GGNN (VCSA)"),
        ("fedavg_transformer",      "FedAvg+Transf."),
    ]

    ds_used, series = [], {label: ([], []) for _, label in methods}
    ours_means, ours_stds = [], []
    for ds in DATASETS:
        base = load_json(f"baselines_{ds}.json")
        ours = load_json(f"vulmorph_{ds}.json")
        if not base or not ours:
            continue
        ds_used.append(ds)
        for key, label in methods:
            m, s = ms(base.get(key), "f1")
            series[label][0].append(m if m is not None else np.nan)
            series[label][1].append(s if s is not None else 0)
        m, s = ms(ours, "f1")
        ours_means.append(m)
        ours_stds.append(s)

    if not ds_used:
        print("Missing data for RQ1 plot")
        return

    x = np.arange(len(ds_used))
    n_bars = len(methods) + 1
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("Greys")
    for i, (label, (means, stds)) in enumerate(series.items()):
        ax.bar(x + (i - n_bars / 2) * width + width / 2, means, width,
               yerr=stds, capsize=2, label=label,
               color=cmap(0.25 + 0.08 * i), edgecolor='black', linewidth=0.5)
    ax.bar(x + (len(methods) - n_bars / 2) * width + width / 2,
           ours_means, width, yerr=ours_stds, capsize=2,
           label='VulMorph-Fed (ours)', color='#2c7fb8', edgecolor='black')

    ax.set_ylabel('Cross-Project F1-Score')
    ax.set_title('RQ1: Cross-Project Vulnerability Detection (mean ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in ds_used])
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(FIGURES_DIR / "rq1_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved rq1_performance.png")


def plot_rq3():
    data = load_json("rq3_privacy.json")
    if not data:
        print("Missing data for RQ3 plot")
        return

    eps_labels, means, stds = [], [], []
    for k in ['0.1', '0.5', '1.0', '2.0', '5.0', 'inf']:
        if k in data:
            m, s = ms(data[k], "f1")
            if m is None:
                continue
            eps_labels.append('No DP' if k == 'inf' else k)
            means.append(m)
            stds.append(s)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(eps_labels, means, yerr=stds, marker='o', linewidth=2,
                markersize=8, capsize=4, color='#d95f02')
    ax.set_xlabel(r'Per-round Privacy Budget ($\varepsilon$)')
    ax.set_ylabel('F1-Score')
    ax.set_title('RQ3: Privacy-Utility Trade-off (mean ± std)')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "rq3_privacy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved rq3_privacy.png")


def plot_rq4():
    data = load_json("rq4_scalability.json")
    if not data:
        print("Missing data for RQ4 plot")
        return

    clients, means, stds, ccr_total = [], [], [], []
    for k in sorted(data.keys(), key=int):
        m, s = ms(data[k], "f1")
        if m is None:
            continue
        clients.append(int(k))
        means.append(m)
        stds.append(s)
        ccr_total.append(data[k].get('ccr_server_total_kb', 0))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#1b9e77'
    ax1.set_xlabel('Number of Clients ($K$)')
    ax1.set_ylabel('F1-Score', color=color)
    l1 = ax1.errorbar(clients, means, yerr=stds, marker='s', linewidth=2,
                      markersize=8, capsize=4, color=color, label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(clients)

    ax2 = ax1.twinx()
    color = '#7570b3'
    ax2.set_ylabel('Total server traffic per round (KB)', color=color)
    l2 = ax2.plot(clients, ccr_total, marker='^', linewidth=2, markersize=8,
                  color=color, linestyle='--', label='Server traffic')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(ccr_total) * 1.2 if ccr_total else 1)

    lines = [l1, l2[0]]
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    plt.title('RQ4: Scalability and Communication Cost (mean ± std)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "rq4_scalability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved rq4_scalability.png")


if __name__ == "__main__":
    print("Generating plots...")
    plot_rq1()
    plot_rq3()
    plot_rq4()
    print("Done.")
