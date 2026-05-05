import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style for academic plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.autolayout': True
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_json(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def plot_rq1():
    base_data = load_json("baselines.json")
    prop_data = load_json("devign_real.json")
    
    if not base_data or not prop_data:
        print("Missing data for RQ1 plot")
        return
        
    labels = ['Devign (Cent.)', 'GAT (Cent.)', 'FedAvg+GAT', 'VulMorph-Fed']
    f1_scores = [
        base_data.get('centralised_devign', {}).get('f1', 0),
        base_data.get('centralised_gat', {}).get('f1', 0),
        base_data.get('fedavg_gat', {}).get('f1', 0),
        prop_data.get('f1', 0)
    ]
    
    x = np.arange(len(labels))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#cccccc', '#cccccc', '#999999', '#2c7fb8']
    bars = ax.bar(x, f1_scores, width, color=colors, edgecolor='black')
    
    ax.set_ylabel('Cross-Project F1-Score')
    ax.set_title('RQ1: Cross-Project Vulnerability Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(f1_scores) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "rq1_performance.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved rq1_performance.pdf")

def plot_rq3():
    data = load_json("rq3_privacy.json")
    if not data:
        print("Missing data for RQ3 plot")
        return
        
    # epsilons: '0.1', '0.5', '1.0', '2.0', '5.0', 'inf'
    eps_vals = []
    f1_scores = []
    
    for k in ['0.1', '0.5', '1.0', '2.0', '5.0', 'inf']:
        if k in data:
            eps_vals.append('No DP' if k == 'inf' else k)
            f1_scores.append(data[k].get('f1', 0))
            
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_vals, f1_scores, marker='o', linewidth=2, markersize=8, color='#d95f02')
    
    ax.set_xlabel('Privacy Budget ($\epsilon$)')
    ax.set_ylabel('F1-Score')
    ax.set_title('RQ3: Privacy-Utility Tradeoff')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(max(0, min(f1_scores) - 0.1), max(f1_scores) + 0.05)
    
    for i, txt in enumerate(f1_scores):
        ax.annotate(f'{txt:.4f}', (eps_vals[i], f1_scores[i]), 
                    xytext=(0, 10), textcoords='offset points', ha='center')
                    
    plt.savefig(FIGURES_DIR / "rq3_privacy.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved rq3_privacy.pdf")

def plot_rq4():
    data = load_json("rq4_scalability.json")
    if not data:
        print("Missing data for RQ4 plot")
        return
        
    clients = []
    f1_scores = []
    ccr = []
    
    for k in sorted(data.keys(), key=int):
        clients.append(int(k))
        f1_scores.append(data[k].get('f1', 0))
        ccr.append(data[k].get('ccr_kb', 0))
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = '#1b9e77'
    ax1.set_xlabel('Number of Clients ($K$)')
    ax1.set_ylabel('F1-Score', color=color)
    line1 = ax1.plot(clients, f1_scores, marker='s', linewidth=2, markersize=8, color=color, label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(clients)
    ax1.set_ylim(max(0, min(f1_scores) - 0.05), max(f1_scores) + 0.05)
    
    ax2 = ax1.twinx()  
    color = '#7570b3'
    ax2.set_ylabel('Comm. Cost per Round (KB)', color=color)  
    line2 = ax2.plot(clients, ccr, marker='^', linewidth=2, markersize=8, color=color, linestyle='--', label='Comm. Cost')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(ccr) * 1.2)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('RQ4: Scalability and Communication Cost')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(FIGURES_DIR / "rq4_scalability.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved rq4_scalability.pdf")

if __name__ == "__main__":
    print("Generating plots...")
    plot_rq1()
    plot_rq3()
    plot_rq4()
    print("All plots generated successfully.")
