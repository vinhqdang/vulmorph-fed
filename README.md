# VulMorph-Fed

> **Cross-Project Software Vulnerability Detection via Federated Vulnerability Morphology Learning**
>
> Research prototype — targeting *Information and Software Technology* (IST), Elsevier.

---

## Overview

**VulMorph-Fed** is a privacy-preserving federated learning framework for detecting software vulnerabilities across heterogeneous projects. It introduces three chained innovations:

| Component | What it does |
|---|---|
| **VCSA** (Vulnerability-Critical Subgraph Abstraction) | Differentiable edge-masking MLP that isolates vulnerability-critical subgraphs and applies 8-type morphological abstraction to eliminate project-specific lexical bias. |
| **MCFPA** (Morphology-Conditioned Federated Prototype Aggregation) | CWE-affinity-weighted server aggregation of local prototypes; Laplace DP applied before upload. |
| **MGMP** (Morphology-Guided Message Passing) | Custom PyG `MessagePassing` layer that injects global prototype attention into local graph convolution. |

---

## Project Structure

```
vulmorph-fed/
├── data/
│   ├── morphology.py          # 8-type abstract morphology taxonomy + embedding
│   ├── dataset.py             # Mock CPG dataset for testing
│   └── loaders/
│       └── diversevul_loader.py  # DiverseVul JSONL graph loader
├── models/
│   ├── vcsa.py                # VCSA edge masking + L_SCL
│   ├── mgmp.py                # MGMP message-passing layer
│   ├── vulmorph.py            # Full local client model
│   └── baselines/
│       ├── gnn_baselines.py   # Devign, GAT baselines
│       └── nlp_baselines.py   # CodeBERTSimple baseline
├── fl/
│   ├── client.py              # VulMorphClient (local training + prototype upload)
│   ├── server.py              # VulMorphServer (MCFPA aggregation)
│   └── baselines/
│       └── fedavg.py          # Standard FedAvg weight-averaging baseline
├── utils/
│   ├── metrics.py             # F1, AUC, Paired Accuracy, CP-F1
│   ├── privacy.py             # Laplace DP for prototypes
│   └── stats.py               # Wilcoxon signed-rank + Cliff's delta
├── experiments/
│   └── run_ablations.py       # Full ablation study runner (Table 5.5)
├── main.py                    # Federated training simulation entry point
└── requirements.txt
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Full VulMorph-Fed (simulated data)

```bash
python main.py --num_clients 5 --rounds 10 --local_epochs 2 --epsilon 2.0
```

### Run All Ablation Variants

```bash
python experiments/run_ablations.py --num_clients 3 --rounds 5
```

Results are saved to `results/ablations.json`.

### Individual Ablation Flags

```bash
# w/o VCSA
python main.py --no_vcsa

# w/o Morphological Abstraction
python main.py --no_morphology

# w/o MCFPA (uniform averaging)
python main.py --no_cwe_affinity

# w/o MGMP (standard GAT)
python main.py --no_mgmp

# w/o Differential Privacy
python main.py --no_dp

# Local Only (no federation)
python main.py --local_only

# Privacy sweep
python main.py --epsilon 0.1
python main.py --epsilon 1.0
python main.py --epsilon 5.0
python main.py --epsilon inf   # no DP
```

---

## Real-World Data

The `data/loaders/diversevul_loader.py` loader expects JSONL files where each line is:

```json
{
  "nodes": [{"label": "CallExpression", "token": "malloc"}, ...],
  "edges": [[0, 1], [1, 2], ...],
  "target": 1,
  "cwe": 122
}
```

Download DiverseVul from https://github.com/wagner-group/diversevul and pre-process using Joern or a tree-sitter parser to produce this format.

---

## Datasets (from plan.md §4)

| Dataset | Scale | CWEs | Role |
|---|---|---|---|
| Devign | 27K functions | — | GNN baseline; within-project eval |
| BigVul | 188K functions | 91 | Large-scale training |
| DiverseVul | 349K functions | 150 | **Primary** cross-project FL simulation |
| CVEfixes | 11K CVEs | 272 | Multi-language; prototype construction |
| PrimeVul | 236K functions | — | Strictest benchmark; temporal split |
| MegaVul | BigVul superset | — | Pre-extracted graphs |

---

## Baselines

| Baseline | Category | Location |
|---|---|---|
| Devign (GGNN) | Centralised GNN | `models/baselines/gnn_baselines.py` |
| GAT (CPVD-style) | Centralised GNN | `models/baselines/gnn_baselines.py` |
| CodeBERTSimple | Centralised LLM | `models/baselines/nlp_baselines.py` |
| FedAvg + GAT | Federated | `fl/baselines/fedavg.py` |

---

## Citation

> VulMorph-Fed: Cross-Project Software Vulnerability Detection via Federated Vulnerability Morphology Learning. (Manuscript in preparation.)
