# VulMorph-Fed

> **Cross-Project Software Vulnerability Detection via Federated Vulnerability Morphology Learning**
>
> Research prototype with a fully reproducible experiment pipeline: every table
> and figure in the manuscript is generated programmatically from the result
> files produced by `experiments/run_all.sh`. No number is entered by hand.

---

## Overview

**VulMorph-Fed** is a privacy-preserving federated learning framework for
detecting software vulnerabilities across heterogeneous projects. It has three
components:

| Component | What it does |
|---|---|
| **VCSA** (Vulnerability-Critical Subgraph Abstraction) | Deterministic, context-aware mapping of code tokens onto a small semantic taxonomy (8/16/32 types + `UNKNOWN`), plus a learned soft edge-importance mask. Node features contain **no project-specific tokens**. |
| **MCFPA** (Morphology-Conditioned Federated Prototype Aggregation) | Clients share CWE-conditioned prototype vectors instead of model parameters. Per-sample embeddings are L1-clipped (radius `R`) and each prototype row is perturbed with a per-class-calibrated Laplace mechanism → provable per-round ε-DP with sequential composition over rounds (ε_tot = T·ε). |
| **MGMP** (Morphology-Guided Message Passing) | Custom PyG layer fusing edge-weighted local aggregation with scaled-dot-product attention over the global prototype bank, gated per node by its morphological type. |

**Inference protocol**: the deployed detector is the uniform probability
ensemble of the K client models conditioned on the final global prototype bank.

---

## Project Structure

```
vulmorph-fed/
├── data/
│   ├── morphology.py             # Nested 8/16/32-type taxonomies + embedding
│   └── loaders/real_datasets.py  # Dataset loaders, phi_type rules, cross-project split
├── models/
│   ├── vcsa.py                   # Edge-mask MLP + structural contrastive loss
│   ├── mgmp.py                   # MGMP message-passing layer
│   ├── vulmorph.py               # Full local client model (morphology-only features)
│   └── baselines/                # GGNN/GAT (lex or morph input), Transformer-seq
├── fl/
│   ├── client.py                 # Local training, clipped prototypes, calibrated DP
│   └── server.py                 # MCFPA affinity-weighted aggregation
├── utils/
│   ├── metrics.py                # F1/AUC/P/R (fixed threshold 0.5)
│   ├── privacy.py                # L1 clipping, calibrated Laplace, composition
│   └── stats.py                  # Wilcoxon signed-rank + Cliff's delta
├── experiments/
│   ├── run_all.sh                # ← regenerates EVERYTHING (tables + figures)
│   ├── run_main.py               # Full model per dataset (RQ1)
│   ├── run_baselines.py          # 8-baseline suite (RQ1)
│   ├── run_public_vs_fed.py      # Public-only vs public+federated (RQ1b)
│   ├── run_ablations.py          # Component ablations (RQ2)
│   ├── run_taxonomy.py           # |T| ∈ {8,16,32} sensitivity (RQ2b)
│   ├── run_rq3_rq4.py            # Privacy sweep + scalability (RQ3/RQ4)
│   ├── generate_tables.py        # JSON results → LaTeX tables (mean ± std)
│   └── generate_plots.py         # JSON results → figures with error bars
├── manuscript/latex/             # Paper sources (tables.tex is auto-generated)
└── main.py                       # Single-run CLI entry point
```

---

## Reproducing the Paper

```bash
conda activate py313          # Python 3.13, torch 2.10, torch_geometric 2.7
pip install -r requirements.txt

# Everything (4 datasets × 8 baselines + full model, ablations, taxonomy,
# privacy sweep, scalability, public-vs-fed; 3 seeds each; several hours on CPU):
bash experiments/run_all.sh 42,43,44

# Results land in experiments/results/*.json;
# tables.tex and figures are regenerated and copied into manuscript/.
```

Datasets are streamed automatically from public HuggingFace mirrors and cached
under `.cache/datasets/`:

| Dataset | Source | Split notes |
|---|---|---|
| Devign | `DetectVul/devign` | 2 projects (FFmpeg/QEMU) → train one, test the other |
| BigVul | `bstee615/bigvul` | 234 projects, CWE-annotated |
| DiverseVul | `bstee615/diversevul` | 580 projects, CWE lists (first = primary) |
| PrimeVul | `ASSERT-KTH/PrimeVul` (`train_unpaired`) | strictest recent benchmark |

**Cross-project split**: project-level partition; smallest projects held out
until ≈20% of samples form the test set; no function from a test project is
ever seen by any client. Remaining projects are assigned round-robin to K=4
clients. Seeds {42,43,44} control shuffling, initialisation and DP noise.

### Single runs

```bash
python main.py --dataset devign --max_samples 8000 --num_clients 4 \
               --rounds 10 --local_epochs 2 --num_cwes 10 --epsilon 2.0

# Ablation flags: --no_vcsa --no_morphology --no_cwe_affinity --no_mgmp --no_dp --local_only
# Taxonomy size:  --taxonomy_size {8,16,32}
# Privacy budget: --epsilon {0.1,...,inf}; clip radius: --delta_f (default 1.0)
```

---

## Label model (explicit)

- Detection is **strictly binary**; all P/R/F1/AUC come from the binary head at threshold 0.5.
- CWE annotations condition **only** the prototype bank; multi-CWE functions use their first-listed (primary) CWE; benign functions get no prototype.
- The bank has |C| = 10 slots: top-9 most frequent training CWEs + shared `OTHER`.

## Privacy accounting (explicit)

- Per round: each non-empty prototype row is the mean of L1-clipped (radius R=1)
  embeddings; Laplace scale = 2R/(N_c·ε) → the released bank is ε-DP
  (parallel composition across disjoint CWE buckets).
- Across T rounds: sequential composition, ε_total = T·ε (reported in all tables).

---

## Citation

> VulMorph-Fed: Cross-Project Software Vulnerability Detection via Federated
> Vulnerability Morphology Learning. (Under revision.)
