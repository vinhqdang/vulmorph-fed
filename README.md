# Grammar-Derived Operation Abstraction for Cross-Project Vulnerability Detection

> Research prototype with a fully reproducible experiment pipeline: every table and figure in the manuscript is generated programmatically from result files produced by `experiments/run_all.sh`. No number is entered by hand.

---

## Overview

This repository contains the official implementation for **"Grammar-Derived Operation Abstraction for Cross-Project Vulnerability Detection"** (under revision for *Journal of Computer Languages*).

### Core Finding
Learned vulnerability detectors generalize poorly across projects primarily because of the **node representation** rather than model architecture. Substituting project-specific lexical source tokens with project-invariant **grammar node kinds** ($\kappa$, fixed by the C grammar via `tree-sitter`) improves cross-project ranking performance by $+0.049$ to $+0.106$ AUC across datasets and backbones. Furthermore, an exact information-theoretic analysis demonstrates that an engineered operation taxonomy ($\phi$) is $98.2\%$ redundant with grammar kinds ($H(\phi \mid \kappa) = 0.06$ bits), proving that parser node kinds carry the entire effect at zero design cost.

---

## Repository Structure

```
vulmorph-fed/
├── data/
│   ├── morphology.py             # Taxonomic definitions (|T| ∈ {8, 16, 32}) & MorphologyEmbedding
│   └── loaders/
│       ├── ast_graphs.py         # tree-sitter C parse tree construction (AST + SIB + DU edges)
│       ├── api_classes.py        # API allow-lists & identifier-component callee classifier
│       └── real_datasets.py      # Dataset loaders (BigVul, DiverseVul, Devign) & GroupKFold splits
├── models/
│   ├── vulmorph.py               # VulMorph GNN model (kind + morph embeddings)
│   ├── vcsa.py                   # VCSA soft edge-mask MLP & Structural Contrastive Loss
│   ├── mgmp.py                   # MGMP message-passing layer
│   ├── encoders.py               # GNN backbones (GGNN, GAT, GIN)
│   └── baselines/                # Baseline architectures
├── fl/
│   ├── client.py                 # Client local training & prototype extraction
│   └── server.py                 # Prototype aggregation server
├── utils/
│   ├── metrics.py                # AUC, AUPRC, F1 metric computations
│   ├── privacy.py                # Differential privacy utilities
│   └── stats.py                  # Wilcoxon signed-rank + Cliff's delta statistics
├── experiments/
│   ├── run_all.sh                # ← Master script: runs experiments, emits tables & checks prose
│   ├── compute_corpus_stats.py   # Computes exact corpus properties & node distributions
│   ├── build_phitype_goldset.py  # Builds Devign callee goldset sample
│   ├── annotate_goldset.py       # Evaluates phi_type static analysis against ground truth
│   ├── run_representation.py     # Main representation comparison (RQ2, RQ3, RQ4)
│   ├── analyze_representation.py # Computes cluster-bootstrap CIs for representation comparisons
│   ├── analyze_information.py    # Computes H(phi|kappa) and H(kappa|phi) conditional entropies
│   ├── emit_tables.py            # Converts JSON result files to manuscript LaTeX tables
│   ├── check_manuscript.py       # Verifies all quoted prose statistics against JSON data
│   └── make_figures.py           # Generates manuscript figures (redundancy & forest plots)
├── manuscript/latex/             # Manuscript LaTeX sources (tables are auto-generated)
└── main.py                       # Single-run CLI entry point
```

---

## Reproducing the Paper

```bash
# Install dependencies
pip install -r requirements.txt

# Run the master pipeline (runs analysis, emits LaTeX tables, checks manuscript facts, builds PDF):
bash experiments/run_all.sh cpu

# Or generate tables & check manuscript facts directly from existing result files:
bash experiments/run_all.sh --tables-only
```

### Key Commands

```bash
# 1. Compute corpus properties (Table 1 & Table 2):
python experiments/compute_corpus_stats.py

# 2. Evaluate phi_type as a static analysis (Table 3 & Table 3b):
python experiments/annotate_goldset.py --output experiments/results/phitype_eval.json

# 3. Compute information-theoretic redundancy (Table 5):
python experiments/analyze_information.py

# 4. Generate all LaTeX tables programmatically:
python experiments/emit_tables.py

# 5. Verify prose facts against JSON result artifacts:
python experiments/check_manuscript.py
```

---

## Citation

> Grammar-Derived Operation Abstraction for Cross-Project Vulnerability Detection. (Under revision.)
