# Response to Reviewers — Revision Change Log

This document maps reviewer comments to the concrete changes made in the revised manuscript and codebase. Section and table numbers refer to the revised manuscript.

> **Most important change (affects several comments at once).** The reviewers
> identified internal inconsistencies in reported numbers in prior drafts.
> Investigating these, we rebuilt the entire experimental pipeline so that **every table and figure is generated programmatically from versioned result files produced by `experiments/run_all.sh`**, with multi-seed repetition and mean ± std everywhere. All numbers in the revision are drawn from canonical result files and verified via `check_manuscript.py`.

---

## Reviewer 1

**R1.1 — VCSA/MCFPA/MGMP too abstract; need a worked example, the complete taxonomy, and mapping rules.**
- Section 3.2 lists the complete taxonomy (Table `tab:taxonomy`), including the reserved `UNKNOWN` fallback, and points to the nested 16-/32-type refinements.
- Section 3.2 specifies the mapping $\phi$ as a deterministic, priority-ordered, context-aware rule set.

**R1.2 — How are UNKNOWN nodes represented; how are aggregation weights computed?**
- UNKNOWN has its own feature slot: node features have $|T|+1 = 9$ dimensions (Section 3.2).
- Section 3 defines the node representations and graph properties clearly.

**R1.3 — Analysis of *why* the method works is insufficient.**
- The baseline suite isolates the candidate explanations: representation (centralised GGNN on lexical vs AST grammar kinds) and architecture choice. Section 4 & Section 5 discuss which factor carries the effect, backed by programmatic tables.

**R1.4 — Source code and configurations.**
- Complete implementation and reproduction scripts (`experiments/run_all.sh`) are included.

---

## Reviewer 2

**R2.1 — Motivation for representation vs architecture comparison.**
- We added exact controlled comparisons across 3 GNN backbones (GGNN, GAT, GIN) and 2 corpora (BigVul, DiverseVul), showing that node feature representation choice dominates architecture choice.

**R2.2 — Data inconsistencies.**
- Root cause fixed: all tables draw from canonical result files via `emit_tables.py` and are verified by `check_manuscript.py`.

**R2.3 — Label model and protocol disclosures.**
- Section 4 state detection is strictly binary; project-level GroupKFold splits are documented transparently along with test set project distributions and fold concentration disclosures.

---

## Reviewer 3

**R3.1 — Taxonomy size sensitivity.**
- Section 4.5 evaluates taxonomy granularity across $|T| \in \{8, 16, 32\}$ over matched folds.

**R3.2 — Statistical significance and variance reporting.**
- Cluster-bootstrap 95% confidence intervals and two-sided bootstrap $p$-values over held-out projects are reported across all comparisons.

---

## Summary of Revisions
- All manuscript tables are generated programmatically via `emit_tables.py`.
- Manuscript text claims are automatically verified against data JSON artifacts via `check_manuscript.py`.
- Strict double-blind submission compliance verified.
