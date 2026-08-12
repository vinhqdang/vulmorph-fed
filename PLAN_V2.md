# Research plan v2 — what the evidence supports

## Why this plan replaces the previous one

Three independent adversarial reviews plus our own verification established
that the federated-privacy framing is not supported by the data:

- the DP-FedAvg peer we compared against was a degenerate constant classifier
  (recall = 1.000, thresholds at exactly 0.0/1.0, AUC = chance), so the
  headline "we win under matched privacy budget" reduced to beating a model
  that labels everything vulnerable — and even that reached only p = 0.084;
- our own `significance.json` recorded that non-private FedAvg on the *same*
  representation is **significantly better** than the proposed system
  (p = 0.0039, Cliff's delta = -0.58);
- the ablations showed the learned edge mask and the affinity weighting are
  neutral-to-harmful, i.e. two of three named contributions do not earn their
  place;
- the end-to-end DP claim was never exercised (`--dp_sgd` defaulted off), and
  as implemented the per-example loss was coupled across a batch, so the
  mechanism was not DP at any epsilon.

One result, however, reproduced across every dataset, every backbone and both
the centralised and federated settings:

| | lexical -> abstracted node features (AUC) |
|---|---|
| Centralised GGNN, BigVul | 0.630 -> 0.674 (+0.044) |
| Centralised GGNN, DiverseVul | 0.662 -> 0.713 (+0.051) |
| FedAvg, BigVul | 0.568 -> 0.666 (+0.099) |
| FedAvg, DiverseVul | 0.577 -> 0.725 (+0.148) |

and the ablation agreed: removing the abstraction is the only removal that
costs AUC (+0.049, 5/5 seeds). **The abstraction is the contribution.**

## The paper

**Title (working).** Grammar-Derived Operation Abstraction for Cross-Project
Vulnerability Detection.

**Claim.** Cross-project transfer in graph-based vulnerability detection is
governed primarily by *what the node features encode*, not by the detector
architecture or the training protocol. Replacing project-specific lexical
tokens with a grammar-derived operation abstraction is worth more than
changing backbone, and it transfers because the abstraction alphabet is fixed
by the language grammar rather than by any project's vocabulary.

**Fit for Journal of Computer Languages.** The object of study is a program
abstraction: a total labelling function from a tree-sitter grammar's node
alphabet onto a small operation lattice. It is specified, validated as a
static analysis (precision/recall per class against ground truth), and its
effect on a downstream task is measured. Federated learning and DP appear
only as a corollary application, not as the contribution.

## Contributions

C1. **A specified abstraction.** phi_type as a total, deterministic labelling
    over the tree-sitter C grammar, with a projection lattice
    T32 -> T16 -> T8 that is a homomorphism (proved, not asserted).
C2. **Validation of the abstraction as an analysis.** Per-class precision and
    recall against ground truth, plus the measured precision/coverage
    trade-off between an exact libc allow-list and component-based
    recognition of project-wrapped APIs. This is the experiment whose absence
    let a debug-logging macro be classified as a memory operation.
C3. **The empirical result.** Across datasets x backbones x training
    protocols, the abstraction dominates architecture choice for
    cross-project transfer. Reported with project-level GroupKFold and
    cluster-bootstrap confidence intervals, not seed variance.
C4. **A privacy corollary.** Statistics computed over the abstracted graphs
    are lexically empty by construction, so class-conditioned summaries can
    be released under DP cheaply. Reported honestly, including that
    parameter averaging without privacy remains stronger.
C5. **Negative results**, stated plainly: the learned edge mask and the
    affinity-weighted aggregation do not help; taxonomy refinement beyond
    eight types does not help.

## Experiments

E1 (core). Representation x backbone factorial. Node features in
   {lexical, operation-type only, grammar-kind only, operation+kind} x
   backbones {GGNN, GAT, GIN} x datasets {BigVul, DiverseVul, PrimeVul} under
   an identical centralised protocol. Establishes that the representation
   effect is larger and more consistent than the architecture effect.

E2 (analysis validation). Ground-truth labels for a stratified sample of call
   sites; per-class precision/recall for phi_type; the allow-list vs
   component-matching trade-off curve; documented failure modes.

E3 (protocol). Project-level GroupKFold (5x5 repeated) instead of seeds, with
   a cluster bootstrap over held-out *projects* for CIs on paired metric
   differences. Fixes the earlier design in which three of five "seeds" were
   effectively single-project evaluations at 41-51% test fraction.

E4 (transfer to federation). The same representation contrast under FedAvg,
   showing the effect is not an artefact of centralised training.

E5 (privacy corollary). Prototype-level DP against a correctly implemented
   DP-FedAvg (L2 clipping, Gaussian, server-side noise scaled 1/K, the same
   Renyi accountant used for both arms).

## Protocol rules (fixed in advance)

- Primary metric: **AUC**, with AUPRC secondary. F1 is reported at a
  budget-defined operating point (Recall@FPR=0.01), never at a threshold
  tuned per system on a differently-distributed calibration set.
- Every table reports the trivial all-positive classifier as a floor.
- All systems matched on gradient steps, depth, hidden width and readout.
- Vocabulary and CWE bucketing fitted on training projects only.
- Near-duplicate functions removed before splitting.
- Ensembling, if used at all, applied identically to every system.
