# Response to Reviewers — SCICO-D-26-00310 (revision change log)

This document maps every reviewer comment to the concrete change made in the
revised manuscript and codebase. Section/table numbers refer to the revised
manuscript. The complete implementation and the scripts that regenerate every
reported number are public: https://github.com/vinhqdang/vulmorph-fed

> **Most important change (affects several comments at once).** The reviewers
> identified internal inconsistencies in the reported numbers (Table 4
> precision/recall of 0.0000 with non-zero F1; Table 2 vs Table 3 disagreement).
> Investigating these, we found the previous results pipeline did not reliably
> produce the reported tables. We have rebuilt the entire experimental pipeline
> so that **every table and figure is generated programmatically from versioned
> result files produced by `experiments/run_all.sh`**, with multi-seed
> repetition and mean ± std everywhere. All numbers in the revision are new,
> and the honest picture they paint is substantially more modest than the
> previous draft claimed. We have rewritten the results and claims accordingly
> (see Rev2-Q“claims” and Section 5 of the revision).

---

## Reviewer 1

**R1.1 — VCSA/MCFPA/MGMP too abstract; need a worked example, the complete
taxonomy, and mapping rules.**
- Section 3.2.1 now lists the complete 8-type taxonomy exhaustively
  (Table `tab:taxonomy`), including the reserved `UNKNOWN` fallback, and points
  to the nested 16-/32-type refinements shipped with the code.
- Section 3.2.2 specifies the mapping φ_type as a deterministic, priority-ordered,
  context-aware rule set (and states explicitly how user-defined functions,
  macros and unresolved calls are handled: all map to `CALL_SITE`).
- Figure `fig:worked_example` walks two lexically different but structurally
  similar off-by-one overflows (FFmpeg-style vs OpenSSL-style) through the
  abstraction, showing the shared morphology.

**R1.2 — How are UNKNOWN nodes represented; how are aggregation weights
computed?**
- UNKNOWN has its own feature slot: node features have |T|+1 = 9 dimensions
  (Section 3.2.1). This also resolves the 8-vs-9 dimensionality inconsistency
  flagged by Reviewer 2.
- The aggregation weights are now given in closed form (Eq. `eq:mcfpa`):
  cosine-affinity weights over the clients that hold data for a CWE bucket,
  with explicit fallbacks (uniform mean; previous round's value).

**R1.3 — Analysis of *why* the method works is insufficient.**
- The baseline suite now isolates the two candidate explanations: representation
  (centralised GGNN on lexical vs VCSA graphs) and mechanism (FedAvg vs MCFPA on
  the *same* VCSA representation). Section 5.1 discusses which factor carries
  the effect, backed by the new tables.
- We additionally report abstraction statistics (fraction of typed nodes,
  graph sizes) and client-level F1 variance.

**R1.4 — No source code or configurations.**
- Everything is released, including `experiments/run_all.sh` which regenerates
  all tables and figures from scratch, pinned dataset mirrors, and seeds.

## Reviewer 2

**R2.1 — Motivation for federation not established; public corpora exist;
run a centralised-public vs +federation experiment.**
- We added exactly this experiment (RQ1b, Table `tab:rq1b_public`,
  `experiments/run_public_vs_fed.py`): condition A trains centrally on a public
  partition; condition B adds private clients via DP prototype federation only;
  both are evaluated on the same held-out projects. The measured delta —
  not rhetoric — now carries the federation argument, and Section 5.3 discusses
  it honestly, including the limits of simulating private silos with public data.

**R2.2 / Q3–Q4 — Data inconsistencies (Table 4 zeros; Table 2 vs 3
disagreement).**
- Root cause found and fixed: the previous table generator emitted 0.0000 for
  metrics missing from the underlying files, and two tables were fed from
  inconsistent files. The rebuilt generator renders missing values as “--”,
  never as zeros, and all tables draw from one canonical set of result files.
  All experiments were re-run from scratch (3 seeds, mean ± std).

**R2.3 — Label model unspecified (multi-CWE functions, benign prototypes,
choice of 10 classes, how binary metrics arise).**
- New Section 3.3 “Label Model” states all of this explicitly: detection is
  strictly binary (all P/R/F1/AUC from the binary head at threshold 0.5);
  CWE conditions only the prototypes; multi-CWE functions contribute to their
  first-listed (primary) CWE; benign functions get no prototype; the bank has
  10 slots = top-9 training CWEs + OTHER, computed per dataset and released.

**R2.4 — VCSA underspecified: taxonomy not listed, φ_type undefined, edge set
undefined, |T| vs |T|+1 inconsistency, no abstraction statistics.**
- Taxonomy: fully listed (Table `tab:taxonomy`). φ_type: fully specified
  (Section 3.2.2). Edge set: defined in Section 3.1 (NCS + data-dependence
  proxy edges; the abstraction does not delete nodes — features are collapsed
  and edges are *softly* down-weighted by a learned mask, Eq. `eq:edge_mask`;
  the earlier “bidirectional reachability” formulation was removed because it
  did not match the implementation). Feature dimensionality: |T|+1 (UNKNOWN has
  a slot). Statistics: ~25–26% of nodes receive a non-UNKNOWN type across all
  four corpora; average node/edge counts reported.

**R2 (repro) — “CPG generation claimed but a proxy graph from tokenised source
is used.”**
- The revision states this honestly and prominently (Section 3.1 and
  Limitations): our graphs are lightweight lexical dependence graphs, not
  compiler-grade CPGs; all baselines consume the same graphs so comparisons are
  internally fair; the graph-construction interface is isolated so Joern CPGs
  can be substituted.

**R2 (inference) — Algorithm never aggregates parameters; which model produces
the numbers?**
- New Section 3.6 defines the inference protocol precisely: parameters are never
  aggregated; the deployed detector is the uniform probability ensemble of the
  K client models conditioned on the shared prototype bank (Eq. `eq:ensemble`);
  per-client mean/std F1 is reported to expose client-level variance.

**R2 (claims) — “end-to-end differentiable”, “formal analysis” without a
theorem, “significantly outperforms” without tests.**
- “End-to-end differentiable” removed: φ_type is stated to be a discrete,
  non-differentiable assignment; only the edge mask is learned.
- The privacy analysis is now a numbered Proposition with a proof (L1 clipping,
  per-class calibrated Laplace, parallel composition across CWE buckets) plus a
  Corollary giving the composed end-to-end budget ε_tot = T·ε. The
  implementation now actually enforces the clipping the proof requires.
- All “significantly outperforms” language is gone; Wilcoxon signed-rank +
  Cliff’s δ over paired dataset–seed F1 values are reported, and comparisons
  that do not reach p < 0.05 are stated as such.

**R2 (composition) — reported ε was per round, not end-to-end.**
- Fixed: every privacy table now reports both ε/round and ε_tot = T·ε
  (Corollary in Section 3.4.1).

**R2 (structure) — Section 2 too long relative to Section 3; IST artifacts;
communication-cost inconsistencies (300 KB vs 150 KB vs 120–260 KB).**
- Section 3 was expanded from ~3 pages to the full specification described
  above; Section 2 was kept compact.
- Venue-specific artifacts removed.
- Communication cost is now stated once, precisely, and used consistently:
  2·|C|·d·4 B = 10 KB per client per round (upload+download), K× at the server
  (e.g. 200 KB/round at K = 20). Abstract, Section 3.7, RQ4 all agree.

**R2 (citations) — duplicate refs [1]/[16]; [31] mis-attributed; [7] wrong
pages/DOI.**
- ref.bib fully deduplicated (ten former duplicate pairs merged). CSVD-TF
  re-attributed to Cai et al., JSS 213 (2024) 112038 (verified against the DOI
  record). Yamamoto et al. SANER 2023 corrected to pages 485–496 /
  DOI …00052. The Real-Vul replication study re-attributed to
  P. Chakraborty et al. (TSE 2024). Additional entries verified against DOI
  records (Vul-LMGNNs → Information Fusion 115:102748; VulFL first author
  Zhou; etc.).

## Reviewer 3

**R3.1 — Taxonomy size insufficiently motivated; test 16 and 32 types.**
- Added RQ2b (`experiments/run_taxonomy.py`, Table `tab:rq2b_taxonomy`):
  the full system run with strictly nested |T| ∈ {8, 16, 32} taxonomies under
  identical conditions, with seed variance, so the sensitivity to taxonomy
  granularity is now measured rather than asserted.

**R3.2 — Baseline confounds: no FedAvg+GGNN, baselines use different input
representations, no LM baselines.**
- Added: FedAvg + GGNN on the *same* VCSA-abstracted graphs (isolates the
  aggregation mechanism); centralised GGNN on VCSA graphs (isolates the
  representation); centralised VulMorph oracle (upper reference for our own
  architecture); centralised and FedAvg Transformer sequence baselines.
  We state explicitly that the Transformer is trained from scratch and that a
  pre-trained CodeBERT comparison is future work requiring GPU budget — this
  limitation is acknowledged rather than hidden.

**R3.3 — Table 4 mathematically inconsistent (P=R=0 with F1>0).**
- Root cause and fix as in R2.2 above. The regenerated table reports all four
  metrics, computed from the same predictions, at every ε.

**R3.4 — Cross-project split underspecified.**
- New Section 4.1.2: project-level granularity (all functions of a repository
  in one partition); smallest projects held out to ≈20% of samples as the test
  set; remaining projects round-robin across K = 4 clients; the Devign
  degenerate two-project case is called out explicitly; seeds {42,43,44} listed;
  what each seed controls is stated.

**R3.5 — No variance or significance reporting.**
- Every number in every table is mean ± std over 3 seeds; per-seed raw values
  are released; Wilcoxon + Cliff’s δ over paired dataset–seed values are
  reported with the caveat about test power at this seed count.

## Handling editor / general

- Language pass performed throughout; venue artifacts removed.
- The revised manuscript is deliberately more modest in its claims: absolute
  cross-project performance is reported as-is (it remains far from
  within-project levels, consistent with the replication literature), and the
  contribution is framed as a fully specified, provably private, reproducible
  federated framework evaluated under controls that isolate representation
  from mechanism.
