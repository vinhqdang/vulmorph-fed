# Submission history

Grammar-Derived Operation Abstraction for Cross-Project Vulnerability Detection

| # | Venue | Manuscript no. | Submitted | Decision | Decision date | Outcome |
|---|-------|----------------|-----------|----------|---------------|---------|
| 1 | Journal of Computer Languages (Elsevier) | COLA-D-26-00245 | 13 August 2026 (cover letter date; initial submission) | Reject (Editor-in-Chief: Marjan Mernik) | October 2026 (reviews dated 2 and 24 October 2026) | Rejected after review by two reviewers |

## Submission 1: COLA-D-26-00245 (Journal of Computer Languages)

- **Version submitted:** initial submission (double-blind package with a separate title page).
- **Reviews received:** Reviewer 1 (2 October 2026), Reviewer 2 (24 October 2026).
- **Decision:** reject. The editor's letter advised against publication on the
  strength of both reports.

### Reviewer 1 (summary)

Could not follow the manuscript: the abstract was judged too long and hard to
read, and the English, writing style and organisation were judged to obstruct
comprehension of the message. Recommended rejection on that basis. Specific
complaints: the terms "lexical tokens", "engineered operation taxonomy" and
"the parser's own grammar node kinds" are not explained for a reader outside
the sub-field, and the claim that abandoning lexical features "proved useful"
while the proposed approaches "did not achieve the expected results" reads as
contradictory.

### Reviewer 2 (summary of the 15 numbered points)

1. State the paper organisation at the end of Section 1.
2. Tables 1, 4 and 9 are not cited in the text.
3. No statistical test beyond the bootstrap (suggested Wilcoxon, chi-square).
4. Say whether the datasets are balanced and how imbalance is handled.
5. Provide a block diagram of the approach.
6. Provide an algorithm or pseudocode.
7. Provide an ablation study.
8. Compare with existing related approaches.
9. Fold counts are inconsistent (10/5/5/9 folds across Table 6 although the text
   claims 10 matched folds); headline result rests on 5 folds; the
   granularity claim rests on one fold vs a 5-fold run.
10. Lexical baseline underspecified; internal AST nodes carry no token in lexical
    mode; add lexical+kind and anonymised-identifier+kind baselines; position
    against Devign/ReVeal node-type features.
11. kappa+phi is never compared directly with kappa; DiverseVul point estimate
    favours kappa+phi.
12. The architecture claim rests on three similar GNNs (AUC span 0.027); narrow
    it or add a transformer / pretrained model.
13. Samples small and not random (8,000 functions in corpus order; about 430
    vulnerable in BigVul; Android is 69% of the test set in some folds); report
    per project or without Android; explain prevalence 0.054 vs 8.8%.
14. phi is validated on Devign (FFmpeg and QEMU only), not on the experimental
    corpora; 212 names, some classes with 2 true positives; state who annotated,
    add a second annotator with agreement, validate on the experimental corpora.
15. Missing hyper-parameters (hidden size, learning rate, epochs, early
    stopping, seeds) and effect of the 200-node cap on each class.

### Response

No resubmission to the same journal is possible (final decision). All points
above were addressed in the revised manuscript prepared for submission
elsewhere; the point-by-point record is in
`manuscript/submission/revision_response.md`.

### Next submission

| # | Venue | Status |
|---|-------|--------|
| 2 | to be chosen by the author | revised manuscript prepared (this repository) |
