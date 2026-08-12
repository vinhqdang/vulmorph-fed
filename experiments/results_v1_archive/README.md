# Archived results (superseded)

Produced before three corrections that change the model's inputs and the
data pipeline, so none of these numbers are reproducible by the current code:

1. `classify_api` matched substrings, misclassifying writes, predicates and
   logging macros as memory operations (and missing genuine copies).
2. The CWE prototype vocabulary was fitted on the full corpus before the
   project split (test-set leakage into the structural vocabulary).
3. The RQ2 ablation table read a Devign run under a BigVul caption.

Retained only for provenance and for the record of what the earlier framing
claimed.
