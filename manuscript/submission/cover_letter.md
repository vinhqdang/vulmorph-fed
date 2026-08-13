[DATE]

The Editors
Journal of Computer Languages
Elsevier

Dear Editors,

We submit for your consideration our manuscript, **"Grammar-Derived Operation
Abstraction for Cross-Project Vulnerability Detection"**, for publication as a
research article in the Journal of Computer Languages.

**What the paper is about.** The paper studies a program abstraction: a
labelling function over the node alphabet of a C grammar, and the question of
what a node in a program graph should carry when a learned analysis is expected
to transfer to code it has never seen. We compare, under strict experimental
control, three node representations — project-specific lexical tokens, an
engineered taxonomy of operation types constructed from API allow-lists and
identifier rules, and the parser's own grammar node kinds.

**Why we believe it fits this journal.** The object of study is a language-level
artefact rather than a machine-learning architecture. We specify the abstraction
as a total function over a tree-sitter grammar alphabet, give the projection
lattice relating its granularities with a nesting proof, and — unusually for
this application area — evaluate it *as a static analysis*, reporting per-class
precision and recall against annotated ground truth. That evaluation is not
decoration: it uncovered three defects invisible to downstream accuracy,
including a rule that classified pointer writes as deallocations and a class
that counted 655 stream-I/O call sites as memory operations.

**The principal finding.** The representation matters more than the
architecture, and the representation that works is the one that costs nothing.
Grammar node kinds improve cross-project ranking by +0.078 AUC (p = 0.007) and
+0.071 AUPRC (p = 0.002) over lexical tokens, winning every project-level fold,
whereas the three message-passing schemes we compare differ by less than half as
much under a fixed representation. The engineered taxonomy — the artefact we
built first and expected to be the contribution — is statistically
indistinguishable from lexical features (p ≈ 0.9). We report this negative
result at the same length as the positive one, and explain it: an
information-theoretic analysis shows the taxonomy is 98.2% predictable from the
grammar kind alone (H(φ | κ) = 0.06 bits), because most of its rules are
syntax-directed and therefore already decided by the parser. We recommend that
redundancy check as routine practice before an abstraction is credited with an
effect.

**Methodological care.** Cross-project evaluation in this area is easy to get
wrong, so we control for the failure modes documented in the literature:
near-duplicate functions are removed before splitting; projects are the unit of
partition, with repositories too large to place without dominating a fold held
train-only and reported; vocabularies are fitted on training projects only; and
confidence intervals come from a cluster bootstrap that resamples held-out
projects rather than functions. Every table reports the trivial all-positive
classifier as a floor.

**Originality and availability.** The manuscript is original, is not under
consideration elsewhere, and has not been published previously. All code,
annotations, and the scripts that regenerate every reported number are publicly
available at https://github.com/vinhqdang/vulmorph-fed.

We confirm that all authors have approved the manuscript and agree to its
submission, and we declare no competing interests.

Thank you for your consideration.

Yours sincerely,

[CORRESPONDING AUTHOR NAME]
[Affiliation]
[Email]
on behalf of all authors
