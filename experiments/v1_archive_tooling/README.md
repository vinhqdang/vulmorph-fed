# Archived tooling from the withdrawn version

Nothing in this directory produces any number in the current paper. It is kept
so that the withdrawn work is inspectable rather than silently deleted.

`run_all_v1.sh` and `generate_tables.py` drove an earlier system built around
federated aggregation with a differential-privacy budget. That system is not
evaluated in the current manuscript. Its results were withdrawn, not merely
superseded: validating the operation-labelling function as a static analysis
(RQ1 in the current paper) exposed defects in the labelling that fed every
number the old pipeline produced, and we did not consider the affected results
safe to carry forward. The corresponding result files are archived under
`experiments/results_v1_archive/` and `experiments/results_invalid_oldsplit/`,
each with its own note on why it is invalid.

Do not run these scripts to reproduce the paper. Use `experiments/run_all.sh`,
which reproduces the current manuscript end to end.
