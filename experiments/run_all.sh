#!/bin/bash
# Reproduces every number, table and figure in the manuscript.
#
#   bash experiments/run_all.sh [DEVICE]
#
# The detection runs dominate the cost (hours on CPU, well under an hour on a
# single GPU). The analysis and table steps take minutes and can be re-run on
# their own once experiments/results/*.json exist -- which is the usual case
# when checking that the paper matches its data:
#
#   bash experiments/run_all.sh --tables-only
#
# No table in the manuscript is written by hand. emit_tables.py reads the
# result JSONs and writes manuscript/latex/tab_*.tex, so a table cannot drift
# from the data it reports.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cpu}"
TABLES_ONLY=0
[ "${1:-}" = "--tables-only" ] && { TABLES_ONLY=1; DEVICE=cpu; }

LOG=experiments/logs
mkdir -p "$LOG" experiments/results

COMMON="--max_samples 8000 --n_splits 5 --epochs 10 --device $DEVICE"

if [ "$TABLES_ONLY" -eq 0 ]; then
  echo "=== RQ1: the labelling function as a static analysis ==="
  # Builds the callee sample, then scores the released annotations. The gold
  # set is under version control, so this reproduces without re-annotating.
  python experiments/build_phitype_goldset.py > "$LOG/goldset.log" 2>&1
  python experiments/annotate_goldset.py \
      --output experiments/results/phitype_eval.json > "$LOG/phitype.log" 2>&1

  echo "=== RQ2: feature mode on BigVul, GGNN (adds repeats to reach 10 folds) ==="
  python experiments/run_representation.py --dataset bigvul $COMMON \
      --n_repeats 2 --modes lexical,op,kind,op_kind --backbones ggnn \
      --output experiments/results/core_bigvul.json > "$LOG/core_bigvul.log" 2>&1

  echo "=== RQ3: second corpus, and two further message-passing schemes ==="
  python experiments/run_representation.py --dataset diversevul $COMMON \
      --modes lexical,kind,op_kind --backbones ggnn \
      --output experiments/results/core_diversevul.json > "$LOG/core_dv.log" 2>&1
  python experiments/run_representation.py --dataset bigvul $COMMON \
      --modes lexical,kind,op_kind --backbones gat,gin \
      --output experiments/results/backbone_bigvul.json > "$LOG/backbone.log" 2>&1

  echo "=== RQ4: taxonomy granularity, same folds at each granularity ==="
  # The fold assignment must match across granularities for the paired
  # comparison in analyze_granularity.py to be meaningful; it verifies this
  # rather than trusting it.
  for T in 16 32; do
    python experiments/run_representation.py --dataset bigvul $COMMON \
        --taxonomy_size $T --modes lexical,op_kind --backbones ggnn \
        --output "experiments/results/taxonomy_t${T}.json" \
        > "$LOG/taxonomy_t${T}.log" 2>&1
  done
fi

echo "=== Analysis ==="
python experiments/analyze_representation.py \
    experiments/results/core_bigvul.json \
    experiments/results/core_diversevul.json \
    experiments/results/backbone_bigvul.json \
    --n_boot 2000 --n_seeds 10 | tee "$LOG/analyze_representation.log"
python experiments/analyze_granularity.py | tee "$LOG/analyze_granularity.log"
python experiments/analyze_information.py | tee "$LOG/analyze_information.log"

echo "=== Tables and figures ==="
python experiments/emit_tables.py
python experiments/make_figures.py

echo "=== Manuscript ==="
# Figures live in manuscript/figures but are referenced as figures/*.pdf from
# manuscript/latex, so the parent directory has to be on the input path.
( cd manuscript/latex \
  && export TEXINPUTS="..:.:" \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null ) \
  && echo "wrote manuscript/latex/main.pdf"

echo "=== DONE ==="
