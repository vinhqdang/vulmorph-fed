#!/bin/bash
# Reproduces every number, table and figure in the manuscript.
#
#   bash experiments/run_all.sh            # full run (many CPU-hours)
#   bash experiments/run_all.sh --tables-only
#
# The detection grid dominates the cost (about 25 cells per hour on four CPU
# cores; the grid has roughly 300 cells). The analysis and table steps take
# minutes and can be re-run on their own once experiments/results/*.json exist.
#
# No table or prose number is typed by hand: emit_tables.py reads the result
# JSONs and writes manuscript/latex/tab_*.tex and numbers.tex (LaTeX macros).

set -euo pipefail
cd "$(dirname "$0")/.."

TABLES_ONLY=0
[ "${1:-}" = "--tables-only" ] && TABLES_ONLY=1

mkdir -p experiments/logs experiments/results

if [ "$TABLES_ONLY" -eq 0 ]; then
  echo "=== Detection grid (random 10k samples, 10 project-level folds) ==="
  bash experiments/launch_grid.sh

  echo "=== Reference baselines ==="
  for c in bigvul diversevul; do
    python experiments/run_size_baseline.py --dataset $c
    python experiments/run_pretrained.py --dataset $c --output results/${c}_unixcoder.json
  done

  echo "=== RQ1: validate phi on the experimental corpora ==="
  for c in bigvul diversevul; do
    python experiments/build_phitype_goldset.py --dataset $c --sample_n 10000 \
        --per_class 14 --tail 100 --output experiments/goldset_$c.tsv
  done
  # The three blind annotation passes (experiments/annotations/) are released
  # with the repository; score_phitype.py reads them.
  python experiments/score_phitype.py
fi

echo "=== Analysis ==="
python experiments/compute_corpus_stats.py
python experiments/analyze_information.py
python experiments/analyze_representation.py --n_boot 2000 --n_seeds 10

echo "=== Tables and figures ==="
python experiments/emit_tables.py
python experiments/check_manuscript.py
python experiments/make_figures.py

echo "=== Manuscript ==="
( cd manuscript/latex \
  && export TEXINPUTS="..:.:" \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null ) \
  && echo "wrote manuscript/latex/main.pdf"
echo "=== DONE ==="
