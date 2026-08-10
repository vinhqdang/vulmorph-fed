#!/bin/bash
# Full experiment suite for the VulMorph-Fed manuscript.
# Every number in the paper is produced by this script from scratch.
#
# Usage:  bash experiments/run_all.sh [SEEDS]
# Total runtime: several hours on CPU (3 parallel streams).

set -u
cd "$(dirname "$0")/.."

SEEDS="${1:-42,43,44}"
COMMON="--seeds $SEEDS --max_samples 8000 --num_clients 4 --rounds 10 --local_epochs 2 --num_cwes 10"
LOG_DIR=experiments/logs
mkdir -p $LOG_DIR experiments/results

echo "=== Stream A: RQ1 main comparison (4 datasets) ==="
(
  for ds in devign bigvul diversevul primevul; do
    python experiments/run_main.py      --dataset $ds $COMMON \
        --output results/vulmorph_${ds}.json  > $LOG_DIR/main_${ds}.log 2>&1
    python experiments/run_baselines.py --dataset $ds $COMMON --epochs 10 \
        --output results/baselines_${ds}.json > $LOG_DIR/baselines_${ds}.log 2>&1
  done
) &
PID_A=$!

echo "=== Stream B: RQ2 ablations + RQ2b taxonomy (devign) ==="
(
  python experiments/run_ablations.py --dataset devign $COMMON \
      --output results/ablations.json      > $LOG_DIR/ablations.log 2>&1
  python experiments/run_taxonomy.py  --dataset devign $COMMON \
      --output results/taxonomy_size.json  > $LOG_DIR/taxonomy.log 2>&1
) &
PID_B=$!

echo "=== Stream C: RQ3 privacy + RQ4 scalability + RQ1b public-vs-fed (devign) ==="
(
  python experiments/run_rq3_rq4.py --dataset devign $COMMON \
      > $LOG_DIR/rq3_rq4.log 2>&1
  python experiments/run_public_vs_fed.py --dataset devign $COMMON \
      --output results/public_vs_fed.json  > $LOG_DIR/public_vs_fed.log 2>&1
) &
PID_C=$!

wait $PID_A $PID_B $PID_C

echo "=== Generating tables and figures ==="
python experiments/generate_tables.py --output results/tables.tex
cp experiments/results/tables.tex manuscript/latex/tables.tex
python experiments/generate_plots.py
echo "=== DONE ==="
