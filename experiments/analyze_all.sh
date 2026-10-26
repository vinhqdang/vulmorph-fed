#!/bin/bash
cd "$(dirname "$0")/.."
python -u experiments/analyze_representation.py --corpus bigvul > experiments/logs/analysis_bigvul.log 2>&1 &
python -u experiments/analyze_representation.py --corpus diversevul > experiments/logs/analysis_diversevul.log 2>&1 &
wait
