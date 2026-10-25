#!/bin/bash
# Run the grid with 4 workers; safe to re-run (finished cells are skipped).
cd "$(dirname "$0")/.."
python experiments/grid_jobs.py | xargs -P 4 -I{} sh -c "{}"
