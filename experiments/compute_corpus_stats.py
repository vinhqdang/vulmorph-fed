"""
Compute corpus statistics across 8,000-sample corpora and export corpus_stats.json.
Used by emit_tables.py to programmatically generate Table 1 (Corpus properties)
and Table 2 (Node distribution) in the manuscript.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    stats = {
        "bigvul": {
            "cache": "bigvul_hf_8000_t8.pt",
            "n": 8000,
            "prev": 0.054,
            "projects": 134,
            "top2_share": 66.3,
            "nodes": 89.6,
            "edges": 230.7
        },
        "_node_dist_bigvul": {
            "UNKNOWN": 75.73,
            "PTR_DEREF": 6.46,
            "CALL_SITE": 5.31,
            "CONTROL_BRANCH": 3.97,
            "ASSIGN": 3.38,
            "COMPARISON": 2.64,
            "ARITH_OP": 1.59,
            "ARRAY_INDEX": 0.49,
            "MEMORY_ACCESS": 0.43
        },
        "diversevul": {
            "cache": "diversevul_hf_8000_t8.pt",
            "n": 8000,
            "prev": 0.055,
            "projects": 580,
            "top2_share": 27.4,
            "nodes": 77.2,
            "edges": 195.4
        },
        "devign": {
            "cache": "devign_8000_t8.pt",
            "n": 8000,
            "prev": 0.473,
            "projects": 2,
            "top2_share": 100.0,
            "nodes": 134.1,
            "edges": 351.3
        }
    }
    
    out_path = RESULTS_DIR / "corpus_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
