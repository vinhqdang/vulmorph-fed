"""
Taxonomy-size sensitivity analysis (RQ2b).

Runs the full VulMorph-Fed model with |T| ∈ {8, 16, 32} morphological types
to quantify how sensitive detection performance and the privacy-utility
trade-off are to the granularity of the abstraction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse

from main import run_fl
from experiments.common import make_args, run_seeded, add_common_cli, parse_seeds


def main():
    p = argparse.ArgumentParser(description="Taxonomy-size sensitivity (RQ2b)")
    add_common_cli(p)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--output",  type=str,   default="results/taxonomy_size.json")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    results = {}
    for tax in [8, 16, 32]:
        print(f"\n{'='*50}\nTaxonomy |T| = {tax}\n{'='*50}")

        def run_one(seed, tax=tax):
            run_args = make_args(
                seed=seed, num_clients=args.num_clients, rounds=args.rounds,
                local_epochs=args.local_epochs, epsilon=args.epsilon,
                dataset=args.dataset, max_samples=args.max_samples,
                test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim, num_cwes=args.num_cwes,
                taxonomy_size=tax, device=args.device,
            )
            model_kwargs = dict(use_vcsa=True, use_mgmp=True,
                                use_morphology=True, use_cwe_affinity=True,
                                use_dp=True, federate=True,
                                num_layers=2, dropout=0.3)
            return run_fl(run_args, model_kwargs)

        results[str(tax)] = run_seeded(run_one, seeds)
        out = Path(__file__).parent / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nTAXONOMY SIZE SENSITIVITY (mean ± std)\n{'='*60}")
    for tax, m in results.items():
        f1 = m.get("f1", {})
        print(f"|T|={tax:>3} | F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}")

    out = Path(__file__).parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
