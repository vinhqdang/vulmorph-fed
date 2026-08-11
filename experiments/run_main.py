"""
Full VulMorph-Fed runner for the RQ1 main comparison.

Runs the complete framework (VCSA + MCFPA + MGMP, DP at ε=2.0/round) on one
dataset over multiple seeds and stores aggregated metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse

from main import run_fl
from experiments.common import make_args, run_seeded, add_common_cli, parse_seeds


def main():
    p = argparse.ArgumentParser(description="VulMorph-Fed main runner (RQ1)")
    add_common_cli(p)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--output",  type=str,   default=None)
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    def run_one(seed):
        run_args = make_args(
            seed=seed, num_clients=args.num_clients, rounds=args.rounds,
            local_epochs=args.local_epochs, epsilon=args.epsilon,
            dataset=args.dataset, max_samples=args.max_samples,
            test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim, num_cwes=args.num_cwes,
            taxonomy_size=args.taxonomy_size, device=args.device,
            dp_sgd=getattr(args, 'dp_sgd', False),
            dp_noise_multiplier=getattr(args, 'dp_noise_multiplier', 1.0),
        )
        model_kwargs = dict(use_vcsa=True, use_mgmp=True, use_morphology=True,
                            use_cwe_affinity=True, use_dp=True,
                            federate=True, num_layers=2, dropout=0.3)
        return run_fl(run_args, model_kwargs)

    results = run_seeded(run_one, seeds)

    output = args.output or f"results/vulmorph_{args.dataset}.json"
    out = Path(__file__).parent / output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    f1 = results.get("f1", {})
    print(f"\nVulMorph-Fed on {args.dataset}: "
          f"F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}")
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
