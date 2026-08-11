"""
Ablation Study Runner for VulMorph-Fed (RQ2).

Runs all ablation variants on the real cross-project split, repeated over
multiple seeds, and reports mean ± std for every metric.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from main import run_fl
from experiments.common import make_args, run_seeded, add_common_cli, parse_seeds


ABLATION_VARIANTS = {
    "Full VulMorph-Fed":             {},
    "w/o VCSA":                      {"no_vcsa": True},
    "w/o Morphological Abstraction": {"no_morphology": True},
    "w/o MCFPA (Uniform Avg)":       {"no_cwe_affinity": True},
    "w/o MGMP (Standard GAT)":       {"no_mgmp": True},
    "w/o DP":                        {"no_dp": True},
    "Local Only":                    {"local_only": True},
}


def main():
    p = argparse.ArgumentParser(description="VulMorph-Fed Ablation Runner (RQ2)")
    add_common_cli(p)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--output",  type=str,   default="results/ablations.json")
    p.add_argument("--resume",  action="store_true")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    results = {}
    out_path = Path(__file__).parent / args.output
    if args.resume and out_path.exists():
        try:
            results = json.load(open(out_path))
            print(f"Resuming: {sorted(results)} already complete")
        except Exception:
            results = {}

    for name, overrides in ABLATION_VARIANTS.items():
        if name in results:
            print(f"Skipping {name} (already complete)")
            continue
        print(f"\n{'='*55}\n  Variant: {name}\n{'='*55}")

        def run_one(seed, overrides=overrides):
            eps = float('inf') if overrides.get("no_dp") else args.epsilon
            run_args = make_args(
                seed=seed, num_clients=args.num_clients, rounds=args.rounds,
                local_epochs=args.local_epochs, epsilon=eps,
                dataset=args.dataset, max_samples=args.max_samples,
                test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim, num_cwes=args.num_cwes,
                taxonomy_size=args.taxonomy_size, device=args.device,
            dp_sgd=getattr(args, 'dp_sgd', False),
            dp_noise_multiplier=getattr(args, 'dp_noise_multiplier', 1.0),
                **{k: v for k, v in overrides.items()},
            )
            model_kwargs = dict(
                use_vcsa=not overrides.get("no_vcsa", False),
                use_mgmp=not overrides.get("no_mgmp", False),
                use_morphology=not overrides.get("no_morphology", False),
                use_cwe_affinity=not overrides.get("no_cwe_affinity", False),
                use_dp=not overrides.get("no_dp", False),
                federate=not overrides.get("local_only", False),
                num_layers=2, dropout=0.3,
            )
            return run_fl(run_args, model_kwargs)

        results[name] = run_seeded(run_one, seeds)
        if args.output:
            out = Path(__file__).parent / args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

    # ── Print Summary Table ──────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("ABLATION STUDY RESULTS (mean ± std over seeds)")
    print(f"{'='*72}")
    print(f"{'Variant':<34} | {'F1':>15} | {'AUC':>15}")
    print(f"{'-'*34}-+-{'-'*15}-+-{'-'*15}")
    for name, m in results.items():
        f1 = m.get("f1", {})
        auc = m.get("auc", {})
        print(f"{name:<34} | {f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f} "
              f"| {auc.get('mean', 0):.4f}±{auc.get('std', 0):.4f}")

    if args.output:
        out = Path(__file__).parent / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
