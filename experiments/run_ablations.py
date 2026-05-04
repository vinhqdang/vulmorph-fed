"""
Ablation Study Runner for VulMorph-Fed (Section 5.5 of the research plan).

Runs all ablation variants in sequence and prints a comparison table.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from main import run_fl, parse_args as base_parse_args


ABLATION_VARIANTS = {
    "Full VulMorph-Fed":           {},
    "w/o VCSA":                    {"no_vcsa": True},
    "w/o Morphological Abstraction": {"no_morphology": True},
    "w/o MCFPA (Uniform Avg)":     {"no_cwe_affinity": True},
    "w/o MGMP (Standard GAT)":     {"no_mgmp": True},
    "w/o DP":                      {"no_dp": True},
    "Local Only":                  {"local_only": True},
}


def make_args(base_ns, overrides: dict):
    """Clone a Namespace and apply boolean overrides."""
    import copy
    ns = copy.deepcopy(base_ns)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def main():
    p = argparse.ArgumentParser(description="VulMorph-Fed Ablation Runner")
    p.add_argument("--num_clients",  type=int, default=3)
    p.add_argument("--rounds",       type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--total_graphs", type=int, default=300)
    p.add_argument("--num_cwes",     type=int, default=5)
    p.add_argument("--vocab_size",   type=int, default=1000)
    p.add_argument("--embed_dim",    type=int, default=64)
    p.add_argument("--hidden_dim",   type=int, default=64)
    p.add_argument("--num_layers",   type=int, default=2)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--alpha",        type=float, default=0.1)
    p.add_argument("--gamma",        type=float, default=0.01)
    p.add_argument("--epsilon",      type=float, default=2.0)
    p.add_argument("--delta_f",      type=float, default=0.1)
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output",       type=str, default="results/ablations.json")
    args = p.parse_args()

    # Set ablation flags as False by default
    for flag in ["no_vcsa","no_mgmp","no_morphology","no_cwe_affinity","no_dp","local_only"]:
        setattr(args, flag, False)

    results = {}

    for name, overrides in ABLATION_VARIANTS.items():
        print(f"\n{'='*55}")
        print(f"  Variant: {name}")
        print(f"{'='*55}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        run_args = make_args(args, overrides)

        model_kwargs = dict(
            use_vcsa       = not run_args.no_vcsa,
            use_mgmp       = not run_args.no_mgmp,
            use_morphology = not run_args.no_morphology,
            use_cwe_affinity = not run_args.no_cwe_affinity,
            use_dp         = not run_args.no_dp,
            federate       = not run_args.local_only,
            num_layers     = run_args.num_layers,
            dropout        = run_args.dropout,
        )

        metrics = run_fl(run_args, model_kwargs)
        results[name] = metrics

    # ── Print Summary Table ──────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"{'Variant':<38} | {'F1':>7} | {'AUC':>7} | {'Prec':>7} | {'Rec':>7}")
    print(f"{'-'*38}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for name, m in results.items():
        print(
            f"{name:<38} | {m.get('f1',0):>7.4f} | {m.get('auc',0):>7.4f} "
            f"| {m.get('precision',0):>7.4f} | {m.get('recall',0):>7.4f}"
        )

    # Save JSON
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
