"""
Privacy sweep and scalability experiments for VulMorph-Fed (plan.md §5.3, §5.4).

RQ3: Privacy-Utility Tradeoff — F1 vs epsilon curve
     ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, ∞}

RQ4: Scalability — performance as K ∈ {3, 5, 10, 20} clients
     Also computes client-level F1 variance and Communication Cost per Round.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import argparse
import copy

from data.loaders.real_datasets import load_devign, split_by_project, ListDataset
from main import run_fl, parse_args as base_parse_args


def make_args(seed=42, num_clients=4, rounds=10, local_epochs=2,
              epsilon=2.0, total_graphs=5000, num_cwes=10,
              hidden_dim=128, embed_dim=64, dataset="devign",
              max_samples=8000, test_fraction=0.2, **kwargs):
    """Build an args namespace for run_fl."""
    import argparse
    ns = argparse.Namespace(
        dataset=dataset,
        data_path=None,
        max_samples=max_samples,
        test_fraction=test_fraction,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        total_graphs=total_graphs,
        vocab_size=10000,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_cwes=num_cwes,
        num_layers=2,
        batch_size=64,
        lr=1e-3,
        dropout=0.3,
        alpha=0.1,
        gamma=0.01,
        epsilon=epsilon,
        delta_f=0.1,
        no_vcsa=False,
        no_mgmp=False,
        no_morphology=False,
        no_cwe_affinity=False,
        no_dp=(epsilon == float('inf')),
        local_only=False,
        device="cpu",
        seed=seed,
        output=None,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ── RQ3: Privacy Sweep ───────────────────────────────────────────────────────

def run_privacy_sweep(args):
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
    results = {}

    for eps in epsilons:
        key = "inf" if eps == float('inf') else str(eps)
        print(f"\n{'='*50}\nPrivacy Sweep ε={key}\n{'='*50}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        run_args = make_args(
            seed=args.seed, num_clients=args.num_clients,
            rounds=args.rounds, local_epochs=args.local_epochs,
            epsilon=eps, dataset=args.dataset,
            max_samples=args.max_samples, test_fraction=args.test_fraction,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            num_cwes=args.num_cwes,
        )

        model_kwargs = dict(
            use_vcsa=True, use_mgmp=True, use_morphology=True,
            use_cwe_affinity=True,
            use_dp=(eps != float('inf')),
            federate=True, num_layers=2, dropout=0.3,
        )
        metrics = run_fl(run_args, model_kwargs)
        metrics["epsilon"] = key
        results[key] = metrics
        print(f"  ε={key} → F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    return results


# ── RQ4: Scalability Sweep ───────────────────────────────────────────────────

def run_scalability_sweep(args):
    client_counts = [3, 5, 10, 20]
    results = {}

    for K in client_counts:
        print(f"\n{'='*50}\nScalability K={K} clients\n{'='*50}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        run_args = make_args(
            seed=args.seed, num_clients=K,
            rounds=args.rounds, local_epochs=args.local_epochs,
            epsilon=2.0, dataset=args.dataset,
            max_samples=args.max_samples, test_fraction=args.test_fraction,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            num_cwes=args.num_cwes,
        )

        model_kwargs = dict(
            use_vcsa=True, use_mgmp=True, use_morphology=True,
            use_cwe_affinity=True, use_dp=True,
            federate=True, num_layers=2, dropout=0.3,
        )
        metrics = run_fl(run_args, model_kwargs)

        # Communication cost per round: O(|C|*d) bytes
        num_cwes = run_args.num_cwes
        hidden_dim = run_args.hidden_dim
        # Each float is 4 bytes; prototype bank is (num_cwes, hidden_dim)
        ccr_kb = (num_cwes * hidden_dim * 4 * 2 * K) / 1024  # upload + download
        metrics["ccr_kb"] = round(ccr_kb, 2)
        metrics["num_clients"] = K

        results[str(K)] = metrics
        print(f"  K={K} → F1={metrics['f1']:.4f} CCR={ccr_kb:.1f} KB/round")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Privacy & Scalability Experiments")
    p.add_argument("--dataset",       type=str,   default="devign")
    p.add_argument("--max_samples",   type=int,   default=8000)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--num_clients",   type=int,   default=4)
    p.add_argument("--rounds",        type=int,   default=10)
    p.add_argument("--local_epochs",  type=int,   default=2)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--num_cwes",      type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_privacy",    type=str, default="results/rq3_privacy.json")
    p.add_argument("--output_scalability",type=str, default="results/rq4_scalability.json")
    p.add_argument("--skip_privacy",      action="store_true")
    p.add_argument("--skip_scalability",  action="store_true")
    args = p.parse_args()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    if not args.skip_privacy:
        privacy_results = run_privacy_sweep(args)
        out = Path(__file__).parent / args.output_privacy
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(privacy_results, f, indent=2)
        print(f"\nPrivacy sweep saved → {out}")

        print(f"\n{'='*55}")
        print("RQ3: PRIVACY-UTILITY TRADEOFF")
        print(f"{'='*55}")
        print(f"{'ε':<8} | {'F1':>7} | {'AUC':>7} | {'Precision':>10} | {'Recall':>7}")
        print(f"{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*7}")
        for eps_key, m in privacy_results.items():
            print(f"{eps_key:<8} | {m['f1']:>7.4f} | {m['auc']:>7.4f} | "
                  f"{m['precision']:>10.4f} | {m['recall']:>7.4f}")

    if not args.skip_scalability:
        scale_results = run_scalability_sweep(args)
        out = Path(__file__).parent / args.output_scalability
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(scale_results, f, indent=2)
        print(f"\nScalability results saved → {out}")

        print(f"\n{'='*60}")
        print("RQ4: SCALABILITY")
        print(f"{'='*60}")
        print(f"{'K':>4} | {'F1':>7} | {'AUC':>7} | {'CCR (KB/rnd)':>13}")
        print(f"{'':>4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*13}")
        for k, m in scale_results.items():
            print(f"{k:>4} | {m['f1']:>7.4f} | {m['auc']:>7.4f} | {m.get('ccr_kb', 0):>13.1f}")


if __name__ == "__main__":
    main()
