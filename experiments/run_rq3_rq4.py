"""
Privacy sweep and scalability experiments for VulMorph-Fed.

RQ3: Privacy-Utility Tradeoff — F1 vs epsilon curve
     per-round ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, ∞}; the composed end-to-end
     budget over T rounds (sequential composition, T·ε) is reported alongside.

RQ4: Scalability — performance as K ∈ {3, 5, 10, 20} clients.
     Communication cost per round is reported both per client
     (upload + download of one prototype bank = 2·|C|·d·4 bytes) and in
     total at the server (K times the per-client cost).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse

from main import run_fl
from utils.privacy import composed_epsilon
from experiments.common import make_args, run_seeded, add_common_cli, parse_seeds


def _model_kwargs(use_dp=True):
    return dict(use_vcsa=True, use_mgmp=True, use_morphology=True,
                use_cwe_affinity=True, use_dp=use_dp,
                federate=True, num_layers=2, dropout=0.3)


# ── RQ3: Privacy Sweep ───────────────────────────────────────────────────────

def run_privacy_sweep(args, seeds):
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
    results = {}

    for eps in epsilons:
        key = "inf" if eps == float('inf') else str(eps)
        print(f"\n{'='*50}\nPrivacy Sweep ε={key}\n{'='*50}")

        def run_one(seed, eps=eps):
            run_args = make_args(
                seed=seed, num_clients=args.num_clients, rounds=args.rounds,
                local_epochs=args.local_epochs, epsilon=eps,
                dataset=args.dataset, max_samples=args.max_samples,
                test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim, num_cwes=args.num_cwes,
                taxonomy_size=args.taxonomy_size, device=args.device,
            )
            return run_fl(run_args, _model_kwargs(use_dp=(eps != float('inf'))))

        agg = run_seeded(run_one, seeds)
        agg["epsilon_per_round"] = key
        agg["epsilon_composed"] = (
            "inf" if eps == float('inf')
            else composed_epsilon(eps, args.rounds)
        )
        results[key] = agg

    return results


# ── RQ4: Scalability Sweep ───────────────────────────────────────────────────

def run_scalability_sweep(args, seeds):
    client_counts = [3, 5, 10, 20]
    results = {}

    for K in client_counts:
        print(f"\n{'='*50}\nScalability K={K} clients\n{'='*50}")

        def run_one(seed, K=K):
            run_args = make_args(
                seed=seed, num_clients=K, rounds=args.rounds,
                local_epochs=args.local_epochs, epsilon=2.0,
                dataset=args.dataset, max_samples=args.max_samples,
                test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim, num_cwes=args.num_cwes,
                taxonomy_size=args.taxonomy_size, device=args.device,
            )
            return run_fl(run_args, _model_kwargs(use_dp=True))

        agg = run_seeded(run_one, seeds)

        # Communication cost per round (float32 prototypes, |C| x d bank):
        bank_bytes = (args.num_cwes + 1) * args.hidden_dim * 4
        agg["ccr_client_kb"] = round(2 * bank_bytes / 1024, 2)      # up + down
        agg["ccr_server_total_kb"] = round(2 * bank_bytes * K / 1024, 2)
        agg["num_clients"] = K
        results[str(K)] = agg

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Privacy & Scalability Experiments")
    add_common_cli(p)
    p.add_argument("--output_privacy",     type=str, default="results/rq3_privacy.json")
    p.add_argument("--output_scalability", type=str, default="results/rq4_scalability.json")
    p.add_argument("--skip_privacy",       action="store_true")
    p.add_argument("--skip_scalability",   action="store_true")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    if not args.skip_privacy:
        privacy_results = run_privacy_sweep(args, seeds)
        out = Path(__file__).parent / args.output_privacy
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(privacy_results, f, indent=2)
        print(f"\nPrivacy sweep saved → {out}")

        print(f"\n{'='*70}\nRQ3: PRIVACY-UTILITY TRADEOFF (mean ± std)\n{'='*70}")
        for eps_key, m in privacy_results.items():
            f1 = m.get("f1", {})
            print(f"ε/round={eps_key:<6} (composed={m['epsilon_composed']}) | "
                  f"F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}")

    if not args.skip_scalability:
        scale_results = run_scalability_sweep(args, seeds)
        out = Path(__file__).parent / args.output_scalability
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(scale_results, f, indent=2)
        print(f"\nScalability results saved → {out}")

        print(f"\n{'='*70}\nRQ4: SCALABILITY (mean ± std)\n{'='*70}")
        for k, m in scale_results.items():
            f1 = m.get("f1", {})
            print(f"K={k:>3} | F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f} | "
                  f"client CCR={m['ccr_client_kb']} KB | "
                  f"server total={m['ccr_server_total_kb']} KB")


if __name__ == "__main__":
    main()
