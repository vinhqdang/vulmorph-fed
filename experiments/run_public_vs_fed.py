"""
Public-data-only vs. federated-increment experiment (RQ1b).

Addresses the question: *what does a private federated client contribute
that public corpora do not already supply?*

Protocol (same held-out cross-project test set in both conditions):
  A. PUBLIC-ONLY : train the full VulMorph model centrally on a "public"
                   partition (a fraction of the training projects/samples).
  B. PUBLIC+FED  : the same public partition is one client; the remaining
                   training data is split across K-1 "private" clients that
                   participate ONLY through DP-noised prototype federation
                   (no raw data pooling).

The delta (B - A) is the measured value of federation on top of the public
corpus.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import random

import numpy as np
import torch

from data.loaders.real_datasets import (split_by_project, ListDataset,
                                        bucket_cwes, carve_calibration,
                                        downsample_benign)
from fl.client import VulMorphClient
from fl.server import VulMorphServer
from main import evaluate
from experiments.common import add_common_cli, parse_seeds, run_seeded, load_graphs


def _train_federation(client_datasets, test_dataset, args, use_dp=True,
                      epsilon=2.0, cal_dataset=None):
    model_kwargs = dict(use_vcsa=True, use_mgmp=True, use_morphology=True,
                        num_layers=2, dropout=0.3)
    clients = [
        VulMorphClient(
            client_id=i, dataset=ds, vocab_size=10000,
            embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            num_cwes=args.num_cwes, device=args.device, batch_size=64, lr=1e-3,
            use_dp=use_dp, **model_kwargs,
        )
        for i, ds in enumerate(client_datasets)
    ]
    server = VulMorphServer(num_cwes=args.num_cwes + 1, hidden_dim=args.hidden_dim,
                            device=args.device, use_cwe_affinity=True)

    metrics = {}
    for _ in range(args.rounds):
        protos = []
        for c in clients:
            c.train_local(global_prototypes=server.global_prototypes,
                          epochs=args.local_epochs, alpha=0.1, gamma=0.01)
            protos.append(c.get_noisy_prototypes(epsilon=epsilon, delta_f=1.0))
        server.aggregate_prototypes(protos)
    metrics = evaluate(clients, server.global_prototypes, test_dataset,
                       cal_dataset)
    return metrics


def run_condition(seed, args, public_fraction=0.5):
    rng = random.Random(seed)
    data_list = load_graphs(args.dataset, args.max_samples, args.taxonomy_size)
    bucket_cwes(data_list, args.num_cwes)
    client_buckets, test_raw = split_by_project(
        data_list, num_clients=1, test_fraction=args.test_fraction, seed=seed)
    train_all = [d for b in client_buckets for d in b]

    # The public/private partition must be PROJECT-DISJOINT: if the two halves
    # were IID samples of one pool they would be exchangeable by construction,
    # and the measured delta could not represent "structural evidence the
    # public corpus lacks". We therefore split whole repositories.
    by_proj = {}
    for d in train_all:
        by_proj.setdefault(getattr(d, "project", "unknown"), []).append(d)
    projs = sorted(by_proj)
    rng.shuffle(projs)

    target = public_fraction * len(train_all)
    public_projs, count = [], 0
    for p in projs:
        if count >= target and len(public_projs) < len(projs) - 1:
            break
        public_projs.append(p)
        count += len(by_proj[p])
    public_set = set(public_projs)
    public = [d for p in public_projs for d in by_proj[p]]
    private = [d for p in projs if p not in public_set for d in by_proj[p]]
    print(f"    public: {len(public_projs)} projects / {len(public)} fns; "
          f"private: {len(projs) - len(public_projs)} projects / {len(private)} fns")

    # Calibration slice comes from the public partition (always available to
    # the model owner) and keeps true prevalence.
    rng.shuffle(public)
    n_cal = max(1, int(0.1 * len(public)))
    cal_raw, public = public[:n_cal], public[n_cal:]

    public = downsample_benign(public, 4.0, seed)
    private = downsample_benign(private, 4.0, seed)

    n_private_clients = max(1, args.num_clients - 1)
    chunk = max(1, len(private) // n_private_clients)
    private_buckets = [private[i * chunk:(i + 1) * chunk]
                       for i in range(n_private_clients)]
    private_buckets[-1].extend(private[n_private_clients * chunk:])

    test_dataset = ListDataset(test_raw)
    cal_dataset = ListDataset(cal_raw)

    # A: public-only. Sharded across the SAME number of clients as condition B
    # and evaluated with the same ensemble protocol, so the measured delta
    # reflects private participation rather than an ensembling artifact.
    n_clients_b = 1 + len([b for b in private_buckets if b])
    chunk = max(1, len(public) // n_clients_b)
    public_shards = [public[i * chunk:(i + 1) * chunk] for i in range(n_clients_b)]
    public_shards[-1].extend(public[n_clients_b * chunk:])
    print(f"  Condition A: public-only, {n_clients_b} shards (no DP — data is public)")
    m_public = _train_federation([ListDataset(s) for s in public_shards if s],
                                 test_dataset, args,
                                 use_dp=False, epsilon=float('inf'),
                                 cal_dataset=cal_dataset)

    # B: public client + private clients, DP prototype federation
    print("  Condition B: public + federated private clients")
    datasets = [ListDataset(public)] + [ListDataset(b) for b in private_buckets
                                        if b]
    m_fed = _train_federation(datasets, test_dataset, args,
                              use_dp=True, epsilon=2.0,
                              cal_dataset=cal_dataset)

    return {
        "public_only_f1": m_public["f1"], "public_only_auc": m_public["auc"],
        "public_fed_f1": m_fed["f1"], "public_fed_auc": m_fed["auc"],
        "delta_f1": m_fed["f1"] - m_public["f1"],
        "delta_auc": m_fed["auc"] - m_public["auc"],
    }


def main():
    p = argparse.ArgumentParser(description="Public-only vs public+federated (RQ1b)")
    add_common_cli(p)
    p.add_argument("--public_fraction", type=float, default=0.5)
    p.add_argument("--output", type=str, default="results/public_vs_fed.json")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    results = run_seeded(lambda s: run_condition(s, args, args.public_fraction),
                         seeds)

    print(f"\n{'='*60}\nPUBLIC-ONLY vs PUBLIC+FEDERATED (mean ± std)\n{'='*60}")
    for key in ["public_only_f1", "public_fed_f1", "delta_f1"]:
        m = results.get(key, {})
        print(f"{key:<18}: {m.get('mean', 0):.4f} ± {m.get('std', 0):.4f}")

    out = Path(__file__).parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
