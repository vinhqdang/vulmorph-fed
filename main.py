import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import get_client_datasets
from fl.client import VulMorphClient
from fl.server import VulMorphServer
from utils.metrics import compute_metrics


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def build_clients(args, datasets, model_kwargs):
    return [
        VulMorphClient(
            client_id=i,
            dataset=datasets[i],
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_cwes=args.num_cwes,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            use_dp=model_kwargs.pop('use_dp', True),
            **model_kwargs,
        )
        for i in range(args.num_clients)
    ]


def evaluate(clients, global_prototypes):
    """Evaluate all local models; return aggregated classification metrics."""
    all_y_true, all_y_pred = [], []

    for client in clients:
        client.model.eval()
        proto = global_prototypes.to(client.device) if global_prototypes is not None else None

        with torch.no_grad():
            for batch in client.train_loader:
                batch = batch.to(client.device)
                logits, _, _ = client.model(batch, prototypes=proto)
                probs = torch.sigmoid(logits.squeeze(-1))
                all_y_true.extend(batch.y.cpu().numpy())
                all_y_pred.extend(probs.cpu().numpy())

    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# ───────────────────────────────────────────────────────────────────────────
# Federated training loop
# ───────────────────────────────────────────────────────────────────────────

def run_fl(args, model_kwargs=None):
    """
    Main federated learning simulation loop.

    Each round:
      1. Clients train locally.
      2. Upload DP-noised prototypes to the server.
      3. Server aggregates via MCFPA.
      4. Clients receive updated P*.

    Returns final evaluation metrics dict.
    """
    if model_kwargs is None:
        model_kwargs = {}

    datasets = get_client_datasets(
        total_graphs=args.total_graphs,
        num_clients=args.num_clients,
        num_cwes=args.num_cwes,
    )

    use_dp = model_kwargs.pop('use_dp', True)
    use_cwe_affinity = model_kwargs.pop('use_cwe_affinity', True)
    federate = model_kwargs.pop('federate', True)

    clients = [
        VulMorphClient(
            client_id=i,
            dataset=datasets[i],
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_cwes=args.num_cwes,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            use_dp=use_dp,
            **model_kwargs,
        )
        for i in range(args.num_clients)
    ]

    server = VulMorphServer(
        num_cwes=args.num_cwes,
        hidden_dim=args.hidden_dim,
        device=args.device,
        use_cwe_affinity=use_cwe_affinity,
    )

    history = []

    for r in range(args.rounds):
        client_protos = []

        for client in tqdm(clients, desc=f"Round {r+1}/{args.rounds} – training", leave=False):
            client.train_local(
                global_prototypes=server.global_prototypes,
                epochs=args.local_epochs,
                alpha=args.alpha,
                gamma=args.gamma,
            )
            if federate:
                protos = client.get_noisy_prototypes(
                    epsilon=args.epsilon, delta_f=args.delta_f
                )
                client_protos.append(protos)

        if federate and client_protos:
            server.aggregate_prototypes(client_protos)

        metrics = evaluate(clients, server.global_prototypes)
        history.append(metrics)
        print(
            f"  Round {r+1:>2} | F1={metrics['f1']:.4f} "
            f"AUC={metrics['auc']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f}"
        )

    return history[-1] if history else {}


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VulMorph-Fed Simulation")

    # Federated setup
    p.add_argument("--num_clients",   type=int,   default=5)
    p.add_argument("--rounds",        type=int,   default=10)
    p.add_argument("--local_epochs",  type=int,   default=2)
    p.add_argument("--total_graphs",  type=int,   default=1000)

    # Model architecture
    p.add_argument("--vocab_size",    type=int,   default=1000)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--num_cwes",      type=int,   default=10)
    p.add_argument("--num_layers",    type=int,   default=2)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.3)

    # Loss weights
    p.add_argument("--alpha",         type=float, default=0.1,
                   help="Weight for L_SCL contrastive loss")
    p.add_argument("--gamma",         type=float, default=0.01,
                   help="Weight for L1 edge-sparsity loss")

    # Privacy
    p.add_argument("--epsilon",       type=float, default=2.0,
                   help="Laplace DP privacy budget ε")
    p.add_argument("--delta_f",       type=float, default=0.1,
                   help="Global L2-sensitivity Δf")

    # Ablation flags
    p.add_argument("--no_vcsa",       action="store_true")
    p.add_argument("--no_mgmp",       action="store_true")
    p.add_argument("--no_morphology", action="store_true")
    p.add_argument("--no_cwe_affinity", action="store_true")
    p.add_argument("--no_dp",         action="store_true")
    p.add_argument("--local_only",    action="store_true",
                   help="Each client trains in isolation (no federation)")

    # Misc
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output",        type=str,   default=None,
                   help="JSON file to save final metrics")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("VulMorph-Fed Simulation")
    print(f"  Clients={args.num_clients}  Rounds={args.rounds}  ε={args.epsilon}")
    ablation_flags = []
    if args.no_vcsa:         ablation_flags.append("w/o VCSA")
    if args.no_mgmp:         ablation_flags.append("w/o MGMP")
    if args.no_morphology:   ablation_flags.append("w/o Morphology")
    if args.no_cwe_affinity: ablation_flags.append("w/o CWE-Affinity")
    if args.no_dp:           ablation_flags.append("w/o DP")
    if args.local_only:      ablation_flags.append("Local-Only")
    if ablation_flags:
        print(f"  Ablation: {', '.join(ablation_flags)}")
    print("=" * 60)

    model_kwargs = dict(
        use_vcsa=not args.no_vcsa,
        use_mgmp=not args.no_mgmp,
        use_morphology=not args.no_morphology,
        use_cwe_affinity=not args.no_cwe_affinity,
        use_dp=not args.no_dp,
        federate=not args.local_only,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    metrics = run_fl(args, model_kwargs)

    print("\nFinal Metrics")
    print(f"  F1        : {metrics.get('f1', 0):.4f}")
    print(f"  AUC       : {metrics.get('auc', 0):.4f}")
    print(f"  Precision : {metrics.get('precision', 0):.4f}")
    print(f"  Recall    : {metrics.get('recall', 0):.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved → {args.output}")


if __name__ == "__main__":
    main()
