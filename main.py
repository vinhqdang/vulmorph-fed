import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fl.client import VulMorphClient
from fl.server import VulMorphServer
from utils.metrics import compute_metrics
from data.loaders.real_datasets import (
    load_devign, load_primevul, load_bigvul, load_diversevul,
    split_by_project, ListDataset,
)
from data.dataset import get_client_datasets


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(clients, global_prototypes, test_dataset=None):
    """
    Evaluate all local models.
    If test_dataset is provided, only evaluates on held-out cross-project samples.
    Otherwise evaluates on the local training data (used for ablation quick checks).
    """
    all_y_true, all_y_pred = [], []

    if test_dataset is not None:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        loader = PyGDataLoader(test_dataset, batch_size=64, shuffle=False)
        # Use the first client's model for evaluation (or ensemble — for now first)
        client = clients[0]
        client.model.eval()
        proto = global_prototypes.to(client.device) if global_prototypes is not None else None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(client.device)
                logits, _, _ = client.model(batch, prototypes=proto)
                probs = torch.sigmoid(logits.squeeze(-1))
                all_y_true.extend(batch.y.cpu().numpy())
                all_y_pred.extend(probs.cpu().numpy())
    else:
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

    if not all_y_true:
        return {"f1": 0.0, "auc": 0.5, "precision": 0.0, "recall": 0.0}
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_real_data(args):
    """
    Load real datasets as specified in plan.md §4 and return
    (client_datasets, test_dataset) with cross-project split.
    """
    data_list = []

    if args.dataset == "devign":
        data_list = load_devign(max_samples=args.max_samples)
    elif args.dataset == "primevul":
        data_list = load_primevul(split="train", max_samples=args.max_samples)
    elif args.dataset == "bigvul":
        data_list = load_bigvul(args.data_path, max_samples=args.max_samples)
    elif args.dataset == "diversevul":
        data_list = load_diversevul(args.data_path, max_samples=args.max_samples)

    if not data_list:
        print(f"Warning: could not load dataset '{args.dataset}'. "
              "Falling back to structured synthetic data.")
        return None, None

    # Cross-project split: held-out test projects never seen during training
    client_buckets, test_raw = split_by_project(
        data_list,
        num_clients=args.num_clients,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    client_datasets = [ListDataset(bucket) for bucket in client_buckets]
    test_dataset = ListDataset(test_raw)

    return client_datasets, test_dataset


# ── Federated training loop ──────────────────────────────────────────────────

def run_fl(args, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    use_dp           = model_kwargs.pop('use_dp', True)
    use_cwe_affinity = model_kwargs.pop('use_cwe_affinity', True)
    federate         = model_kwargs.pop('federate', True)

    # ── Dataset ─────────────────────────────────────────────────────────
    client_datasets = test_dataset = None

    if getattr(args, 'dataset', 'synthetic') != 'synthetic':
        client_datasets, test_dataset = load_real_data(args)

    if client_datasets is None:
        # Fall back to structured synthetic data
        client_datasets = get_client_datasets(
            total_graphs=args.total_graphs,
            num_clients=args.num_clients,
            num_cwes=args.num_cwes,
        )

    # Adjust num_clients to match actual number of non-empty splits
    actual_num_clients = len(client_datasets)
    if actual_num_clients != args.num_clients:
        print(f"Adjusting num_clients: {args.num_clients} → {actual_num_clients}")
        args = type(args)(**{**vars(args), 'num_clients': actual_num_clients})

    # Infer vocab_size from data if possible
    vocab_size = getattr(args, 'vocab_size', 10000)

    clients = [
        VulMorphClient(
            client_id=i,
            dataset=client_datasets[i],
            vocab_size=vocab_size,
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
        for client in tqdm(clients, desc=f"Round {r+1}/{args.rounds}", leave=False):
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

        # Evaluate on held-out test projects (cross-project F1)
        # or on local training data if no real dataset
        metrics = evaluate(clients, server.global_prototypes, test_dataset)
        history.append(metrics)

        split_name = "cross-project test" if test_dataset else "train (synthetic)"
        print(
            f"  Round {r+1:>2} [{split_name}] | "
            f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
        )

    return history[-1] if history else {}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VulMorph-Fed")

    # Dataset selection (plan.md §4)
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "devign", "primevul", "bigvul", "diversevul"],
                   help="Dataset to use (plan.md §4)")
    p.add_argument("--data_path",    type=str, default=None,
                   help="Path to local CSV/JSONL for bigvul/diversevul")
    p.add_argument("--max_samples",  type=int, default=10000)
    p.add_argument("--test_fraction",type=float, default=0.2,
                   help="Fraction of projects held out for cross-project evaluation")

    # Federated setup
    p.add_argument("--num_clients",  type=int, default=5)
    p.add_argument("--rounds",       type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--total_graphs", type=int, default=5000,
                   help="Total graphs for synthetic mode")

    # Model architecture
    p.add_argument("--vocab_size",   type=int,   default=10000)
    p.add_argument("--embed_dim",    type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--num_cwes",     type=int,   default=150)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--dropout",      type=float, default=0.3)

    # Loss weights
    p.add_argument("--alpha",        type=float, default=0.1)
    p.add_argument("--gamma",        type=float, default=0.01)

    # Privacy
    p.add_argument("--epsilon",      type=float, default=2.0)
    p.add_argument("--delta_f",      type=float, default=0.1)

    # Ablation flags
    p.add_argument("--no_vcsa",       action="store_true")
    p.add_argument("--no_mgmp",       action="store_true")
    p.add_argument("--no_morphology", action="store_true")
    p.add_argument("--no_cwe_affinity", action="store_true")
    p.add_argument("--no_dp",         action="store_true")
    p.add_argument("--local_only",    action="store_true")

    # Misc
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       type=str,   default=None)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("VulMorph-Fed")
    print(f"  Dataset={args.dataset}  Clients={args.num_clients}  "
          f"Rounds={args.rounds}  ε={args.epsilon}")
    print("=" * 60)

    model_kwargs = dict(
        use_vcsa         = not args.no_vcsa,
        use_mgmp         = not args.no_mgmp,
        use_morphology   = not args.no_morphology,
        use_cwe_affinity = not args.no_cwe_affinity,
        use_dp           = not args.no_dp,
        federate         = not args.local_only,
        num_layers       = args.num_layers,
        dropout          = args.dropout,
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
        print(f"Metrics saved → {args.output}")


if __name__ == "__main__":
    main()
