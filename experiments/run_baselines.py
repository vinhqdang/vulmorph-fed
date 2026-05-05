"""
Centralised and Federated Baseline Runners for VulMorph-Fed.

Baselines from plan.md §6:
  §6.1 Centralised GNN: Devign (GGNN), GATBaseline (CPVD-style)
  §6.3 Federated:       FedAvg + GAT (standard weight-averaging FL)

All baselines use the same Devign data and cross-project split
as VulMorph-Fed for a fair comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from data.loaders.real_datasets import (
    load_devign, split_by_project, ListDataset
)
from models.baselines.gnn_baselines import DevignBaseline, GATBaseline
from utils.metrics import compute_metrics


# ── Shared evaluation ─────────────────────────────────────────────────────

def evaluate_model(model, loader, device):
    model.eval()
    all_y_true, all_y_pred = [], []
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _, _ = model(batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_y_true.extend(batch.y.cpu().numpy())
            all_y_pred.extend(probs.cpu().numpy())
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# ── Centralised training ──────────────────────────────────────────────────

def run_centralised(model_name, train_data, test_data,
                    vocab_size, embed_dim, hidden_dim,
                    epochs, batch_size, lr, device):
    """
    Train a centralised GNN on the full pooled training data.
    Simulates the 'centralised oracle' and centralised baselines.
    """
    if model_name == "devign":
        model = DevignBaseline(vocab_size=vocab_size, embed_dim=embed_dim,
                               hidden_dim=hidden_dim).to(device)
    elif model_name == "gat":
        model = GATBaseline(vocab_size=vocab_size, embed_dim=embed_dim,
                            hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = PyGDataLoader(test_data,  batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(epochs), desc=f"Centralised {model_name}"):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits, _, _ = model(batch)
            loss = bce(logits.squeeze(-1), batch.y)
            loss.backward()
            opt.step()

    metrics = evaluate_model(model, test_loader, device)
    print(f"  Centralised {model_name} → F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
    return metrics


# ── FedAvg + GAT ──────────────────────────────────────────────────────────

def run_fedavg_gat(client_datasets, test_data,
                   vocab_size, embed_dim, hidden_dim,
                   rounds, local_epochs, batch_size, lr, device):
    """
    Standard FedAvg with GATBaseline — weight-averaging FL baseline.
    Simulates VulFL-GNN / FedProx + GAT from plan.md §6.3.
    """
    # Initialise one global model
    global_model = GATBaseline(vocab_size=vocab_size, embed_dim=embed_dim,
                                hidden_dim=hidden_dim).to(device)
    bce = nn.BCEWithLogitsLoss()
    test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)

    for r in range(rounds):
        client_weights = []

        for ds in client_datasets:
            if len(ds) == 0:
                continue
            # Clone global model to client
            local_model = copy.deepcopy(global_model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr)
            loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=True)

            local_model.train()
            for _ in range(local_epochs):
                for batch in loader:
                    batch = batch.to(device)
                    opt.zero_grad()
                    logits, _, _ = local_model(batch)
                    loss = bce(logits.squeeze(-1), batch.y)
                    loss.backward()
                    opt.step()

            client_weights.append(copy.deepcopy(local_model.state_dict()))

        # FedAvg aggregation
        if not client_weights:
            continue
        avg_w = copy.deepcopy(client_weights[0])
        for key in avg_w:
            for cw in client_weights[1:]:
                avg_w[key] = avg_w[key] + cw[key]
            if isinstance(avg_w[key], torch.Tensor):
                avg_w[key] = torch.div(avg_w[key], len(client_weights))
            else:
                avg_w[key] /= len(client_weights)
        global_model.load_state_dict(avg_w)

        metrics = evaluate_model(global_model, test_loader, device)
        print(f"  FedAvg+GAT Round {r+1:>2} | F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    final = evaluate_model(global_model, test_loader, device)
    print(f"  FedAvg+GAT FINAL → F1={final['f1']:.4f} AUC={final['auc']:.4f}")
    return final


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VulMorph-Fed Baseline Runner")
    p.add_argument("--max_samples",  type=int,   default=8000)
    p.add_argument("--num_clients",  type=int,   default=4)
    p.add_argument("--rounds",       type=int,   default=10)
    p.add_argument("--local_epochs", type=int,   default=2)
    p.add_argument("--epochs",       type=int,   default=10,
                   help="Epochs for centralised training")
    p.add_argument("--vocab_size",   type=int,   default=10000)
    p.add_argument("--embed_dim",    type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--test_fraction",type=float, default=0.2)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       type=str,   default="results/baselines.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading Devign...")
    data_list = load_devign(max_samples=args.max_samples)
    client_buckets, test_raw = split_by_project(
        data_list, num_clients=args.num_clients,
        test_fraction=args.test_fraction, seed=args.seed
    )
    client_datasets = [ListDataset(b) for b in client_buckets]
    test_dataset = ListDataset(test_raw)
    all_train = ListDataset([d for b in client_buckets for d in b])

    results = {}

    # Centralised baselines
    for name in ["devign", "gat"]:
        print(f"\n{'='*50}\nBaseline: Centralised {name.upper()}\n{'='*50}")
        m = run_centralised(
            name, all_train, test_dataset,
            vocab_size=args.vocab_size, embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, device=args.device
        )
        results[f"centralised_{name}"] = m

    # FedAvg + GAT
    print(f"\n{'='*50}\nBaseline: FedAvg + GAT\n{'='*50}")
    m = run_fedavg_gat(
        client_datasets, test_dataset,
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, rounds=args.rounds,
        local_epochs=args.local_epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device
    )
    results["fedavg_gat"] = m

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaselines saved → {out}")

    # Summary
    print(f"\n{'='*55}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"{'Method':<30} | {'F1':>7} | {'AUC':>7} | {'Prec':>7} | {'Rec':>7}")
    print(f"{'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for k, v in results.items():
        print(f"{k:<30} | {v['f1']:>7.4f} | {v['auc']:>7.4f} | "
              f"{v['precision']:>7.4f} | {v['recall']:>7.4f}")


if __name__ == "__main__":
    main()
