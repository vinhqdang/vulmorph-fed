"""
Centralised and Federated Baseline Runners for VulMorph-Fed (RQ1 / RQ2c).

Baseline suite (all trained with the same data, split, class weighting and
budget as VulMorph-Fed, over the same seeds):

  Centralised (pooled data, no privacy):
    - centralised_ggnn         GGNN on full lexical graphs  (Devign-style)
    - centralised_gat          GAT  on full lexical graphs  (CPVD-style)
    - centralised_transformer  Transformer encoder on token sequences
                               (structure-free sequence-model family)
    - centralised_ggnn_morph   GGNN on VCSA-abstracted (morphology) graphs
                               → isolates the representation from the FL protocol
    - centralised_vulmorph     Full VulMorph model, pooled data, no federation,
                               no DP → the "centralised oracle" for our method

  Federated:
    - fedavg_gat               FedAvg + GAT on full lexical graphs
    - fedavg_ggnn_morph        FedAvg + GGNN on VCSA-abstracted graphs
                               → same backbone AND same representation as the
                               strongest configuration, differing only in the
                               aggregation mechanism (parameters vs prototypes)
    - fedavg_transformer       FedAvg + Transformer on token sequences
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

from data.loaders.real_datasets import (split_by_project, ListDataset,
                                        carve_calibration)
from models.baselines.gnn_baselines import DevignBaseline, GATBaseline
from models.baselines.nlp_baselines import TransformerSeqBaseline
from utils.metrics import compute_metrics, best_f1_threshold
from experiments.common import (make_args, run_seeded, add_common_cli,
                                parse_seeds, load_graphs)
from main import run_fl


# ── Shared helpers ────────────────────────────────────────────────────────

def _make_model(name, vocab_size, embed_dim, hidden_dim, device):
    if name == "ggnn":
        return DevignBaseline(vocab_size, embed_dim, hidden_dim).to(device)
    if name == "gat":
        return GATBaseline(vocab_size, embed_dim, hidden_dim).to(device)
    if name == "transformer":
        return TransformerSeqBaseline(vocab_size, embed_dim, hidden_dim).to(device)
    if name == "ggnn_morph":
        return DevignBaseline(vocab_size, embed_dim, hidden_dim,
                              input_mode="morph").to(device)
    raise ValueError(f"Unknown model: {name}")


def _bce_for(dataset, device):
    n_pos = sum(1 for i in range(len(dataset)) if float(dataset[i].y[0]) == 1.0)
    n_neg = len(dataset) - n_pos
    w = torch.tensor([n_neg / max(1, n_pos)], device=device).clamp(max=20.0)
    return nn.BCEWithLogitsLoss(pos_weight=w)


def _model_probs(model, loader, device):
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _, _ = model(batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_y_true.extend(batch.y.cpu().numpy())
            all_y_pred.extend(probs.cpu().numpy())
    return np.array(all_y_true), np.array(all_y_pred)


def evaluate_model(model, loader, device, cal_loader=None):
    """Test metrics at a threshold calibrated on training-project samples."""
    thr = 0.5
    if cal_loader is not None:
        yc, pc = _model_probs(model, cal_loader, device)
        if len(yc):
            thr = best_f1_threshold(yc, pc)
    y, p = _model_probs(model, loader, device)
    m = compute_metrics(y, p, threshold=thr)
    m["threshold"] = thr
    return m


# ── Centralised training ──────────────────────────────────────────────────

def run_centralised(model_name, train_data, test_data,
                    vocab_size, embed_dim, hidden_dim,
                    epochs, batch_size, lr, device, cal_data=None):
    model = _make_model(model_name, vocab_size, embed_dim, hidden_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = _bce_for(train_data, device)
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
    cal_loader = (PyGDataLoader(cal_data, batch_size=batch_size, shuffle=False)
                  if cal_data is not None else None)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits, _, _ = model(batch)
            loss = bce(logits.squeeze(-1), batch.y)
            loss.backward()
            opt.step()

    metrics = evaluate_model(model, test_loader, device, cal_loader)
    print(f"  Centralised {model_name} → F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
    return metrics


# ── FedAvg ────────────────────────────────────────────────────────────────

def run_fedavg(model_name, client_datasets, test_data,
               vocab_size, embed_dim, hidden_dim,
               rounds, local_epochs, batch_size, lr, device, cal_data=None,
               dp_epsilon=None, dp_clip=1.0):
    """
    Standard FedAvg; when dp_epsilon is set, applies the Laplace mechanism to
    each client's model UPDATE (DP-FedAvg): the update Δθ = θ_local − θ_global
    is L1-clipped to `dp_clip` and perturbed with per-coordinate Laplace noise
    of scale dp_clip/ε, giving ε-DP per round at client level — the standard
    way to obtain a formal guarantee for parameter-sharing FL, and the
    matched-privacy-budget counterpart of VulMorph-Fed's prototype mechanism.
    """
    global_model = _make_model(model_name, vocab_size, embed_dim, hidden_dim, device)
    test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
    cal_loader = (PyGDataLoader(cal_data, batch_size=batch_size, shuffle=False)
                  if cal_data is not None else None)

    sizes = [len(ds) for ds in client_datasets]

    for r in range(rounds):
        client_weights, client_sizes = [], []

        for ds in client_datasets:
            if len(ds) == 0:
                continue
            local_model = copy.deepcopy(global_model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr)
            bce = _bce_for(ds, device)
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

            local_sd = local_model.state_dict()
            if dp_epsilon is not None:
                # DP-FedAvg: clip + noise the UPDATE, then reconstruct weights
                global_sd = global_model.state_dict()
                delta = {k: (local_sd[k].float() - global_sd[k].float())
                         for k in local_sd}
                l1 = sum(d.abs().sum() for d in delta.values()).clamp(min=1e-12)
                scale = min(1.0, dp_clip / float(l1))
                b = dp_clip / dp_epsilon
                noised = {}
                for k, d in delta.items():
                    d = d * scale
                    noise = torch.tensor(
                        np.random.laplace(0.0, b, size=d.shape),
                        dtype=d.dtype, device=d.device)
                    noised[k] = (global_sd[k].float() + d + noise
                                 ).to(local_sd[k].dtype)
                local_sd = noised

            client_weights.append(copy.deepcopy(local_sd))
            client_sizes.append(len(ds))

        if not client_weights:
            continue

        # Sample-size-weighted FedAvg aggregation
        total = sum(client_sizes)
        avg_w = {}
        for key in client_weights[0]:
            acc = None
            for w, n in zip(client_weights, client_sizes):
                term = w[key].float() * (n / total)
                acc = term if acc is None else acc + term
            avg_w[key] = acc.to(client_weights[0][key].dtype)
        global_model.load_state_dict(avg_w)

    final = evaluate_model(global_model, test_loader, device, cal_loader)
    print(f"  FedAvg+{model_name} FINAL → F1={final['f1']:.4f} AUC={final['auc']:.4f}")
    return final


# ── Centralised VulMorph (oracle) ─────────────────────────────────────────

def run_centralised_vulmorph(args, seed):
    """Full VulMorph model on pooled data: one client, no DP."""
    run_args = make_args(
        seed=seed, num_clients=1, rounds=args.rounds,
        local_epochs=args.local_epochs, epsilon=float('inf'),
        dataset=args.dataset, max_samples=args.max_samples,
        test_fraction=args.test_fraction, hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim, num_cwes=args.num_cwes,
        taxonomy_size=args.taxonomy_size, device=args.device,
    )
    model_kwargs = dict(use_vcsa=True, use_mgmp=True, use_morphology=True,
                        use_cwe_affinity=True, use_dp=False,
                        federate=True, num_layers=2, dropout=0.3)
    return run_fl(run_args, model_kwargs)


# ── CLI ──────────────────────────────────────────────────────────────────

BASELINES = [
    # Non-private references (raw data pooled, or parameters shared in clear)
    ("centralised_ggnn",        "centralised", "ggnn"),
    ("centralised_gat",         "centralised", "gat"),
    ("centralised_transformer", "centralised", "transformer"),
    ("centralised_ggnn_morph",  "centralised", "ggnn_morph"),
    ("fedavg_gat",              "fedavg",      "gat"),
    ("fedavg_ggnn_morph",       "fedavg",      "ggnn_morph"),
    ("fedavg_transformer",      "fedavg",      "transformer"),
    # Matched-privacy-budget peers (formal ε-DP per round, like VulMorph-Fed)
    ("dp_fedavg_gat",           "dp_fedavg",   "gat"),
    ("dp_fedavg_ggnn_morph",    "dp_fedavg",   "ggnn_morph"),
]


def _save(results, output):
    out = Path(__file__).parent / output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="VulMorph-Fed Baseline Runner")
    add_common_cli(p)
    p.add_argument("--epochs",     type=int,   default=10,
                   help="Epochs for centralised training")
    p.add_argument("--vocab_size", type=int,   default=10000)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output",     type=str,   default="results/baselines.json")
    p.add_argument("--only",       type=str,   default=None,
                   help="Comma-separated subset of baseline names to run")
    p.add_argument("--dp_epsilon", type=float, default=2.0,
                   help="Per-round ε for the DP-FedAvg baselines (matches "
                        "VulMorph-Fed's per-round prototype budget)")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    only = set(args.only.split(",")) if args.only else None
    results = {}

    def load_split(seed):
        data_list = load_graphs(args.dataset, args.max_samples,
                                args.taxonomy_size)
        client_buckets, test_raw = split_by_project(
            data_list, num_clients=args.num_clients,
            test_fraction=args.test_fraction, seed=seed,
        )
        client_buckets, cal_raw = carve_calibration(client_buckets, seed=seed)
        return ([ListDataset(b) for b in client_buckets],
                ListDataset(test_raw),
                ListDataset([d for b in client_buckets for d in b]),
                ListDataset(cal_raw))

    for name, mode, model_name in BASELINES:
        if only and name not in only:
            continue
        print(f"\n{'='*55}\nBaseline: {name}\n{'='*55}")

        def run_one(seed, mode=mode, model_name=model_name):
            client_datasets, test_dataset, all_train, cal_dataset = load_split(seed)
            if mode == "centralised":
                return run_centralised(
                    model_name, all_train, test_dataset,
                    vocab_size=args.vocab_size, embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, device=args.device,
                    cal_data=cal_dataset)
            return run_fedavg(
                model_name, client_datasets, test_dataset,
                vocab_size=args.vocab_size, embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim, rounds=args.rounds,
                local_epochs=args.local_epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device, cal_data=cal_dataset,
                dp_epsilon=(args.dp_epsilon if mode == "dp_fedavg" else None))

        results[name] = run_seeded(run_one, seeds)
        _save(results, args.output)

    if (only is None) or ("centralised_vulmorph" in only):
        print(f"\n{'='*55}\nBaseline: centralised_vulmorph (oracle)\n{'='*55}")
        results["centralised_vulmorph"] = run_seeded(
            lambda s: run_centralised_vulmorph(args, s), seeds)
        _save(results, args.output)

    out = Path(__file__).parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaselines saved → {out}")

    print(f"\n{'='*70}\nBASELINE RESULTS (mean ± std)\n{'='*70}")
    for k, v in results.items():
        f1, auc = v.get("f1", {}), v.get("auc", {})
        print(f"{k:<28} | F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f} "
              f"| AUC={auc.get('mean', 0):.4f}±{auc.get('std', 0):.4f}")


if __name__ == "__main__":
    main()
