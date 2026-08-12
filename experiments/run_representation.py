"""
E1 - the core experiment: representation x backbone factorial.

Trains every combination of node-feature mode and backbone under one identical
protocol, on project-level GroupKFold folds, and reports AUC/AUPRC with
cluster-bootstrap confidence intervals on the paired difference against the
lexical baseline.

Everything except the factor under study is held fixed: depth, hidden width,
readout, dropout, optimiser, gradient-step budget, class weighting and the
evaluation split. A difference between cells is therefore attributable to the
factor that varies, which is what the previous version of this study could
not claim.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import itertools

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.loaders.real_datasets import ListDataset, dedupe_functions
from data.morphology import get_taxonomy
from data.loaders.ast_graphs import NUM_AST_KINDS
from models.encoders import RepresentationModel, FEATURE_MODES, BACKBONES
from utils.protocol import (project_group_kfold, cluster_bootstrap_ci,
                            trivial_all_positive)
from experiments.common import load_graphs


def train_eval(model, train_data, test_data, device, epochs, batch_size, lr):
    """Fixed protocol shared by every cell of the grid."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = sum(1 for d in train_data if float(d.y[0]) == 1.0)
    n_neg = len(train_data) - n_pos
    pos_w = torch.tensor([n_neg / max(1, n_pos)], device=device).clamp(max=20.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    tl = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    vl = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            logits, _, _ = model(batch)
            bce(logits.squeeze(-1), batch.y).backward()
            opt.step()

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            logits, _, _ = model(batch)
            ys.extend(batch.y.cpu().numpy())
            ps.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    return np.asarray(ys), np.asarray(ps)


def main():
    p = argparse.ArgumentParser(description="Representation x backbone study")
    p.add_argument("--dataset", type=str, default="bigvul")
    p.add_argument("--max_samples", type=int, default=8000)
    p.add_argument("--taxonomy_size", type=int, default=8)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--modes", type=str, default=",".join(FEATURE_MODES))
    p.add_argument("--backbones", type=str, default=",".join(BACKBONES))
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    backbones = [b for b in args.backbones.split(",") if b]

    data = load_graphs(args.dataset, args.max_samples, args.taxonomy_size)
    data = dedupe_functions(data)
    num_op_types = len(get_taxonomy(args.taxonomy_size)[0])

    out_path = Path(__file__).parent / (
        args.output or f"results/representation_{args.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}

    folds = list(project_group_kfold(data, args.n_splits, args.n_repeats))
    print(f"{args.dataset}: {len(data)} functions, {len(folds)} folds")
    if folds:
        fr = [f[2]["test_fraction"] for f in folds]
        sh = [f[2]["largest_test_project_share"] for f in folds]
        print(f"  test fraction {min(fr):.2f}-{max(fr):.2f}; "
              f"largest test project share {min(sh):.2f}-{max(sh):.2f}")
        results["_folds"] = [f[2] for f in folds]

    for fi, (tr, te, info) in enumerate(folds):
        train_data = [data[i] for i in tr]
        test_data = [data[i] for i in te]
        groups = [getattr(data[i], "project", "?") for i in te]

        for mode, bb in itertools.product(modes, backbones):
            key = f"{mode}|{bb}"
            cell = results.setdefault(key, {"auc": [], "auprc": [], "folds": []})
            if fi in cell["folds"]:
                continue
            torch.manual_seed(1000 + fi)
            np.random.seed(1000 + fi)
            model = RepresentationModel(
                feature_mode=mode, backbone=bb, vocab_size=10000,
                num_op_types=num_op_types, num_kinds=NUM_AST_KINDS,
                embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers)
            y, s = train_eval(model, train_data, test_data, args.device,
                              args.epochs, args.batch_size, args.lr)
            auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
            ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0
            cell["auc"].append(float(auc))
            cell["auprc"].append(float(ap))
            cell["folds"].append(fi)
            cell.setdefault("scores", {})[str(fi)] = {
                "y": [float(v) for v in y], "s": [float(v) for v in s],
                "groups": list(groups)}
            print(f"  fold {fi:>2} {key:<18} AUC={auc:.4f} AUPRC={ap:.4f}")
            json.dump(results, open(out_path, "w"))

        results.setdefault("_trivial", []).append(
            trivial_all_positive([float(d.y[0]) for d in test_data]))
        json.dump(results, open(out_path, "w"))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
