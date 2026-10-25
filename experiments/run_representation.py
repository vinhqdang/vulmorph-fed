"""
E1 - the core experiment: representation x backbone factorial.

Trains every combination of node-feature mode and backbone under one identical
protocol on project-level GroupKFold folds, and stores per-fold AUC/AUPRC plus
the raw held-out scores, so that every downstream statistic (paired bootstrap,
Wilcoxon, per-project and leave-one-project-out summaries) is computed from the
same predictions.

Everything except the factor under study is held fixed: depth, hidden width,
readout, dropout, optimiser, gradient-step budget, class weighting and the
evaluation split. Corpora are simple random samples (seeded), not the first N
rows. Lexical vocabularies are cut to tokens seen in the TRAINING folds only;
anything else is <UNK> at test time.

Hyper-parameters are fixed in advance and are NOT tuned on any fold: there is
no early stopping and no validation split; every cell trains for exactly
`--epochs` epochs. The model initialisation / shuffling seed is 1000 + fold
index and is shared by every cell on that fold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import itertools
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from data.loaders.sampled import load_sampled
from data.loaders.real_datasets import dedupe_functions
from data.morphology import get_taxonomy
from data.loaders.ast_graphs import NUM_AST_KINDS
from models.encoders import (RepresentationModel, FEATURE_MODES, BACKBONES,
                             EDGE_SETS)
from utils.protocol import project_group_kfold, trivial_all_positive

LEX_VOCAB = 10000


def lex_remap_for(train_data, n_tokens, vocab_size=LEX_VOCAB, min_count=2):
    """Corpus-wide token id -> compact id from TRAINING graphs only."""
    cnt = torch.zeros(n_tokens, dtype=torch.long)
    for d in train_data:
        cnt += torch.bincount(d.x_lex, minlength=n_tokens)[:n_tokens]
    cnt[0] = 0
    ids = torch.nonzero(cnt >= min_count).squeeze(-1)
    ids = ids[torch.argsort(-cnt[ids])][:vocab_size - 1]
    remap = torch.zeros(n_tokens, dtype=torch.long)
    remap[ids] = torch.arange(1, len(ids) + 1)
    return remap, int(len(ids))


def _bucketed_batches(data, batch_size):
    order = sorted(range(len(data)),
                   key=lambda i: data[i].num_nodes + 8 * float(torch.rand(1)))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    perm = torch.randperm(len(chunks)).tolist()
    for j in perm:
        yield Batch.from_data_list([data[i] for i in chunks[j]])


def train_eval(model, train_data, test_data, epochs, batch_size, lr):
    """Fixed protocol shared by every cell of the grid."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = sum(1 for d in train_data if float(d.y[0]) == 1.0)
    n_neg = len(train_data) - n_pos
    pos_w = torch.tensor([n_neg / max(1, n_pos)]).clamp(max=20.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    bucketed = getattr(model, "backbone_name", "") == "tf"
    vl = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
    if not bucketed:
        tl = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        if bucketed:
            # Dense-padded attention costs the padded length, so the
            # Transformer is trained on length-bucketed batches (same data,
            # same batch size, same number of steps; batches are shuffled).
            tl = _bucketed_batches(train_data, batch_size)
        for batch in tl:
            opt.zero_grad()
            logits, _, _ = model(batch)
            bce(logits.squeeze(-1), batch.y).backward()
            opt.step()

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in vl:
            logits, _, _ = model(batch)
            ys.extend(batch.y.numpy())
            ps.extend(torch.sigmoid(logits.squeeze(-1)).numpy())
    return np.asarray(ys), np.asarray(ps)


def main():
    p = argparse.ArgumentParser(description="Representation x backbone study")
    p.add_argument("--dataset", type=str, default="bigvul")
    p.add_argument("--n", type=int, default=10000,
                   help="functions drawn uniformly at random")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--taxonomy_size", type=int, default=8)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--edge_set", type=str, default="all",
                   choices=list(EDGE_SETS))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--modes", type=str, default=",".join(FEATURE_MODES))
    p.add_argument("--backbones", type=str, default=",".join(BACKBONES))
    p.add_argument("--folds", type=str, default="",
                   help="comma-separated fold indices (default: all)")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    modes = [m for m in args.modes.split(",") if m]
    backbones = [b for b in args.backbones.split(",") if b]

    data, vocab, anon = load_sampled(args.dataset, args.n, args.sample_seed,
                                     args.taxonomy_size)
    data = dedupe_functions(data)
    num_op_types = len(get_taxonomy(args.taxonomy_size)[0])
    n_tokens = len(vocab)

    out_path = Path(__file__).parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}
    results["_config"] = {k: v for k, v in vars(args).items()
                          if k not in ("output", "folds", "threads")}
    results["_config"]["n_functions_after_dedupe"] = len(data)

    folds = list(project_group_kfold(data, args.n_splits, args.n_repeats))
    only = {int(f) for f in args.folds.split(",") if f}
    print(f"{args.dataset}: {len(data)} functions, {len(folds)} folds")
    results["_folds"] = [f[2] for f in folds]

    for fi, (tr, te, info) in enumerate(folds):
        if only and fi not in only:
            continue
        train_data = [data[i] for i in tr]
        test_data = [data[i] for i in te]
        groups = [getattr(data[i], "project", "?") for i in te]
        remap, n_vocab = lex_remap_for(train_data, n_tokens)
        results.setdefault("_vocab", {})[str(fi)] = n_vocab

        for mode, bb in itertools.product(modes, backbones):
            key = f"{mode}|{bb}"
            cell = results.setdefault(key, {"auc": [], "auprc": [], "folds": []})
            if fi in cell["folds"]:
                continue
            torch.manual_seed(1000 + fi)
            np.random.seed(1000 + fi)
            model = RepresentationModel(
                feature_mode=mode, backbone=bb, vocab_size=LEX_VOCAB,
                num_op_types=num_op_types, num_kinds=NUM_AST_KINDS,
                embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, edge_set=args.edge_set)
            model.features.set_lex_remap(remap)
            t0 = time.time()
            y, s = train_eval(model, train_data, test_data, args.epochs,
                              args.batch_size, args.lr)
            auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
            ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0
            cell["auc"].append(float(auc))
            cell["auprc"].append(float(ap))
            cell["folds"].append(fi)
            cell.setdefault("scores", {})[str(fi)] = {
                "y": [float(v) for v in y],
                "s": [round(float(v), 5) for v in s],
                "groups": list(groups)}
            print(f"  fold {fi:>2} {key:<18} AUC={auc:.4f} AUPRC={ap:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            json.dump(results, open(out_path, "w"))

        results.setdefault("_trivial", {})[str(fi)] = trivial_all_positive(
            [float(d.y[0]) for d in test_data])
        json.dump(results, open(out_path, "w"))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
