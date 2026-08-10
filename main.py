import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fl.client import VulMorphClient
from fl.server import VulMorphServer
from utils.metrics import compute_metrics, best_f1_threshold
from data.loaders.real_datasets import (
    load_devign, load_primevul, load_bigvul, load_diversevul,
    load_bigvul_hf, load_diversevul_hf, load_primevul_hf,
    split_by_project, ListDataset, abstraction_stats, bucket_cwes,
    carve_calibration,
)
from data.morphology import get_taxonomy
from data.dataset import get_client_datasets
from utils.privacy import composed_epsilon


# ── Evaluation ───────────────────────────────────────────────────────────────

def _client_probs(client, loader, global_prototypes):
    """Predicted probabilities of one client model over a fixed loader."""
    client.model.eval()
    proto = (global_prototypes.to(client.device)
             if global_prototypes is not None else None)
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(client.device)
            logits, _, _ = client.model(batch, prototypes=proto)
            probs = torch.sigmoid(logits.squeeze(-1))
            y_true.extend(batch.y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_prob)


def ensemble_probs(clients, global_prototypes, dataset):
    """Uniform-ensemble probabilities of the K client models over a dataset."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader(dataset, batch_size=64, shuffle=False)
    prob_sum, y_true, per_client = None, None, []
    for client in clients:
        yt, yp = _client_probs(client, loader, global_prototypes)
        y_true = yt
        prob_sum = yp if prob_sum is None else prob_sum + yp
        per_client.append(yp)
    if y_true is None:
        return None, None, []
    return y_true, prob_sum / len(clients), per_client


def evaluate(clients, global_prototypes, test_dataset=None, cal_dataset=None):
    """
    Inference protocol (reported in the manuscript, Sec. Experimental Setup):

    Every client ends each round with its own encoder plus the shared global
    prototype bank. The *deployed* global detector is the uniform probability
    ensemble of the K client models, each conditioned on the same global
    prototype bank:  p(y|G) = (1/K) * sum_k sigmoid(f_k(G; P*)).

    The decision threshold is calibrated on `cal_dataset` (a held-out slice
    of TRAINING-project samples at true prevalence) when provided; otherwise
    the default 0.5 is used. The test set never influences the threshold.
    """
    if test_dataset is not None:
        y_true, probs, per_client = ensemble_probs(
            clients, global_prototypes, test_dataset)
        if y_true is None or len(y_true) == 0:
            return {"f1": 0.0, "auc": 0.5, "precision": 0.0, "recall": 0.0}

        thr = 0.5
        if cal_dataset is not None and len(cal_dataset) > 0:
            yc, pc, _ = ensemble_probs(clients, global_prototypes, cal_dataset)
            thr = best_f1_threshold(yc, pc)

        metrics = compute_metrics(y_true, probs, threshold=thr)
        metrics["threshold"] = thr
        client_f1s = [
            compute_metrics(y_true, p, threshold=thr)["f1"] for p in per_client
        ]
        metrics["client_f1_mean"] = float(np.mean(client_f1s))
        metrics["client_f1_std"] = float(np.std(client_f1s))
        return metrics

    # No held-out set: evaluate each client on its local training data
    all_y_true, all_y_pred = [], []
    for client in clients:
        yt, yp = _client_probs(client, client.train_loader, global_prototypes)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)

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
    tax = getattr(args, "taxonomy_size", 8)

    if args.dataset == "devign":
        data_list = load_devign(max_samples=args.max_samples, taxonomy_size=tax)
    elif args.dataset == "primevul":
        data_list = load_primevul_hf(max_samples=args.max_samples,
                                     taxonomy_size=tax)
    elif args.dataset == "bigvul":
        if args.data_path:
            data_list = load_bigvul(args.data_path, max_samples=args.max_samples,
                                    taxonomy_size=tax)
        else:
            data_list = load_bigvul_hf(max_samples=args.max_samples,
                                       taxonomy_size=tax)
    elif args.dataset == "diversevul":
        if args.data_path:
            data_list = load_diversevul(args.data_path,
                                        max_samples=args.max_samples,
                                        taxonomy_size=tax)
        else:
            data_list = load_diversevul_hf(max_samples=args.max_samples,
                                           taxonomy_size=tax)

    if not data_list:
        print(f"Warning: could not load dataset '{args.dataset}'. "
              "Falling back to structured synthetic data.")
        return None, None, None

    # Map raw CWE ids to the fixed prototype vocabulary:
    # top (num_cwes - 1) most frequent CWEs + shared OTHER bucket.
    bucket_cwes(data_list, args.num_cwes)

    stats = abstraction_stats(data_list)
    if stats:
        print(f"Abstraction stats (|T|={tax}): "
              f"typed_node_ratio={stats['typed_node_ratio']:.3f}, "
              f"avg_nodes={stats['avg_nodes']:.1f}, "
              f"avg_edges={stats['avg_edges']:.1f}")

    # Cross-project split: held-out test projects never seen during training
    client_buckets, test_raw = split_by_project(
        data_list,
        num_clients=args.num_clients,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    # Calibration slice (true prevalence, training projects only), then
    # benign downsampling of the remaining training samples (4:1 cap).
    client_buckets, cal_raw = carve_calibration(client_buckets, seed=args.seed)

    client_datasets = [ListDataset(bucket) for bucket in client_buckets]
    test_dataset = ListDataset(test_raw)
    cal_dataset = ListDataset(cal_raw)

    return client_datasets, cal_dataset, test_dataset


# ── Federated training loop ──────────────────────────────────────────────────

def run_fl(args, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    use_dp           = model_kwargs.pop('use_dp', True)
    use_cwe_affinity = model_kwargs.pop('use_cwe_affinity', True)
    federate         = model_kwargs.pop('federate', True)

    # ── Dataset ─────────────────────────────────────────────────────────
    client_datasets = cal_dataset = test_dataset = None

    if getattr(args, 'dataset', 'synthetic') != 'synthetic':
        client_datasets, cal_dataset, test_dataset = load_real_data(args)

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

    # Morphology embedding table must match the taxonomy size (|T| + 1
    # categories including UNKNOWN).
    tax = getattr(args, 'taxonomy_size', 8)
    model_kwargs.setdefault('num_morph_types', len(get_taxonomy(tax)[0]))

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
        num_cwes=args.num_cwes + 1,   # +1 benign slot
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

        # Per-round progress at the uncalibrated threshold (cheap)
        metrics = evaluate(clients, server.global_prototypes, test_dataset)
        history.append(metrics)

        split_name = "cross-project test" if test_dataset else "train (synthetic)"
        print(
            f"  Round {r+1:>2} [{split_name}] | "
            f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
        )

    # Final evaluation with the calibrated decision threshold (threshold
    # chosen on training-project calibration samples, never the test set).
    if test_dataset is not None:
        final = evaluate(clients, server.global_prototypes,
                         test_dataset, cal_dataset)
        print(f"  FINAL (calibrated thr={final.get('threshold', 0.5):.3f}) | "
              f"F1={final['f1']:.4f} AUC={final['auc']:.4f} "
              f"P={final['precision']:.4f} R={final['recall']:.4f}")
    else:
        final = history[-1] if history else {}
    if final and use_dp and federate:
        # End-to-end privacy accounting under sequential composition:
        # each round consumes ε_round, so T rounds consume T · ε_round.
        final["epsilon_per_round"] = args.epsilon
        final["epsilon_total"] = composed_epsilon(args.epsilon, args.rounds)
    return final


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
    p.add_argument("--taxonomy_size", type=int,  default=8, choices=[8, 16, 32],
                   help="Morphological taxonomy size |T| (RQ2b sensitivity)")
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--dropout",      type=float, default=0.3)

    # Loss weights
    p.add_argument("--alpha",        type=float, default=0.1)
    p.add_argument("--gamma",        type=float, default=0.01)

    # Privacy
    p.add_argument("--epsilon",      type=float, default=2.0)
    p.add_argument("--delta_f",      type=float, default=1.0,
                   help="L1 clipping radius R for per-sample embeddings (DP)")

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
