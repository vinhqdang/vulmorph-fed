"""
Shared helpers for all VulMorph-Fed experiment runners.

Every experiment is repeated over multiple random seeds; results are stored
as {metric: {"mean": .., "std": .., "values": [..]}} so tables can report
mean ± std and significance tests can consume the per-seed values.
"""

import argparse
import copy
from typing import Callable, Dict, List

import numpy as np
import torch


DEFAULT_SEEDS = [42, 43, 44]


def make_args(seed=42, num_clients=4, rounds=10, local_epochs=2,
              epsilon=2.0, total_graphs=5000, num_cwes=10,
              hidden_dim=128, embed_dim=64, dataset="devign",
              max_samples=8000, test_fraction=0.2, taxonomy_size=8,
              **kwargs) -> argparse.Namespace:
    """Build a full args namespace for main.run_fl."""
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
        taxonomy_size=taxonomy_size,
        num_layers=2,
        batch_size=64,
        lr=1e-3,
        dropout=0.3,
        alpha=0.1,
        gamma=0.01,
        epsilon=epsilon,
        delta_f=1.0,
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


def run_seeded(run_fn: Callable[[int], Dict[str, float]],
               seeds: List[int]) -> Dict[str, Dict]:
    """
    Run `run_fn(seed)` for each seed and aggregate scalar metrics as
    mean/std/values. Non-scalar entries from the last run are kept as-is.
    """
    per_seed = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        m = run_fn(s)
        per_seed.append(m)
        print(f"    seed={s} → " + " ".join(
            f"{k}={v:.4f}" for k, v in m.items()
            if isinstance(v, (int, float)) and k in
            ("f1", "auc", "precision", "recall")))

    agg: Dict[str, Dict] = {}
    keys = set().union(*[m.keys() for m in per_seed])
    for k in keys:
        vals = [m[k] for m in per_seed if isinstance(m.get(k), (int, float))]
        if len(vals) == len(per_seed) and vals:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
    agg["num_seeds"] = len(seeds)
    agg["seeds"] = list(seeds)
    return agg


def add_common_cli(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--dataset",       type=str,   default="devign")
    p.add_argument("--max_samples",   type=int,   default=8000)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--num_clients",   type=int,   default=4)
    p.add_argument("--rounds",        type=int,   default=10)
    p.add_argument("--local_epochs",  type=int,   default=2)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--num_cwes",      type=int,   default=10)
    p.add_argument("--taxonomy_size", type=int,   default=8, choices=[8, 16, 32])
    p.add_argument("--seeds",         type=str,   default="42,43,44",
                   help="Comma-separated random seeds")
    return p


def parse_seeds(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def load_graphs(dataset: str, max_samples: int, taxonomy_size: int = 8,
                data_path: str = None):
    """Load a dataset by name (mirrors main.load_real_data, without CWE bucketing)."""
    from data.loaders.real_datasets import (
        load_devign, load_primevul, load_bigvul, load_diversevul,
        load_bigvul_hf, load_diversevul_hf, load_primevul_hf,
    )
    if dataset == "devign":
        return load_devign(max_samples=max_samples, taxonomy_size=taxonomy_size)
    if dataset == "primevul":
        return load_primevul_hf(max_samples=max_samples,
                                taxonomy_size=taxonomy_size)
    if dataset == "bigvul":
        if data_path:
            return load_bigvul(data_path, max_samples=max_samples,
                               taxonomy_size=taxonomy_size)
        return load_bigvul_hf(max_samples=max_samples, taxonomy_size=taxonomy_size)
    if dataset == "diversevul":
        if data_path:
            return load_diversevul(data_path, max_samples=max_samples,
                                   taxonomy_size=taxonomy_size)
        return load_diversevul_hf(max_samples=max_samples,
                                  taxonomy_size=taxonomy_size)
    raise ValueError(f"Unknown dataset {dataset}")
