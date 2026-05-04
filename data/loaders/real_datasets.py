"""
Real Dataset Loaders for VulMorph-Fed (plan.md §4).

Implements loaders for:
  - Devign   (HuggingFace: DetectVul/devign)
  - PrimeVul (HuggingFace: starsofchance/PrimeVul)
  - BigVul   (GitHub CSV: ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)
  - DiverseVul (GitHub JSON: wagner-group/diversevul)

Since full CPG extraction requires Joern (heavy external tool), we use a
**token-level proxy graph** representation:
  - Each token in the function becomes a node with a morphological label
    (derived from simple heuristics on C/C++ token content).
  - Sequential edges simulate control flow between adjacent tokens.
  - Dependency edges connect tokens sharing the same identifier.

This gives a real vulnerability signal while keeping the framework
self-contained. The loader interface is compatible with VulMorphClient.
"""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch_geometric.data import Data, Dataset

from data.morphology import MORPHOLOGY_MAP, NUM_MORPHOLOGY_TYPES

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".cache/datasets")

# Simple token → morphology heuristic for C/C++
TOKEN_MORPH_RULES = [
    (r"\b(malloc|free|calloc|realloc|memcpy|memmove|memset|strcpy|strncpy|sprintf)\b",
     "MEMORY_ACCESS"),
    (r"\b\w+\s*\[",           "ARRAY_INDEX"),
    (r"\*\w+|\w+->",          "PTR_DEREF"),
    (r"\b(if|else|switch|case|for|while|do|goto|break|continue)\b",
     "CONTROL_BRANCH"),
    (r"[+\-\*/%&|^~]+",       "ARITH_OP"),
    (r"(==|!=|<=|>=|<|>)",    "COMPARISON"),
    (r"\b\w+\s*\(",           "CALL_SITE"),
    (r"\b\w+\s*=(?!=)",       "ASSIGN"),
]

def _token_to_morph(token: str) -> int:
    for pattern, mtype in TOKEN_MORPH_RULES:
        if re.search(pattern, token):
            return MORPHOLOGY_MAP[mtype]
    return MORPHOLOGY_MAP["UNKNOWN"]


def _build_vocab(all_tokens: List[List[str]], max_vocab: int = 10000) -> Dict[str, int]:
    """Build vocabulary from list-of-token-lists."""
    from collections import Counter
    counter = Counter(t for toks in all_tokens for t in toks)
    vocab = {"<UNK>": 0, "<PAD>": 1}
    for tok, _ in counter.most_common(max_vocab - 2):
        vocab[tok] = len(vocab)
    return vocab


def _tokenize(code: str, max_tokens: int = 100) -> List[str]:
    """Very lightweight C/C++ tokenizer (no parsing needed)."""
    # Split on whitespace and common delimiters
    tokens = re.findall(r"[\w]+|[^\s\w]", code)
    return tokens[:max_tokens]


def _code_to_graph(code: str, vocab: Dict[str, int], max_tokens: int = 100) -> Optional[Data]:
    """Convert a C/C++ function string to a token-level proxy graph."""
    tokens = _tokenize(code, max_tokens)
    if len(tokens) < 3:
        return None

    n = len(tokens)
    x_lex = torch.tensor([vocab.get(t, 0) for t in tokens], dtype=torch.long)
    x_morph = torch.tensor([_token_to_morph(t) for t in tokens], dtype=torch.long)

    # Sequential edges (simulating control flow)
    src = list(range(n - 1))
    dst = list(range(1, n))

    # Dependency edges: connect same-identifier tokens
    tok_positions: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        if re.match(r"^[a-zA-Z_]\w*$", t) and t not in {
            "if", "else", "for", "while", "return", "int", "char", "void",
            "unsigned", "signed", "long", "short", "struct", "typedef"
        }:
            tok_positions.setdefault(t, []).append(i)

    for positions in tok_positions.values():
        for i in range(len(positions) - 1):
            src.append(positions[i])
            dst.append(positions[i + 1])

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x_lex=x_lex, x_morph=x_morph, edge_index=edge_index, num_nodes=n)


# ── Devign Loader ────────────────────────────────────────────────────────────

def load_devign(max_samples: int = 5000, cache: bool = True) -> List[Data]:
    """
    Load Devign dataset from HuggingFace (DetectVul/devign).
    Returns a list of PyG Data objects.
    Reference: plan.md §4.1, Zhou et al. NeurIPS 2019.
    """
    cache_path = CACHE_DIR / f"devign_{max_samples}.pt"
    if cache and cache_path.exists():
        print(f"Loading Devign from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    try:
        from datasets import load_dataset
        print("Downloading Devign from HuggingFace (DetectVul/devign)...")
        hf_ds = load_dataset("DetectVul/devign", split="train")
    except Exception as e:
        print(f"Could not load Devign from HuggingFace: {e}")
        return []

    # Build vocab from all functions
    all_tokens = [_tokenize(row["func"], 100) for row in hf_ds]
    vocab = _build_vocab(all_tokens)

    data_list = []
    cwe_id = 0  # Devign doesn't have per-sample CWE; use 0 as placeholder

    for i, row in enumerate(hf_ds):
        if len(data_list) >= max_samples:
            break
        label = int(row["target"])
        graph = _code_to_graph(row["func"], vocab)
        if graph is None:
            continue
        graph.y = torch.tensor([float(label)])
        graph.cwe = torch.tensor([cwe_id if label == 1 else -1], dtype=torch.long)
        graph.project = row.get("project", "devign")
        data_list.append(graph)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)

    print(f"Devign: loaded {len(data_list)} samples, "
          f"vuln={sum(1 for d in data_list if d.y[0]==1)}")
    return data_list


# ── PrimeVul Loader ──────────────────────────────────────────────────────────

def load_primevul(split: str = "train", max_samples: int = 5000, cache: bool = True) -> List[Data]:
    """
    Load PrimeVul from HuggingFace (starsofchance/PrimeVul).
    Supports temporal train/test split.
    Reference: plan.md §4.1, Ding et al. ICSE 2025.
    """
    cache_path = CACHE_DIR / f"primevul_{split}_{max_samples}.pt"
    if cache and cache_path.exists():
        print(f"Loading PrimeVul ({split}) from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    try:
        from datasets import load_dataset
        print(f"Downloading PrimeVul ({split}) from HuggingFace...")
        hf_ds = load_dataset("starsofchance/PrimeVul", split=split)
    except Exception as e:
        print(f"Could not load PrimeVul from HuggingFace: {e}")
        return []

    all_tokens = [_tokenize(row.get("func", row.get("code", "")), 100) for row in hf_ds]
    vocab = _build_vocab(all_tokens)

    data_list = []
    for row in hf_ds:
        if len(data_list) >= max_samples:
            break
        code = row.get("func", row.get("code", ""))
        label = int(row.get("target", row.get("label", 0)))
        cwe_raw = row.get("cwe", "-1")
        # Parse CWE to int: "CWE-119" -> 119; use modulo for embedding index
        try:
            cwe_id = int(re.search(r"\d+", str(cwe_raw)).group()) % 150 if label == 1 else -1
        except Exception:
            cwe_id = 0 if label == 1 else -1

        graph = _code_to_graph(code, vocab)
        if graph is None:
            continue
        graph.y = torch.tensor([float(label)])
        graph.cwe = torch.tensor([cwe_id], dtype=torch.long)
        graph.project = row.get("project", "primevul")
        data_list.append(graph)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)

    print(f"PrimeVul ({split}): loaded {len(data_list)} samples, "
          f"vuln={sum(1 for d in data_list if d.y[0]==1)}")
    return data_list


# ── BigVul Loader ────────────────────────────────────────────────────────────

def load_bigvul(csv_path: str, max_samples: int = 10000, cache: bool = True) -> List[Data]:
    """
    Load BigVul from a local CSV file.
    Download from: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
    Expected columns: func_before, vul, CWE ID, project (optional).
    Reference: plan.md §4.1, Fan et al. MSR 2020.
    """
    if not os.path.exists(csv_path):
        print(f"BigVul CSV not found at {csv_path}. Skipping.")
        return []

    cache_key = hashlib.md5(csv_path.encode()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"bigvul_{cache_key}_{max_samples}.pt"
    if cache and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    import csv
    rows = []
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_samples:
                break

    all_tokens = [_tokenize(r.get("func_before", ""), 100) for r in rows]
    vocab = _build_vocab(all_tokens)

    data_list = []
    for row in rows:
        code = row.get("func_before", "")
        label = int(row.get("vul", 0))
        cwe_raw = row.get("CWE ID", "-1")
        try:
            cwe_id = int(re.search(r"\d+", str(cwe_raw)).group()) % 150 if label == 1 else -1
        except Exception:
            cwe_id = 0 if label == 1 else -1

        graph = _code_to_graph(code, vocab)
        if graph is None:
            continue
        graph.y = torch.tensor([float(label)])
        graph.cwe = torch.tensor([cwe_id], dtype=torch.long)
        graph.project = row.get("project", "bigvul")
        data_list.append(graph)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)

    print(f"BigVul: loaded {len(data_list)} samples, "
          f"vuln={sum(1 for d in data_list if d.y[0]==1)}")
    return data_list


# ── DiverseVul Loader ─────────────────────────────────────────────────────────

def load_diversevul(json_path: str, max_samples: int = 10000, cache: bool = True) -> List[Data]:
    """
    Load DiverseVul from a local JSONL file.
    Download from: https://github.com/wagner-group/diversevul
    Expected fields: func, target, cwe (optional), project (optional).
    Reference: plan.md §4.1, Chen et al. RAID 2023.
    """
    if not os.path.exists(json_path):
        print(f"DiverseVul JSON not found at {json_path}. Skipping.")
        return []

    cache_key = hashlib.md5(json_path.encode()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"diversevul_{cache_key}_{max_samples}.pt"
    if cache and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    rows = []
    with open(json_path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break

    all_tokens = [_tokenize(r.get("func", ""), 100) for r in rows]
    vocab = _build_vocab(all_tokens)

    data_list = []
    for row in rows:
        code = row.get("func", "")
        label = int(row.get("target", 0))
        cwe_raw = row.get("cwe", "-1")
        try:
            cwe_id = int(re.search(r"\d+", str(cwe_raw)).group()) % 150 if label == 1 else -1
        except Exception:
            cwe_id = 0 if label == 1 else -1

        graph = _code_to_graph(code, vocab)
        if graph is None:
            continue
        graph.y = torch.tensor([float(label)])
        graph.cwe = torch.tensor([cwe_id], dtype=torch.long)
        graph.project = row.get("project", row.get("repo", "diversevul"))
        data_list.append(graph)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)

    print(f"DiverseVul: loaded {len(data_list)} samples, "
          f"vuln={sum(1 for d in data_list if d.y[0]==1)}")
    return data_list


# ── Cross-Project Federated Split ─────────────────────────────────────────────

def split_by_project(
    data_list: List[Data],
    num_clients: int,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[List[Data]], List[Data]]:
    """
    Partition a dataset by project for cross-project federated evaluation.

    - Groups samples by `data.project`.
    - Holds out `test_fraction` of projects as the cross-project test set.
    - Distributes the remaining projects across K clients.
    - Falls back to random sample split when there are too few distinct projects
      (e.g. Devign which has a single 'devign' project field).

    Returns:
        client_datasets: List of K lists of Data objects (train split).
        test_dataset:    List of Data objects from held-out projects.
    """
    random.seed(seed)

    # Group by project
    by_project: Dict[str, List[Data]] = {}
    for d in data_list:
        proj = getattr(d, 'project', 'unknown')
        by_project.setdefault(proj, []).append(d)

    projects = list(by_project.keys())
    random.shuffle(projects)

    # Need at least num_clients + 1 projects for a meaningful split.
    # When too few projects exist, fall back to random sample-level split.
    if len(projects) < num_clients + 1:
        print(f"Only {len(projects)} distinct project(s) found; "
              f"falling back to random sample-level split.")
        random.shuffle(data_list)
        n_test = max(1, int(len(data_list) * test_fraction))
        test_raw = data_list[:n_test]
        train_raw = data_list[n_test:]
        chunk = max(1, len(train_raw) // num_clients)
        client_buckets = [
            train_raw[i * chunk: (i + 1) * chunk]
            for i in range(num_clients)
        ]
        # Put any remainder in the last client
        client_buckets[-1].extend(train_raw[num_clients * chunk:])
        print(f"Random split: {len(train_raw)} train, {len(test_raw)} test samples.")
        return client_buckets, test_raw

    n_test_projects = max(1, int(len(projects) * test_fraction))
    test_projects = set(projects[:n_test_projects])
    train_projects = projects[n_test_projects:]

    test_dataset = [d for p in test_projects for d in by_project[p]]

    client_buckets: List[List[Data]] = [[] for _ in range(num_clients)]
    for i, proj in enumerate(train_projects):
        client_buckets[i % num_clients].extend(by_project[proj])

    # Guard against any empty buckets
    non_empty = [b for b in client_buckets if b]
    if len(non_empty) < num_clients:
        print(f"Warning: only {len(non_empty)}/{num_clients} clients have data. "
              f"Reducing num_clients to {len(non_empty)}.")
        client_buckets = non_empty

    print(f"Cross-project split: {len(train_projects)} train projects across "
          f"{len(client_buckets)} clients, {len(test_projects)} test projects "
          f"({len(test_dataset)} samples).")

    return client_buckets, test_dataset


class ListDataset(Dataset):
    """Thin wrapper to expose a plain List[Data] as a PyG Dataset."""
    def __init__(self, data_list: List[Data]):
        super().__init__(root=None)
        self._data = data_list

    def len(self): return len(self._data)
    def get(self, idx): return self._data[idx]
