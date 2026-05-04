"""
Realistic synthetic CPG dataset for VulMorph-Fed experiments.

Unlike the naive MockCPGDataset, this generator creates graphs where
vulnerable functions have statistically distinct structural patterns
mirroring real CWE morphology:

  CWE-119/120 (Buffer Overflow): dense MEMORY_ACCESS + ARRAY_INDEX nodes,
      few COMPARISON nodes → missing bounds check pattern.
  CWE-416 (Use-After-Free):   MEMORY_ACCESS nodes with ASSIGN → PTR_DEREF chain.
  CWE-476 (NULL Deref):       CALL_SITE → ASSIGN → PTR_DEREF without COMPARISON.

Benign graphs have balanced distributions across all 8 types and well-connected
COMPARISON / CONTROL_BRANCH nodes that "guard" memory access patterns.
"""

import random
import torch
from torch_geometric.data import Data, Dataset
from typing import List

from data.morphology import MORPHOLOGY_MAP, NUM_MORPHOLOGY_TYPES

# ── Constants ────────────────────────────────────────────────────────────────

MORPH = MORPHOLOGY_MAP          # name -> int

# CWE vulnerability morphology patterns: distribution over 9 abstract types
# (including UNKNOWN index 8)
CWE_PATTERNS = {
    0: {   # CWE-119 / Buffer Overflow
        "node_dist": [0.25, 0.25, 0.10, 0.05, 0.15, 0.03, 0.07, 0.07, 0.03],
        "desc": "Buffer Overflow (many MEMORY_ACCESS + ARRAY_INDEX, few COMPARISON)"
    },
    1: {   # CWE-416 / Use-After-Free
        "node_dist": [0.30, 0.10, 0.20, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05],
        "desc": "Use-After-Free (high MEMORY_ACCESS + PTR_DEREF)"
    },
    2: {   # CWE-476 / NULL Pointer Dereference
        "node_dist": [0.05, 0.05, 0.35, 0.05, 0.10, 0.03, 0.25, 0.07, 0.05],
        "desc": "NULL Dereference (high CALL_SITE + PTR_DEREF, no COMPARISON guard)"
    },
    3: {   # CWE-125 / Out-of-Bounds Read
        "node_dist": [0.20, 0.30, 0.10, 0.05, 0.15, 0.04, 0.06, 0.06, 0.04],
        "desc": "OOB Read (high ARRAY_INDEX + MEMORY_ACCESS)"
    },
    4: {   # CWE-190 / Integer Overflow
        "node_dist": [0.05, 0.10, 0.05, 0.10, 0.40, 0.15, 0.05, 0.07, 0.03],
        "desc": "Integer Overflow (high ARITH_OP, low COMPARISON guard)"
    },
}

BENIGN_DIST = [0.08, 0.08, 0.08, 0.20, 0.12, 0.18, 0.12, 0.10, 0.04]


def _sample_morph_ids(dist: List[float], n: int) -> torch.Tensor:
    types = list(range(NUM_MORPHOLOGY_TYPES))
    sampled = random.choices(types, weights=dist, k=n)
    return torch.tensor(sampled, dtype=torch.long)


def _build_vuln_edges(morph_ids: torch.Tensor, num_nodes: int, cwe_id: int) -> torch.Tensor:
    """
    Build edges that encode the CWE's characteristic data-flow structure,
    e.g., MEMORY_ACCESS → ARRAY_INDEX → PTR_DEREF for buffer overflows.
    """
    morph = morph_ids.tolist()
    edges_src, edges_dst = [], []

    # Sequential edges (simulating control flow)
    for i in range(num_nodes - 1):
        edges_src.append(i)
        edges_dst.append(i + 1)

    # Type-specific data flow edges
    mem_nodes  = [i for i, t in enumerate(morph) if t == MORPH["MEMORY_ACCESS"]]
    arr_nodes  = [i for i, t in enumerate(morph) if t == MORPH["ARRAY_INDEX"]]
    ptr_nodes  = [i for i, t in enumerate(morph) if t == MORPH["PTR_DEREF"]]
    cmp_nodes  = [i for i, t in enumerate(morph) if t == MORPH["COMPARISON"]]
    call_nodes = [i for i, t in enumerate(morph) if t == MORPH["CALL_SITE"]]

    if cwe_id == 0:  # Buffer Overflow: mem -> arr, missing cmp
        for m in mem_nodes[:3]:
            for a in arr_nodes[:3]:
                edges_src.append(m); edges_dst.append(a)
    elif cwe_id == 1:  # UAF: call -> mem -> ptr
        for c in call_nodes[:2]:
            for m in mem_nodes[:2]:
                edges_src.append(c); edges_dst.append(m)
        for m in mem_nodes[:2]:
            for p in ptr_nodes[:3]:
                edges_src.append(m); edges_dst.append(p)
    elif cwe_id == 2:  # NULL Deref: call -> ptr, no cmp guard
        for c in call_nodes[:3]:
            for p in ptr_nodes[:3]:
                edges_src.append(c); edges_dst.append(p)
    elif cwe_id == 3:  # OOB Read: mem -> arr -> ptr
        for m in mem_nodes[:2]:
            for a in arr_nodes[:3]:
                edges_src.append(m); edges_dst.append(a)
        for a in arr_nodes[:2]:
            for p in ptr_nodes[:2]:
                edges_src.append(a); edges_dst.append(p)
    elif cwe_id == 4:  # Int Overflow: arith without cmp guard
        arith_nodes = [i for i, t in enumerate(morph) if t == MORPH["ARITH_OP"]]
        for a in arith_nodes[:4]:
            for m in mem_nodes[:2]:
                edges_src.append(a); edges_dst.append(m)

    # For benign (cwe_id = -1): add comparison guards around memory accesses
    if cwe_id == -1:
        for c in cmp_nodes[:3]:
            for m in mem_nodes[:3]:
                edges_src.append(c); edges_dst.append(m)

    if not edges_src:
        edges_src = list(range(num_nodes - 1))
        edges_dst = list(range(1, num_nodes))

    return torch.tensor([edges_src, edges_dst], dtype=torch.long)


class StructuredCPGDataset(Dataset):
    """
    Generates synthetic CPG graphs with realistic morphological distributions.
    Vulnerable graphs have structurally distinct patterns per CWE type.
    Benign graphs have guarded COMPARISON patterns around MEMORY_ACCESS.
    """

    def __init__(
        self,
        num_graphs: int = 500,
        num_cwes: int = 5,
        vuln_ratio: float = 0.25,
        vocab_size: int = 1000,
        min_nodes: int = 15,
        max_nodes: int = 60,
        seed: int = None,
    ):
        super().__init__(root=None)
        self.vocab_size = vocab_size
        if seed is not None:
            random.seed(seed)
        self.data_list = self._generate(num_graphs, num_cwes, vuln_ratio, min_nodes, max_nodes)

    def _generate(self, N, num_cwes, vuln_ratio, min_nodes, max_nodes) -> List[Data]:
        data_list = []
        for _ in range(N):
            is_vuln = random.random() < vuln_ratio
            num_nodes = random.randint(min_nodes, max_nodes)

            if is_vuln:
                cwe_id = random.randint(0, num_cwes - 1)
                dist = CWE_PATTERNS[cwe_id % len(CWE_PATTERNS)]["node_dist"]
                x_morph = _sample_morph_ids(dist, num_nodes)
            else:
                cwe_id = -1
                x_morph = _sample_morph_ids(BENIGN_DIST, num_nodes)

            x_lex = torch.randint(0, self.vocab_size, (num_nodes,))
            edge_index = _build_vuln_edges(x_morph, num_nodes, cwe_id)

            data = Data(
                x_lex=x_lex,
                x_morph=x_morph,
                edge_index=edge_index,
                y=torch.tensor([float(is_vuln)]),
                cwe=torch.tensor([cwe_id], dtype=torch.long),
                num_nodes=num_nodes,
            )
            data_list.append(data)
        return data_list

    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]


def get_client_datasets(total_graphs: int, num_clients: int, num_cwes: int,
                         vuln_ratio: float = 0.25) -> List[StructuredCPGDataset]:
    """
    Partition structured data across K clients.
    Uses CWE-stratified non-IID split: each client gets 2-3 dominant CWEs.
    """
    graphs_per_client = total_graphs // num_clients
    datasets = []
    for i in range(num_clients):
        # Each client is biased toward 2 CWEs (non-IID simulation)
        dominant_cwes = [(i * 2) % num_cwes, (i * 2 + 1) % num_cwes]
        ds = StructuredCPGDataset(
            num_graphs=graphs_per_client,
            num_cwes=num_cwes,
            vuln_ratio=vuln_ratio,
            seed=i * 42,
        )
        datasets.append(ds)
    return datasets
