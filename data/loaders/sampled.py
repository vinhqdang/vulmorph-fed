"""
Seeded simple-random samples of BigVul and DiverseVul, as program graphs.

The earlier loaders took the first N rows in corpus order, which is neither
random nor representative (corpus order follows project/commit order). Here a
sample is drawn uniformly at random, without replacement, from the whole
`train` split of the public mirror, with a fixed seed recorded in the cache
name. Graphs carry every feature mode used by the experiments (lexical, alpha-
renamed, grammar kind, operation type), the edge type, and truncation
bookkeeping (`n_full`, `truncated`).
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from data.loaders.ast_graphs import code_to_ast_graph

CACHE_DIR = Path(".cache/sampled_v1")

CORPORA = {
    # name: (hf id, code column, label fn, project column)
    "bigvul": ("bstee615/bigvul", "func_before",
               lambda r: int(r.get("vul", 0) or 0), "project"),
    "devign": ("DetectVul/devign", "func",
               lambda r: int(bool(r["target"])) if not isinstance(r["target"], str)
               else int(r["target"].strip().lower() in ("1", "true")),
               "project"),
    "diversevul": ("bstee615/diversevul", "func",
                   lambda r: int(r.get("target", 0) or 0), "project"),
}


def load_sampled(corpus: str, n: int, seed: int = 0, taxonomy_size: int = 8,
                 max_nodes: int = 200, cache: bool = True
                 ) -> Tuple[List, Dict[str, int], Dict[str, int]]:
    """Return (graphs, lexical token dict, anonymised token dict)."""
    tag = f"{corpus}_n{n}_s{seed}_t{taxonomy_size}_c{max_nodes}"
    path = CACHE_DIR / f"{tag}.pt"
    if cache and path.exists():
        return torch.load(path, weights_only=False)

    from datasets import load_dataset
    hf, code_key, label_fn, proj_key = CORPORA[corpus]
    ds = load_dataset(hf, split="train")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    vocab, anon = {"<UNK>": 0}, {"<UNK>": 0}
    graphs, tried = [], 0
    for i in order:
        if len(graphs) >= n:
            break
        row = ds[i]
        tried += 1
        code = row.get(code_key) or ""
        g = code_to_ast_graph(code, vocab, max_nodes=max_nodes,
                              taxonomy_size=taxonomy_size, anon_vocab=anon,
                              grow_vocab=True)
        if g is None:
            continue
        g.y = torch.tensor([float(label_fn(row))])
        g.project = row.get(proj_key) or corpus
        g.src_index = int(i)
        graphs.append(g)
    print(f"{corpus}: sampled {len(graphs)} parseable functions "
          f"({tried} drawn), vuln={sum(int(g.y[0]) for g in graphs)}")
    out = (graphs, vocab, anon)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return out
