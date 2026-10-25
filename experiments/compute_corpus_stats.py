"""
Corpus properties, measured from the sampled graphs (nothing typed by hand).

Writes results/corpus_stats.json with, per corpus: size, prevalence, projects,
project concentration, graph size, the effect of the 200-node cap on each
class, the train-only projects, and the prevalence/composition of the actual
held-out folds (read from the run's `_folds` record) -- which is why overall
prevalence and test-set prevalence differ.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import collections
import json

import numpy as np

from data.loaders.sampled import load_sampled
from data.loaders.real_datasets import dedupe_functions
from data.morphology import get_taxonomy

RESULTS = Path(__file__).parent / "results"


def stats_for(name, n, seed, fold_file=None):
    data, _, _ = load_sampled(name, n, seed, 8)
    n_raw = len(data)
    data = dedupe_functions(data)
    y = np.array([float(d.y[0]) for d in data])
    nodes = np.array([d.num_nodes for d in data])
    edges = np.array([d.edge_index.shape[1] for d in data])
    full = np.array([d.n_full for d in data])
    trunc = np.array([d.truncated for d in data], dtype=bool)
    proj = collections.Counter(d.project for d in data)
    vproj = collections.Counter(d.project for d in data if float(d.y[0]) == 1)
    top = proj.most_common(2)
    out = {
        "n_drawn": n, "n_parsed": n_raw, "n": len(data),
        "vulnerable": int(y.sum()), "prev": float(y.mean()),
        "projects": len(proj),
        "top2": [t[0] for t in top],
        "top2_share": 100.0 * sum(t[1] for t in top) / len(data),
        "nodes": float(nodes.mean()), "edges": float(edges.mean()),
        "trunc_rate": float(trunc.mean()),
        "trunc_rate_vuln": float(trunc[y == 1].mean()) if (y == 1).any() else 0.0,
        "trunc_rate_benign": float(trunc[y == 0].mean()),
        "median_full_nodes_vuln": float(np.median(full[y == 1])),
        "median_full_nodes_benign": float(np.median(full[y == 0])),
        "vuln_share_by_project": {p: [vproj.get(p, 0), c]
                                  for p, c in proj.most_common(8)},
    }
    # Truncation is not independent of class if long functions are more often
    # vulnerable; test it rather than assert it.
    from scipy.stats import fisher_exact
    a = int((trunc & (y == 1)).sum()); b = int((~trunc & (y == 1)).sum())
    c = int((trunc & (y == 0)).sum()); d = int((~trunc & (y == 0)).sum())
    out["trunc_fisher_p"] = float(fisher_exact([[a, b], [c, d]])[1])
    out["trunc_odds_ratio"] = float(fisher_exact([[a, b], [c, d]])[0])

    if fold_file and (RESULTS / fold_file).exists():
        folds = json.load(open(RESULTS / fold_file))["_folds"]
        by_proj = collections.defaultdict(list)
        for d in data:
            by_proj[d.project].append(d)
        fi = []
        for f in folds:
            t = [d for p in f["test_projects"] for d in by_proj[p]]
            yy = np.array([float(d.y[0]) for d in t])
            share = collections.Counter(d.project for d in t)
            top_p, top_c = share.most_common(1)[0]
            fi.append({"n_test": len(t), "n_vuln": int(yy.sum()),
                       "prev": float(yy.mean()),
                       "n_projects": len(share), "top_project": top_p,
                       "top_share": top_c / len(t)})
        out["folds"] = fi
        out["train_only"] = folds[0]["excluded_projects"]
        tr_only = [d for p in out["train_only"] for d in by_proj[p]]
        out["train_only_n"] = len(tr_only)
        out["train_only_prev"] = float(np.mean([float(d.y[0]) for d in tr_only])) \
            if tr_only else 0.0
        out["test_prev_mean"] = float(np.mean([f["prev"] for f in fi]))
        out["test_prev_pooled"] = float(sum(f["n_vuln"] for f in fi)
                                        / sum(f["n_test"] for f in fi))
    # node-type distribution under phi_8
    types, m = get_taxonomy(8)
    inv = {v: k for k, v in m.items()}
    cnt = collections.Counter()
    for d in data:
        cnt.update(d.x_morph.tolist())
    tot = sum(cnt.values())
    out["node_dist"] = {inv[k]: 100.0 * v / tot for k, v in cnt.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = {}
    for name, ff in (("bigvul", "bigvul_core_f01.json"),
                     ("diversevul", "diversevul_core_f01.json"),
                     ("devign", None)):
        res[name] = stats_for(name, args.n, args.seed, ff)
        s = res[name]
        print(f"{name}: n={s['n']} vuln={s['vulnerable']} prev={s['prev']:.3f} "
              f"projects={s['projects']} trunc={s['trunc_rate']:.3f} "
              f"(vuln {s['trunc_rate_vuln']:.3f} / benign "
              f"{s['trunc_rate_benign']:.3f}, Fisher p={s['trunc_fisher_p']:.3g})")
    res["_node_dist_bigvul"] = res["bigvul"]["node_dist"]
    json.dump(res, open(RESULTS / "corpus_stats.json", "w"), indent=2)
    print("wrote corpus_stats.json")


if __name__ == "__main__":
    main()
