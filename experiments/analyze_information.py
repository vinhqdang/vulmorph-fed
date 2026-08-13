"""
Why does the operation taxonomy add nothing over the grammar kind?

Hypothesis: phi is almost a deterministic coarsening of the grammar-kind
alphabet. Most of its rules are decided by node kind alone (subscript
expression, if statement, assignment, ...); only call classification consults
identifier text. If so, phi carries almost no information that the kind
embedding does not already have, which would explain a null effect exactly.

We test this directly by measuring, over real corpora:

  H(phi | kind)   conditional entropy of the operation label given the kind
  I(phi ; kind)   mutual information
  determinism     fraction of nodes whose phi label is the majority label for
                  their kind, i.e. predictable from kind alone

and we separate call nodes, where phi consults names, from the rest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import collections
import glob
import json
import math
import os

import torch

from data.morphology import get_taxonomy


def entropy(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def analyse(cache_path, taxonomy_size=8):
    data = torch.load(cache_path, weights_only=False)
    types, m = get_taxonomy(taxonomy_size)
    inv = {v: k for k, v in m.items()}

    joint = collections.Counter()
    phi_c = collections.Counter()
    kind_c = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)

    for d in data:
        if not hasattr(d, "x_kind"):
            continue
        for p, k in zip(d.x_morph.tolist(), d.x_kind.tolist()):
            joint[(p, k)] += 1
            phi_c[p] += 1
            kind_c[k] += 1
            by_kind[k][p] += 1

    n = sum(joint.values())
    if not n:
        return None

    h_phi = entropy(phi_c)
    h_kind = entropy(kind_c)
    # H(phi | kind) = sum_k p(k) H(phi | k)
    h_cond = sum((sum(c.values()) / n) * entropy(c) for c in by_kind.values())
    mi = h_phi - h_cond

    # Determinism: predict phi from kind by the majority rule.
    correct = sum(max(c.values()) for c in by_kind.values())
    determinism = correct / n

    # Where does the residual uncertainty live? Almost all of phi's
    # name-dependent behaviour is on call expressions.
    residual = sorted(
        ((sum(c.values()) / n) * entropy(c), k, dict(c))
        for k, c in by_kind.items() if entropy(c) > 0)
    residual.sort(reverse=True)

    return {
        "cache": os.path.basename(cache_path),
        "nodes": n,
        "H_phi": h_phi, "H_kind": h_kind,
        "H_phi_given_kind": h_cond,
        "MI_phi_kind": mi,
        "frac_phi_info_in_kind": (mi / h_phi) if h_phi else float("nan"),
        "determinism": determinism,
        "top_residual_kinds": [
            {"kind_id": k, "weighted_entropy": w,
             "labels": {inv.get(p, p): c for p, c in lab.items()}}
            for w, k, lab in residual[:5]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caches", nargs="*")
    p.add_argument("--output", default="experiments/results/information_analysis.json")
    args = p.parse_args()

    caches = args.caches or sorted(
        glob.glob(".cache/datasets_v3/*_t8.pt"), key=os.path.getsize, reverse=True)[:3]
    out = {}
    for c in caches:
        r = analyse(c)
        if not r:
            continue
        out[r["cache"]] = r
        print(f"\n=== {r['cache']}  ({r['nodes']} nodes) ===")
        print(f"  H(phi)            = {r['H_phi']:.3f} bits")
        print(f"  H(kind)           = {r['H_kind']:.3f} bits")
        print(f"  H(phi | kind)     = {r['H_phi_given_kind']:.4f} bits")
        print(f"  I(phi ; kind)     = {r['MI_phi_kind']:.3f} bits "
              f"({100*r['frac_phi_info_in_kind']:.1f}% of phi's information)")
        print(f"  phi predictable from kind alone: {100*r['determinism']:.2f}% of nodes")
        if r["top_residual_kinds"]:
            print("  residual uncertainty concentrates on:")
            for e in r["top_residual_kinds"]:
                labs = ", ".join(f"{k}:{v}" for k, v in
                                 sorted(e["labels"].items(), key=lambda x: -x[1])[:4])
                print(f"     kind {e['kind_id']:<4} H_w={e['weighted_entropy']:.4f}  {labs}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
