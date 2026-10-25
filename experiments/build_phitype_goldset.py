"""
E2, step 1 - build the evaluation set for phi_type as a static analysis.

Extracts every callee name in a corpus, ranks by frequency, and emits a
stratified sample for annotation together with phi_type's prediction. The
sample is stratified over *predicted* classes so that rare classes are
represented and precision can be estimated per class, and it includes a
random tail sample so recall is estimable rather than only precision.

The output is a TSV the authors annotate with the true class. Nothing here
depends on the annotation, so the sampling is reproducible and auditable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import collections
import csv
import random
import re

from data.loaders.api_classes import classify_api
from data.loaders.ast_graphs import _PARSER, TS_AVAILABLE
from experiments.common import load_graphs   # noqa: F401  (ensures deps load)

_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def collect_callees(dataset: str, max_samples: int, sample_n: int = 0,
                    sample_seed: int = 0):
    """Frequency table (and one example call each) over a corpus.

    For bigvul/diversevul with `sample_n` > 0 the functions are exactly those
    of the random sample used by the experiments, so the validation is run on
    the experimental corpora rather than on a different one.
    """
    from datasets import load_dataset
    src_key, name = {
        "devign": ("func", "DetectVul/devign"),
        "bigvul": ("func_before", "bstee615/bigvul"),
        "diversevul": ("func", "bstee615/diversevul"),
    }[dataset]

    counts, examples = collections.Counter(), {}
    if sample_n and dataset in ("bigvul", "diversevul"):
        from data.loaders.sampled import load_sampled
        graphs, _, _ = load_sampled(dataset, sample_n, sample_seed, 8)
        ds = load_dataset(name, split="train")
        rows = (ds[g.src_index] for g in graphs)
    else:
        rows = (r for i, r in zip(range(max_samples),
                                  load_dataset(name, split="train",
                                               streaming=True)))
    for row in rows:
        code = row.get(src_key) or row.get("func") or ""
        src = code.encode("utf8", "replace")
        try:
            tree = _PARSER.parse(src)
        except Exception:
            continue
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.type == "call_expression":
                fn = n.child_by_field_name("function")
                if fn:
                    nm = src[fn.start_byte:fn.end_byte].decode("utf8", "replace")
                    nm = nm.split("->")[-1].split(".")[-1].strip()
                    if _IDENT.match(nm):
                        counts[nm] += 1
                        if nm not in examples:
                            ex = " ".join(src[n.start_byte:n.end_byte]
                                          .decode("utf8", "replace").split())
                            if len(ex) <= 160:
                                examples[nm] = ex
            stack.extend([c for c in n.children if c.is_named])
    return counts, examples


def write_blind(dataset, items, examples):
    """Blind annotation sheet: no predicted class, shuffled, with context."""
    rng = random.Random(1)
    items = list(items)
    rng.shuffle(items)
    out = Path(__file__).parent / "annotations" / f"blind_{dataset}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "name", "occurrences", "example_call"])
        for i, (nm, occ, _) in enumerate(items):
            w.writerow([i, nm, occ, examples.get(nm, "")])
    print(f"wrote blind sheet ({len(items)} names) -> {out}")


def main():
    p = argparse.ArgumentParser(description="Build phi_type gold-set sample")
    p.add_argument("--dataset", type=str, default="devign")
    p.add_argument("--max_samples", type=int, default=4000)
    p.add_argument("--per_class", type=int, default=25,
                   help="Sampled names per PREDICTED class (precision arm)")
    p.add_argument("--tail", type=int, default=150,
                   help="Random unmatched names (recall arm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--sample_n", type=int, default=0,
                   help="use the random experimental sample of this size")
    p.add_argument("--names_from", type=str, default=None,
                   help="re-export an existing gold-set TSV as a blind file")
    args = p.parse_args()

    if not TS_AVAILABLE:
        raise SystemExit("tree-sitter unavailable")

    counts, examples = collect_callees(args.dataset, args.max_samples,
                                       args.sample_n, args.seed)
    total_sites = sum(counts.values())
    print(f"{args.dataset}: {len(counts)} distinct callees over "
          f"{total_sites} call sites")

    if args.names_from:
        rows = list(csv.DictReader(open(args.names_from), delimiter="\t"))
        write_blind(args.dataset, [(r["name"], int(r["occurrences"]),
                                    r["predicted"]) for r in rows],
                    examples)
        return

    by_pred = collections.defaultdict(list)
    for nm, c in counts.items():
        by_pred[classify_api(nm) or "NONE"].append((nm, c))

    rng = random.Random(args.seed)
    rows = []
    for cls, items in sorted(by_pred.items()):
        items.sort(key=lambda t: -t[1])
        if cls == "NONE":
            # Recall arm: frequency-weighted random sample of unmatched names,
            # so we can estimate how many genuine API calls phi_type misses.
            pool = [nm for nm, c in items for _ in range(min(c, 5))]
            picked = list(dict.fromkeys(rng.sample(pool, min(len(pool), args.tail))))
            chosen = [(nm, dict(items)[nm]) for nm in picked][:args.tail]
        else:
            # Precision arm: the head (what actually drives the statistics)
            # plus a random tail of the same class.
            head = items[:args.per_class // 2]
            rest = items[args.per_class // 2:]
            tail = rng.sample(rest, min(len(rest), args.per_class - len(head)))
            chosen = head + tail
        for nm, c in chosen:
            rows.append({"name": nm, "occurrences": c,
                         "predicted": cls, "true": "", "note": ""})

    out = Path(args.output or
               Path(__file__).parent / f"goldset_{args.dataset}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "occurrences", "predicted",
                                          "true", "note"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    write_blind(args.dataset, [(r["name"], r["occurrences"], r["predicted"])
                               for r in rows], examples)

    print(f"wrote {len(rows)} names to annotate -> {out}")
    print("predicted-class distribution in the sample:")
    for cls, n in collections.Counter(r["predicted"] for r in rows).most_common():
        cov = sum(c for nm, c in by_pred[cls])
        print(f"   {cls:<16} sampled {n:>4}   corpus call sites {cov:>6} "
              f"({100*cov/total_sites:.2f}%)")


if __name__ == "__main__":
    main()
