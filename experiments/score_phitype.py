"""
RQ1, second version: validate phi as a static analysis on the corpora the
representation experiments actually use, against two independent blind
annotators plus a third pass on their disagreements.

Inputs (experiments/):
  goldset_<corpus>.tsv                  names, occurrences, phi's prediction
  annotations/annotator_{A,B}_<corpus>.tsv   blind labels (id,label,note)
  annotations/annotator_C_<corpus>.tsv       adjudication of A/B disagreements
  goldset_devign.tsv  (+ annotate_goldset.ground_truth)   the ORIGINAL
                       single-annotator Devign labels, kept for comparison

Outputs results/phitype_validation.json: inter-annotator agreement (raw and
Cohen's kappa), per-class precision/recall of phi against the consensus with
Wilson 95% intervals (so thin classes are visibly thin), per-corpus scores,
and sensitivity to which annotator is used as ground truth.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import math
from collections import Counter, defaultdict

from data.loaders.api_classes import classify_api

HERE = Path(__file__).parent
CORPORA = ["devign", "bigvul", "diversevul"]
LABELS = ["MEMORY_ALLOC", "MEMORY_REALLOC", "MEMORY_FREE", "MEMORY_COPY",
          "MEMORY_SET", "STRING_COPY", "STRING_CONCAT", "STRING_FORMAT",
          "STRING_LENGTH", "IO_CALL", "NONE"]


def read_tsv(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(a) | set(b)) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_corpus(c):
    gold = read_tsv(HERE / f"goldset_{c}.tsv")
    blind = read_tsv(HERE / "annotations" / f"blind_{c}.tsv")
    # Blind sheet ids index a shuffled list; map id -> name.
    by_id = {r["id"]: r["name"] for r in blind}
    lab = {}
    for ann in "ABC":
        p = HERE / "annotations" / f"annotator_{ann}_{c}.tsv"
        if p.exists():
            lab[ann] = {by_id[r["id"]]: r["label"].strip() for r in read_tsv(p)}
    pred = {r["name"]: (classify_api(r["name"]) or "NONE") for r in gold}
    occ = {r["name"]: int(r["occurrences"]) for r in gold}
    return pred, occ, lab


def consensus(lab):
    out, adjudicated, unresolved = {}, 0, 0
    for nm in lab["A"]:
        a, b = lab["A"][nm], lab["B"].get(nm)
        if a == b:
            out[nm] = a
            continue
        adjudicated += 1
        c = lab.get("C", {}).get(nm)
        if c in (a, b):
            out[nm] = c
        elif c:
            unresolved += 1
            out[nm] = c
        else:
            unresolved += 1
            out[nm] = a
    return out, adjudicated, unresolved


def score(pred, truth, occ):
    tp, fp, fn = Counter(), Counter(), Counter()
    wt = wtot = 0
    for nm, p in pred.items():
        t = truth[nm]
        wtot += occ[nm]
        if p == t:
            wt += occ[nm]
            if t != "NONE":
                tp[t] += 1
        else:
            if p != "NONE":
                fp[p] += 1
            if t != "NONE":
                fn[t] += 1
    return tp, fp, fn, wt / max(1, wtot)


def per_class(tp, fp, fn):
    out = {}
    for c in LABELS:
        if c == "NONE":
            continue
        pr_n, rc_n = tp[c] + fp[c], tp[c] + fn[c]
        if pr_n == 0 and rc_n == 0:
            continue
        out[c] = {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                  "precision": tp[c] / pr_n if pr_n else None,
                  "recall": tp[c] / rc_n if rc_n else None,
                  "p_ci": wilson(tp[c], pr_n) if pr_n else None,
                  "r_ci": wilson(tp[c], rc_n) if rc_n else None}
    return out


def macro(pc, key):
    v = [x[key] for x in pc.values() if x[key] is not None]
    return sum(v) / len(v) if v else float("nan")


def main():
    res = {"corpora": {}, "agreement": {}, "pooled": {}}
    TP, FP, FN = Counter(), Counter(), Counter()
    all_a, all_b = [], []
    for c in CORPORA:
        pred, occ, lab = load_corpus(c)
        if "A" not in lab or "B" not in lab:
            print(f"{c}: annotations missing, skipped")
            continue
        names = list(pred)
        a = [lab["A"][n] for n in names]
        b = [lab["B"][n] for n in names]
        all_a += a
        all_b += b
        po, k = kappa(a, b)
        truth, adj, unres = consensus(lab)
        tp, fp, fn, wacc = score(pred, truth, occ)
        for d, s in ((TP, tp), (FP, fp), (FN, fn)):
            d.update(s)
        pc = per_class(tp, fp, fn)
        sens = {}
        for ann in "AB":
            t2, f2, n2, w2 = score(pred, lab[ann], occ)
            p2 = per_class(t2, f2, n2)
            sens[ann] = {"macro_precision": macro(p2, "precision"),
                         "macro_recall": macro(p2, "recall"),
                         "weighted_accuracy": w2}
        entry = {"n_names": len(names), "agreement": po, "kappa": k,
                 "disagreements": sum(x != y for x, y in zip(a, b)),
                 "adjudicated": adj, "unresolved_three_way": unres,
                 "per_class": pc, "macro_precision": macro(pc, "precision"),
                 "macro_recall": macro(pc, "recall"),
                 "weighted_accuracy": wacc, "sensitivity": sens,
                 "errors": [{"name": n, "predicted": pred[n], "true": truth[n],
                             "occurrences": occ[n]}
                            for n in names if pred[n] != truth[n]]}
        if c == "devign":
            from experiments.annotate_goldset import ground_truth
            orig = {n: ground_truth(n)[0] for n in names}
            _, ko = kappa([orig[n] for n in names], [truth[n] for n in names])
            entry["kappa_original_vs_consensus"] = ko
            entry["agreement_original_vs_consensus"] = sum(
                orig[n] == truth[n] for n in names) / len(names)
        res["corpora"][c] = entry
        print(f"{c}: n={len(names)} agreement={po:.3f} kappa={k:.3f} "
              f"adjudicated={adj} wacc={wacc:.3f} "
              f"macroP={entry['macro_precision']:.3f} "
              f"macroR={entry['macro_recall']:.3f}")
    pc = per_class(TP, FP, FN)
    po, k = kappa(all_a, all_b)
    res["pooled"] = {"per_class": pc, "macro_precision": macro(pc, "precision"),
                     "macro_recall": macro(pc, "recall"),
                     "agreement": po, "kappa": k, "n_names": len(all_a)}
    json.dump(res, open(HERE / "results" / "phitype_validation.json", "w"),
              indent=2)
    print(f"pooled kappa={k:.3f} agreement={po:.3f}")
    for c, v in pc.items():
        print(f"  {c:<15} P={v['precision']} R={v['recall']} "
              f"tp/fp/fn={v['tp']}/{v['fp']}/{v['fn']}")


if __name__ == "__main__":
    main()
