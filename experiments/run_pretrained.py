"""
Pretrained-model reference: frozen UniXcoder embeddings + a linear probe.

Added to answer the question of whether the representation conclusion is an
artefact of three small GNNs. A pre-trained code transformer reads the RAW
source (identifiers included), so it is the strongest 'lexical' competitor in
the study. It is evaluated on exactly the folds of run_representation.py
(same sample, same deduplication, same project-level GroupKFold) and its
held-out scores are stored in the same format, so the same paired tests apply.

Limits, stated plainly: the encoder is frozen (no fine-tuning; this machine
has no GPU), inputs are truncated to 256 sub-word tokens, and the probe is an
L2-regularised logistic regression with balanced class weights. It is a
reference point, not a tuned LineVul reproduction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from data.loaders.sampled import load_sampled, CORPORA
from data.loaders.real_datasets import dedupe_functions
from utils.protocol import project_group_kfold, trivial_all_positive

MODEL = "microsoft/unixcoder-base"
CACHE = Path(".cache/embeddings")


def embed(corpus, data, n, seed, max_len, bs, threads):
    path = CACHE / f"{corpus}_n{n}_s{seed}_unixcoder_{max_len}.npy"
    if path.exists():
        return np.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(threads)
    hf, code_key, _, _ = CORPORA[corpus]
    ds = load_dataset(hf, split="train")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    codes = [ds[d.src_index][code_key] or "" for d in data]
    order = np.argsort([len(c) for c in codes])      # length-sorted batches
    out = np.zeros((len(codes), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = tok([codes[i] for i in idx], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).numpy()
            if (b // bs) % 20 == 0:
                print(f"  embedded {b}/{len(order)} ({time.time()-t0:.0f}s)",
                      flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bigvul")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data, _, _ = load_sampled(args.dataset, args.n, args.sample_seed, 8)
    data = dedupe_functions(data)
    X = embed(args.dataset, data, args.n, args.sample_seed, args.max_len,
              args.batch_size, args.threads)
    y = np.array([float(d.y[0]) for d in data])
    folds = list(project_group_kfold(data, 5, 2))

    cell = {"auc": [], "auprc": [], "folds": [], "scores": {}}
    res = {"_config": {"dataset": args.dataset, "n": args.n,
                       "sample_seed": args.sample_seed, "model": MODEL,
                       "max_len": args.max_len, "taxonomy_size": 8,
                       "edge_set": "all", "probe": "logreg C=0.1 balanced"},
           "_folds": [f[2] for f in folds], "unixcoder|probe": cell,
           "_trivial": {}}
    for fi, (tr, te, info) in enumerate(folds):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        s = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc = roc_auc_score(y[te], s) if len(np.unique(y[te])) > 1 else 0.5
        ap = average_precision_score(y[te], s) if len(np.unique(y[te])) > 1 else 0.0
        cell["auc"].append(float(auc)); cell["auprc"].append(float(ap))
        cell["folds"].append(fi)
        cell["scores"][str(fi)] = {
            "y": [float(v) for v in y[te]],
            "s": [round(float(v), 5) for v in s],
            "groups": [data[i].project for i in te]}
        res["_trivial"][str(fi)] = trivial_all_positive(list(y[te]))
        print(f"fold {fi}: AUC={auc:.4f} AUPRC={ap:.4f}")
    out = Path(__file__).parent / args.output
    json.dump(res, open(out, "w"))
    print(f"mean AUC {np.mean(cell['auc']):.4f}  -> {out}")


if __name__ == "__main__":
    main()
