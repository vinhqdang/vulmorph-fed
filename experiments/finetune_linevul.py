"""
LineVul-style detector: fine-tune CodeBERT (RoBERTa) on raw function source,
evaluated on the same ten project-level folds as every other system.

This is self-contained so it can run on a GPU runtime without the rest of the
repository: it reads colab_data/<corpus>.json.gz (written by
prepare_colab_data.py) and writes results in the format that
analyze_representation.py merges (cell key `linevul|ft`).

Deviations from the published LineVul recipe, stated plainly: only the
function-level classifier is reproduced (no line-level localisation); epochs
are fixed in advance (no validation-based checkpoint selection, matching the
protocol of the other systems); the loss is class-weighted exactly like the
other systems (positive weight = min(n_neg/n_pos, 20)); and training uses
`--epochs` epochs rather than LineVul's longer schedule, for cost.
"""

import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup


def encode(tok, codes, max_len):
    return tok(codes, truncation=True, max_length=max_len, padding=False)["input_ids"]


def batches(ids, order, bs, pad_id):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(ids[j])] = torch.tensor(ids[j])
        yield idx, x, (x != pad_id).long()


def run_fold(fi, fold, ids, y, args, tok, dev):
    torch.manual_seed(1000 + fi)
    np.random.seed(1000 + fi)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).to(dev)
    tr, te = fold["train"], fold["test"]
    if args.smoke:
        tr, te = tr[:48], te[:24]
    ytr = np.asarray([y[i] for i in tr])
    n_pos = max(1, int(ytr.sum()))
    w = torch.tensor([1.0, min((len(ytr) - n_pos) / n_pos, 20.0)], device=dev)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = list(np.random.permutation(len(tr)))
        tot = 0.0
        for idx, x, m in batches([ids[tr[i]] for i in range(len(tr))], order,
                                 args.batch, tok.pad_token_id):
            yb = torch.tensor([ytr[i] for i in idx], dtype=torch.long, device=dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda",
                                dtype=torch.float16):
                logits = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits
            loss = F.cross_entropy(logits.float(), yb, weight=w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += float(loss)
        print(f"  fold {fi} epoch {ep+1}/{args.epochs} loss "
              f"{tot/max(1, len(tr)/args.batch):.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    te_ids = [ids[i] for i in te]
    order = sorted(range(len(te)), key=lambda i: len(te_ids[i]))
    scores = np.zeros(len(te))
    with torch.no_grad():
        for idx, x, m in batches(te_ids, order, 64, tok.pad_token_id):
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda",
                                dtype=torch.float16):
                lg = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits
            p = torch.softmax(lg.float(), dim=-1)[:, 1].cpu().numpy()
            for r, j in enumerate(idx):
                scores[j] = p[r]
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return te, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--folds", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_len, args.epochs, args.folds = 64, 1, "0"

    with gzip.open(args.data, "rt") as f:
        d = json.load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = encode(tok, d["codes"], args.max_len)
    y, proj = d["y"], d["project"]

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res["_config"] = {"dataset": d["dataset"], "model": args.model,
                      "max_len": args.max_len, "epochs": args.epochs,
                      "batch": args.batch, "lr": args.lr, "taxonomy_size": 8,
                      "edge_set": "all", "smoke": args.smoke}
    res["_folds"] = [f["info"] for f in d["folds"]]
    cell = res.setdefault("linevul|ft", {"auc": [], "auprc": [], "folds": [],
                                         "scores": {}})
    only = {int(x) for x in args.folds.split(",") if x}
    for fi, fold in enumerate(d["folds"]):
        if (only and fi not in only) or fi in cell["folds"]:
            continue
        te, s = run_fold(fi, fold, ids, y, args, tok, dev)
        yt = np.asarray([y[i] for i in te])
        auc = roc_auc_score(yt, s) if len(np.unique(yt)) > 1 else 0.5
        ap_ = average_precision_score(yt, s) if len(np.unique(yt)) > 1 else 0.0
        cell["auc"].append(float(auc))
        cell["auprc"].append(float(ap_))
        cell["folds"].append(fi)
        cell["scores"][str(fi)] = {"y": [float(v) for v in yt],
                                   "s": [round(float(v), 5) for v in s],
                                   "groups": [proj[i] for i in te]}
        print(f"fold {fi}: AUC={auc:.4f} AUPRC={ap_:.4f}", flush=True)
        json.dump(res, open(args.out, "w"))
    print("mean AUC", np.mean(cell["auc"]) if cell["auc"] else None)


if __name__ == "__main__":
    main()
