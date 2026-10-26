"""
Frozen UniXcoder + linear probe on FORMAT-NORMALISED source, GPU version of
run_pretrained.py for the colab_data export (same folds, same probe).
"""

import argparse
import gzip
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="codes_norm")
    ap.add_argument("--cell", default="unixnorm|probe")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    d = json.load(gzip.open(args.data, "rt"))
    codes, y, proj = d[args.field], np.asarray(d["y"]), d["project"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base").to(dev).eval()
    order = np.argsort([len(c) for c in codes])
    X = np.zeros((len(codes), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 64):
            idx = order[b:b + 64]
            enc = tok([codes[i] for i in idx], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda",
                                dtype=torch.float16):
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            X[idx] = ((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
    cell = {"auc": [], "auprc": [], "folds": [], "scores": {}}
    res = {"_config": {"dataset": d["dataset"], "field": args.field,
                       "model": "microsoft/unixcoder-base", "max_len": args.max_len,
                       "taxonomy_size": 8, "edge_set": "all",
                       "probe": "logreg C=0.1 balanced"},
           "_folds": [f["info"] for f in d["folds"]], args.cell: cell}
    for fi, f in enumerate(d["folds"]):
        tr, te = f["train"], f["test"]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        s = clf.predict_proba(sc.transform(X[te]))[:, 1]
        cell["auc"].append(float(roc_auc_score(y[te], s)))
        cell["auprc"].append(float(average_precision_score(y[te], s)))
        cell["folds"].append(fi)
        cell["scores"][str(fi)] = {"y": [float(v) for v in y[te]],
                                   "s": [round(float(v), 5) for v in s],
                                   "groups": [proj[i] for i in te]}
        print(f"fold {fi}: AUC={cell['auc'][-1]:.4f}", flush=True)
    json.dump(res, open(args.out, "w"))
    print("mean AUC", float(np.mean(cell["auc"])))


if __name__ == "__main__":
    main()
