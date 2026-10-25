"""
Paired analysis for the representation x backbone study.

All result files for a corpus (results/<corpus>_*.json) are merged into one
table of cells keyed `mode|backbone[@variant]`, where the variant records a
non-default edge set (`@ast`, ...) or taxonomy size (`@t16`). Every comparison
is a paired difference A - B on the folds BOTH cells completed, and the script
refuses to merge files whose fold assignments differ.

For each contrast it reports:

  * the per-fold mean difference and win count,
  * a Wilcoxon signed-rank test over folds (exact, two-sided) -- folds from
    the second repeat share some test projects with the first, so this test
    is reported as a complement to, not a replacement for, the next item,
  * a cluster bootstrap over held-out PROJECTS on the pooled predictions
    (95% percentile interval, two-sided p), repeated under several seeds to
    record how stable the verdict is,
  * a Holm-adjusted p within the family of contrasts for that corpus/metric,
  * the same bootstrap with the single most dominant held-out project
    (Android in BigVul) removed.

Per-project AUCs for the largest held-out projects are written alongside.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.protocol import cluster_bootstrap_ci

RESULTS = Path(__file__).parent / "results"
CORPORA = ("bigvul", "diversevul")

# (A, B): the reported quantity is metric(A) - metric(B).
VS_LEXICAL = ["op", "kind", "op_kind", "lex_kind", "anon", "anon_kind"]
DIRECT = [("op_kind", "kind"), ("kind", "lex_kind"), ("kind", "anon_kind"),
          ("op_kind", "lex_kind"), ("anon_kind", "lex_kind"),
          ("anon", "lexical"), ("kind", "op"), ("lex_kind", "lexical"),
          ("kind_low", "kind"), ("op_wide", "op"), ("kind_low", "op")]
VARIANTS = ["ast", "ast_sib", "ast_du", "t16", "t32"]


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5


def _ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) > 1 else 0.0


def merge_corpus(corpus):
    cells, folds, cfg_by_file = {}, None, {}
    for path in sorted(glob.glob(str(RESULTS / f"{corpus}_*.json"))):
        d = json.load(open(path))
        cfg = d.get("_config", {})
        cfg_by_file[Path(path).name] = cfg
        f = [tuple(x["test_projects"]) for x in d.get("_folds", [])]
        if folds is None:
            folds = f
        elif f != folds:
            raise SystemExit(f"fold assignment differs in {path}")
        suffix = ""
        if cfg.get("edge_set", "all") != "all":
            suffix += "@" + cfg["edge_set"]
        if cfg.get("taxonomy_size", 8) != 8:
            suffix += f"@t{cfg['taxonomy_size']}"
        for key, cell in d.items():
            if key.count("|") != 1:
                continue
            k = key + suffix
            if k in cells:
                # same key from two files: union of folds
                for i, fo in enumerate(cell["folds"]):
                    if fo not in cells[k]["folds"]:
                        for m in ("auc", "auprc"):
                            cells[k][m].append(cell[m][i])
                        cells[k]["folds"].append(fo)
                cells[k]["scores"].update(cell.get("scores", {}))
            else:
                cells[k] = cell
    return cells, folds, cfg_by_file


def pooled_rows(a, b, common, drop_project=None):
    ys, sa, sb, gs = [], [], [], []
    for f in common:
        fa, fb = a["scores"].get(str(f)), b["scores"].get(str(f))
        if not fa or not fb:
            continue
        keep = [i for i, g in enumerate(fa["groups"]) if g != drop_project]
        ys += [fa["y"][i] for i in keep]
        sa += [fa["s"][i] for i in keep]
        sb += [fb["s"][i] for i in keep]
        gs += [f"{f}:{fa['groups'][i]}" for i in keep]
    return ys, sa, sb, gs


def contrast(cells, ka, kb, metric, n_boot, n_seeds, drop_project=None):
    a, b = cells.get(ka), cells.get(kb)
    if not a or not b:
        return None
    common = sorted(set(a["folds"]) & set(b["folds"]))
    if len(common) < 2:
        return None
    fn = _auc if metric == "auc" else _ap
    da = np.array([a[metric][a["folds"].index(f)] for f in common])
    db = np.array([b[metric][b["folds"].index(f)] for f in common])
    d = da - db
    try:
        w = wilcoxon(d, alternative="two-sided", method="auto")
        w_p = float(w.pvalue)
    except ValueError:                      # all differences zero
        w_p = 1.0
    ys, sa, sb, gs = pooled_rows(a, b, common, drop_project)
    ci, excl = None, []
    for s in range(n_seeds):
        c = cluster_bootstrap_ci(ys, sa, sb, gs, fn, n_boot=n_boot, seed=s)
        if s == 0:
            ci = c
        excl.append(1.0 if (c["lo"] > 0 or c["hi"] < 0) else 0.0)
    return {"n_folds": len(common), "mean_a": float(da.mean()),
            "mean_b": float(db.mean()), "std_a": float(da.std()),
            "std_b": float(db.std()),
            "per_fold_mean": float(d.mean()), "wins": int((d > 0).sum()),
            "wilcoxon_p": w_p, **ci,
            "seed_stability": float(np.mean(excl)), "n_seeds": n_seeds}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def analyse_corpus(corpus, n_boot, n_seeds):
    cells, folds, cfgs = merge_corpus(corpus)
    if not cells:
        return None
    backbones = sorted({k.split("|")[1].split("@")[0] for k in cells})
    out = {"auc": {}, "auprc": {}, "_cells": sorted(cells),
           "_n_folds": len(folds), "_configs": cfgs}
    for metric in ("auc", "auprc"):
        res = out[metric]
        for bb in backbones:
            pairs = [(f"{m}|{bb}", f"lexical|{bb}", f"{m}~lexical")
                     for m in VS_LEXICAL + ["kind_low", "op_wide"]]
            pairs += [(f"{a}|{bb}", f"{b}|{bb}", f"{a}~{b}") for a, b in DIRECT]
            for v in VARIANTS:
                suffix = f"@{v}"
                base = "op_kind" if v.startswith("t") else "kind"
                ref = f"{base}|{bb}"
                pairs.append((f"{base}|{bb}{suffix}", ref, f"{base}{suffix}~{base}"))
            for ka, kb, name in pairs:
                r = contrast(cells, ka, kb, metric, n_boot, n_seeds)
                if r:
                    res[f"{bb}|{name}"] = r
                    r_x = contrast(cells, ka, kb, metric, n_boot, 3,
                                   drop_project="Android")
                    if r_x and corpus == "bigvul":
                        res[f"{bb}|{name}"]["without_android"] = {
                            k: r_x[k] for k in ("diff", "lo", "hi",
                                                "p_two_sided", "n_folds")}
        keys = list(res)
        adj = holm([res[k]["p_two_sided"] for k in keys])
        adj_w = holm([res[k]["wilcoxon_p"] for k in keys])
        for k, a, aw in zip(keys, adj, adj_w):
            res[k]["p_holm"] = a
            res[k]["wilcoxon_p_holm"] = aw
    # architecture spread under a fixed representation
    spread = {}
    for mode in ("lexical", "kind", "lex_kind", "op_kind"):
        v = {bb: float(np.mean(cells[f"{mode}|{bb}"]["auc"]))
             for bb in backbones if f"{mode}|{bb}" in cells}
        if len(v) > 1:
            spread[mode] = {"by_backbone": v,
                            "range": max(v.values()) - min(v.values())}
    out["_architecture_spread"] = spread
    # per-project AUC for the largest held-out projects (pooled over folds)
    per_proj = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for key in ("lexical|ggnn", "lex_kind|ggnn", "kind|ggnn", "op_kind|ggnn"):
        c = cells.get(key)
        if not c:
            continue
        for f, sc in c.get("scores", {}).items():
            for y, s, g in zip(sc["y"], sc["s"], sc["groups"]):
                per_proj[g][key][0].append(y)
                per_proj[g][key][1].append(s)
    rows = []
    for g, d in per_proj.items():
        ref = d.get("lexical|ggnn")
        if not ref:
            continue
        npos = int(sum(ref[0]))
        if npos < 10 or npos == len(ref[0]):
            continue
        row = {"project": g, "n": len(ref[0]), "n_vuln": npos}
        for key, (y, s) in d.items():
            row[key.split("|")[0]] = float(_auc(np.array(y), np.array(s)))
        rows.append(row)
    rows.sort(key=lambda r: -r["n_vuln"])
    out["_per_project"] = rows[:10]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--corpora", default=",".join(CORPORA))
    args = p.parse_args()
    summary = {}
    for c in args.corpora.split(","):
        r = analyse_corpus(c, args.n_boot, args.n_seeds)
        if not r:
            continue
        summary[c] = r
        print(f"\n=== {c}: {r['_n_folds']} folds, {len(r['_cells'])} cells ===")
        for k, s in r["auc"].items():
            tag = ("n.s." if not (s["lo"] > 0 or s["hi"] < 0) else "sig")
            print(f"  {k:<28} d={s['diff']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] "
                  f"p={s['p_two_sided']:.3f} wilcoxon p={s['wilcoxon_p']:.3f} "
                  f"wins {s['wins']}/{s['n_folds']} {tag}")
    out = RESULTS / "representation_analysis.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
