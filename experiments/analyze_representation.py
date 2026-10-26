"""
Paired analysis for the representation x backbone study.

All result files for a corpus (results/<corpus>_*.json) are merged into one
table of cells keyed `mode|backbone[@variant]`; the variant records a non-default
edge set (`@ast`, ...) or taxonomy size (`@t16`). Every comparison is a paired
difference A - B on the folds BOTH cells completed, and the script refuses to
merge files whose fold assignments differ.

The reported difference is the MEAN OVER FOLDS of the per-fold metric
difference, i.e. the same quantity as the per-fold table columns. Uncertainty:

  * a cluster bootstrap over held-out PROJECTS: in each replicate projects are
    drawn with replacement (a project is one cluster no matter how many folds
    it is tested in), every function of a drawn project is repeated as often
    as the project was drawn, the metric is recomputed inside each fold, and
    the replicate statistic is the mean over folds (95% percentile interval,
    two-sided p);
  * an exact Wilcoxon signed-rank test over the per-fold differences, and a
    second one over the five repeat-averaged differences (fold k and k+5 come
    from the two repeats of the project-level split, which are not
    independent; with five blocks the smallest attainable p is 0.0625);
  * Holm-adjusted p-values within each corpus/metric family;
  * the same bootstrap with the most dominant held-out project removed.

Metrics: `auc`, `auprc`, and `auc_ss`, the AUC inside bins of function size
(a score that carries no information beyond length has auc_ss = 0.5).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon, rankdata

RESULTS = Path(__file__).parent / "results"
CORPORA = ("bigvul", "diversevul")
N_BINS = 5

VS_LEXICAL = ["op", "kind", "op_kind", "lex_kind", "anon", "anon_kind"]
DIRECT = [("op_kind", "kind"), ("kind", "lex_kind"), ("kind", "anon_kind"),
          ("kind", "anon"), ("op_kind", "lex_kind"), ("anon_kind", "lex_kind"),
          ("anon", "lexical"), ("kind", "op"), ("lex_kind", "lexical"),
          ("kind_low", "kind"), ("op_wide", "op"), ("kind_low", "op")]
VARIANTS = ["ast", "ast_sib", "ast_du", "t16", "t32"]
# bootstrap budget per metric: (resamples, seeds)
BUDGET = {"auc": (1000, 5), "auprc": (500, 1), "auc_ss": (500, 1)}


# ── metrics ──────────────────────────────────────────────────────────────────

def _auc(y, s):
    """Rank-based AUC with average ranks for ties."""
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(s))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ap(y, s):
    """Average precision (step-wise, ties handled at unique score levels)."""
    y = np.asarray(y)
    s = np.asarray(s)
    npos = int((y == 1).sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    last = np.r_[ss[1:] != ss[:-1], True]            # last index of each level
    tp_l, n_l = tp[last], (np.arange(len(ys)) + 1)[last]
    prec = tp_l / n_l
    rec = tp_l / npos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(((rec - rec_prev) * prec).sum())


def _auc_ss(y, s, bins):
    """AUC within bins of function size, weighted by positive-negative pairs."""
    y, s, bins = np.asarray(y), np.asarray(s), np.asarray(bins)
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        yy = y[m]
        n1 = yy.sum()
        if n1 == 0 or n1 == len(yy):
            continue
        w = float(n1 * (len(yy) - n1))
        num += w * _auc(yy, s[m])
        den += w
    return num / den if den else float("nan")


METRIC = {"auc": lambda y, s, b: _auc(y, s),
          "auprc": lambda y, s, b: _ap(y, s),
          "auc_ss": _auc_ss}


# ── data ─────────────────────────────────────────────────────────────────────

def add_size_stratified(cells, corpus):
    """Attach per-fold size-stratified AUC (`auc_ss`) and the bins used."""
    sz = RESULTS / f"{corpus}_size.json"
    if not sz.exists():
        return None
    d = json.load(open(sz))["size|rule"]["scores"]
    allsz = np.concatenate([np.asarray(v["s"]) for v in d.values()])
    edges = np.quantile(allsz, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins_by_fold = {f: np.searchsorted(edges, np.asarray(v["s"]), side="right")
                    for f, v in d.items()}
    for key, cell in cells.items():
        cell["auc_ss"] = []
        for f in cell["folds"]:
            sc = cell["scores"].get(str(f))
            if not sc or str(f) not in bins_by_fold:
                cell["auc_ss"].append(float("nan"))
            else:
                cell["auc_ss"].append(
                    _auc_ss(sc["y"], sc["s"], bins_by_fold[str(f)]))
    return bins_by_fold


def merge_corpus(corpus):
    cells, folds, cfg_by_file = {}, None, {}
    for path in sorted(glob.glob(str(RESULTS / f"{corpus}_*.json"))):
        if Path(path).name.startswith(f"{corpus}_analysis"):
            continue
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
                for i, fo in enumerate(cell["folds"]):
                    if fo not in cells[k]["folds"]:
                        for m in ("auc", "auprc"):
                            cells[k][m].append(cell[m][i])
                        cells[k]["folds"].append(fo)
                cells[k]["scores"].update(cell.get("scores", {}))
            else:
                cells[k] = cell
    bins = add_size_stratified(cells, corpus)
    return cells, folds, cfg_by_file, bins


def trivial_prevalences(corpus):
    prev = {}
    for path in sorted(glob.glob(str(RESULTS / f"{corpus}_*.json"))):
        d = json.load(open(path))
        for f, t in (d.get("_trivial") or {}).items():
            prev[int(f)] = t["precision"]
    return [prev[k] for k in sorted(prev)]


# ── paired contrast ──────────────────────────────────────────────────────────

class Pair:
    """Aligned per-fold arrays for two cells on their common folds."""

    def __init__(self, a, b, common, bins_by_fold, drop_project=None):
        self.folds = []
        self.y, self.sa, self.sb, self.proj, self.bins = [], [], [], [], []
        proj_ids = {}
        for f in common:
            fa, fb = a["scores"].get(str(f)), b["scores"].get(str(f))
            if not fa or not fb:
                continue
            keep = np.array([g != drop_project for g in fa["groups"]])
            self.folds.append(f)
            self.y.append(np.asarray(fa["y"])[keep])
            self.sa.append(np.asarray(fa["s"])[keep])
            self.sb.append(np.asarray(fb["s"])[keep])
            ids = []
            for g, k in zip(fa["groups"], keep):
                if k:
                    ids.append(proj_ids.setdefault(g, len(proj_ids)))
            self.proj.append(np.asarray(ids, dtype=int))
            self.bins.append(np.asarray(bins_by_fold[str(f)])[keep]
                             if bins_by_fold else None)
        self.n_proj = len(proj_ids)

    def stat(self, metric, weights=None):
        """Mean over folds of metric(A) - metric(B); `weights` are per-project
        draw counts (None = observed data)."""
        fn = METRIC[metric]
        out = []
        for i in range(len(self.folds)):
            y, sa, sb, pr, bn = (self.y[i], self.sa[i], self.sb[i],
                                 self.proj[i], self.bins[i])
            if weights is not None:
                rep = weights[pr]
                idx = np.repeat(np.arange(len(y)), rep)
                y, sa, sb = y[idx], sa[idx], sb[idx]
                bn = bn[idx] if bn is not None else None
            ma, mb = fn(y, sa, bn), fn(y, sb, bn)
            if ma == ma and mb == mb:
                out.append(ma - mb)
        return float(np.mean(out)) if out else float("nan")

    def bootstrap(self, metric, n_boot, seed):
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, self.n_proj, self.n_proj),
                            minlength=self.n_proj)
            d = self.stat(metric, w)
            if d == d:
                diffs.append(d)
        return np.asarray(diffs)


def contrast(cells, bins, ka, kb, metric, n_boot, n_seeds, drop_project=None):
    a, b = cells.get(ka), cells.get(kb)
    if not a or not b:
        return None
    common = sorted(set(a["folds"]) & set(b["folds"]))
    if len(common) < 2:
        return None
    key = "auc_ss" if metric == "auc_ss" else metric
    da = np.array([a[key][a["folds"].index(f)] for f in common])
    db = np.array([b[key][b["folds"].index(f)] for f in common])
    ok = ~(np.isnan(da) | np.isnan(db))
    d = (da - db)[ok]
    cf = [f for f, k in zip(common, ok) if k]
    pair = Pair(a, b, cf, bins, drop_project)
    observed = pair.stat(metric)

    def wil(x):
        try:
            return float(wilcoxon(x, alternative="two-sided",
                                  method="auto").pvalue)
        except ValueError:
            return 1.0
    # fold k and k+5 are the two repeats of the same 5-way split
    blocks = []
    for k in range(5):
        pair_d = [dd for f, dd in zip(cf, d) if f % 5 == k]
        if pair_d:
            blocks.append(np.mean(pair_d))
    ci = None
    excl = []
    for s in range(n_seeds):
        bs = pair.bootstrap(metric, n_boot, s)
        if len(bs) == 0:
            continue
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        if s == 0:
            ci = {"diff": observed, "lo": float(lo), "hi": float(hi),
                  "p_two_sided": float(min(1.0, p)), "n_boot": int(len(bs)),
                  "n_clusters": pair.n_proj}
        excl.append(1.0 if (lo > 0 or hi < 0) else 0.0)
    if ci is None:
        return None
    return {"n_folds": len(cf), "mean_a": float(da[ok].mean()),
            "mean_b": float(db[ok].mean()),
            "per_fold_mean": float(d.mean()), "wins": int((d > 0).sum()),
            "wilcoxon_p": wil(d),
            "wilcoxon_p_blocks": wil(np.asarray(blocks)) if len(blocks) > 1 else 1.0,
            **ci, "seed_stability": float(np.mean(excl)), "n_seeds": n_seeds}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


# ── per-corpus analysis ──────────────────────────────────────────────────────

def analyse_corpus(corpus, n_boot_scale=1.0):
    cells, folds, cfgs, bins = merge_corpus(corpus)
    if not cells:
        return None
    backbones = sorted({k.split("|")[1].split("@")[0] for k in cells})
    # reference systems that are not (mode, backbone) cells of the grid
    refs = ("rule", "probe", "gbm", "ft")
    out = {"auc": {}, "auprc": {}, "auc_ss": {}, "_cells": sorted(cells),
           "_n_folds": len(folds), "_configs": cfgs,
           "_trivial_prevalence": trivial_prevalences(corpus)}
    for metric in ("auc", "auprc", "auc_ss"):
        res = out[metric]
        nb, ns = BUDGET[metric]
        nb = max(100, int(nb * n_boot_scale))
        for bb in backbones:
            if bb in ("rule", "probe", "gbm", "ft"):
                continue
            pairs = [(f"{m}|{bb}", f"lexical|{bb}", f"{m}~lexical")
                     for m in VS_LEXICAL + ["kind_low", "op_wide"]]
            pairs += [(f"{a}|{bb}", f"{b}|{bb}", f"{a}~{b}") for a, b in DIRECT]
            for v in VARIANTS:
                base = "op_kind" if v.startswith("t") else "kind"
                pairs.append((f"{base}|{bb}@{v}", f"{base}|{bb}", f"{base}@{v}~{base}"))
            for ka, kb, name in pairs:
                r = contrast(cells, bins, ka, kb, metric, nb, ns)
                if not r:
                    continue
                res[f"{bb}|{name}"] = r
                if metric == "auc" and corpus == "bigvul" and "@" not in name:
                    rx = contrast(cells, bins, ka, kb, metric, 500, 1,
                                  drop_project="Android")
                    if rx:
                        r["without_android"] = {
                            k: rx[k] for k in ("diff", "lo", "hi",
                                               "p_two_sided", "n_folds")}
        for ref, rname in (("size|rule", "size"), ("unixcoder|probe", "unixcoder"),
                           ("trail|rule", "trail"), ("format|gbm", "format"),
                           ("linevul|ft", "linevul"),
                           ("linevulnorm|ft", "linevulnorm"),
                           ("unixnorm|probe", "unixnorm")):
            for m in ("lexical", "lex_kind", "kind", "op_kind"):
                for bb in backbones:
                    if bb in ("rule", "probe", "gbm", "ft"):
                        continue
                    r = contrast(cells, bins, f"{m}|{bb}", ref, metric, nb, ns)
                    if r:
                        res[f"{bb}|{m}~{rname}"] = r
        keys = list(res)
        for fld, src in (("p_holm", "p_two_sided"), ("wilcoxon_p_holm", "wilcoxon_p")):
            adj = holm([res[k][src] for k in keys])
            for k, v in zip(keys, adj):
                res[k][fld] = v

    # architecture spread among the message-passing networks only
    spread = {}
    for mode in ("lexical", "kind", "lex_kind", "op_kind"):
        v = {bb: float(np.mean(cells[f"{mode}|{bb}"]["auc"]))
             for bb in ("ggnn", "gat", "gin") if f"{mode}|{bb}" in cells}
        if len(v) > 1:
            spread[mode] = {"by_backbone": v, "range": max(v.values()) - min(v.values())}
    out["_architecture_spread"] = spread

    # per-project AUC: for each (fold, project) with both classes compute the
    # AUC, then average over the folds in which the project is tested
    per_proj = defaultdict(lambda: defaultdict(list))
    n_eval = defaultdict(lambda: [0, 0])
    for mode in ("lexical", "lex_kind", "kind", "op_kind"):
        c = cells.get(f"{mode}|ggnn")
        if not c:
            continue
        for f, sc in c.get("scores", {}).items():
            y = np.asarray(sc["y"]); s = np.asarray(sc["s"])
            g = np.asarray(sc["groups"])
            for p in np.unique(g):
                m = g == p
                a = _auc(y[m], s[m])
                if a == a:
                    per_proj[p][mode].append(a)
                if mode == "lexical":
                    n_eval[p][0] += int(m.sum())
                    n_eval[p][1] += int(y[m].sum())
    rows = []
    for p, d in per_proj.items():
        if n_eval[p][1] < 10 or "lexical" not in d:
            continue
        row = {"project": p, "n": n_eval[p][0], "n_vuln": n_eval[p][1]}
        for mode, v in d.items():
            row[mode] = float(np.mean(v))
        rows.append(row)
    rows.sort(key=lambda r: -r["n_vuln"])
    out["_per_project"] = rows[:10]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, choices=list(CORPORA))
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply every bootstrap budget (quick checks)")
    args = p.parse_args()
    r = analyse_corpus(args.corpus, args.scale)
    if not r:
        raise SystemExit("no results")
    print(f"=== {args.corpus}: {r['_n_folds']} folds, {len(r['_cells'])} cells ===")
    for k, s in r["auc"].items():
        tag = "n.s." if not (s["lo"] > 0 or s["hi"] < 0) else "sig"
        print(f"  {k:<28} d={s['diff']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] "
              f"p={s['p_two_sided']:.3f} W={s['wilcoxon_p']:.3f} "
              f"Wblk={s['wilcoxon_p_blocks']:.3f} wins {s['wins']}/{s['n_folds']} {tag}")
    out = RESULTS / f"analysis_{args.corpus}.json"
    json.dump(r, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
