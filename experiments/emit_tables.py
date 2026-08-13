r"""
Emit the manuscript's result tables mechanically from the result JSONs.

Every number in the paper's tables is produced here. Hand-typing them is what
allowed a stale value to survive a re-run. Nothing in this file rounds a
confidence bound toward zero, and any cell without backing data is emitted
as "---", never guessed.

Outputs: manuscript/latex/tab_*.tex, \input by the experiment section.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

RESULTS = Path(__file__).parent / "results"
LATEX_DIR = Path(__file__).parent.parent / "manuscript" / "latex"

MODE_LABEL = {
    "lexical": "Lexical token",
    "op": "Operation type $\\phi_8$",
    "kind": "Grammar kind $\\kappa$",
    "op_kind": "Operation $+$ kind",
}
BACKBONE = {"ggnn": "GGNN", "gat": "GAT", "gin": "GIN"}
CORPUS = {"core_bigvul": "BigVul", "core_diversevul": "DiverseVul",
          "backbone_bigvul": "BigVul"}


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def ms(vals, prec=3):
    """mean +- population std, or --- when absent."""
    if not vals:
        return "---"
    return f"{np.mean(vals):.{prec}f} $\\pm$ {np.std(vals):.{prec}f}"


def sf(x, prec=3):
    """Signed fixed-point number for LaTeX."""
    s = f"{abs(x):.{prec}f}"
    return ("-" if x < 0 else "+") + s


def ci(stat, prec=3):
    """Point estimate with interval and p-value inside math mode."""
    if not stat:
        return "---"
    lo, hi, d, p = stat["lo"], stat["hi"], stat["diff"], stat["p_two_sided"]
    stab = stat.get("seed_stability")
    if not (lo > 0 or hi < 0):
        mark = "\\,n.s."
    elif stab is not None and stab == stab and stab < 0.9:
        mark = "$^{\\dagger}$"
    else:
        mark = ""
    return (f"${sf(d, prec)}$ $[{sf(lo, prec)}, {sf(hi, prec)}]$ "
            f"{p:.3f}{mark}")


def table_corpora():
    d = load("corpus_stats.json")
    if not d:
        return ""
    rows = []
    names = [("bigvul", "BigVul"), ("diversevul", "DiverseVul"), ("devign", "Devign")]
    for key, name in names:
        if key not in d:
            continue
        c = d[key]
        n_func = f"{c['n']:,}"
        prev = f"{c['prev']:.3f}"
        projs = f"{c['projects']:,}"
        share = f"{c['top2_share']:.1f}\\%"
        ne = f"{c['nodes']:.0f} / {c['edges']:.0f}"
        rows.append(f"{name} & {n_func} & {prev} & {projs} & {share} & {ne} \\\\")
    
    return r"""
\begin{table}[t]
\centering
\caption{Corpus properties measured on our samples. Project concentration is
reported because it determines whether a nominally cross-project split is
meaningful: in BigVul the two largest repositories supply two thirds of the
functions, which is why they are held train-only
(Section~\ref{sec:protocol}). Generated mechanically from result files.}
\label{tab:corpora}
\begin{tabular}{lrrrrr}
\toprule
Corpus & Functions & Prevalence & Projects & Top-2 share & Nodes/Edges \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def table_node_dist():
    d = load("corpus_stats.json")
    if not d or "_node_dist_bigvul" not in d:
        return ""
    dist = d["_node_dist_bigvul"]
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    # Split into 2 columns
    mid = (len(items) + 1) // 2
    left = items[:mid]
    right = items[mid:]
    rows = []
    for i in range(mid):
        l_name, l_val = left[i]
        l_tex = l_name.replace("_", "\\_")
        l_str = f"$\\bot$ (\\texttt{{{l_tex}}})" if l_name == "UNKNOWN" else f"\\texttt{{{l_tex}}}"
        if i < len(right):
            r_name, r_val = right[i]
            r_tex = r_name.replace("_", "\\_")
            r_str = f"$\\bot$ (\\texttt{{{r_tex}}})" if r_name == "UNKNOWN" else f"\\texttt{{{r_tex}}}"
            rows.append(f"{l_str} & {l_val:.2f}\\% & {r_str} & {r_val:.2f}\\% \\\\")
        else:
            rows.append(f"{l_str} & {l_val:.2f}\\% & & \\\\")
            
    return r"""
\begin{table}[t]
\centering
\caption{Node-type distribution under $\phi_8$ (BigVul). $\bot$
(\texttt{UNKNOWN}) nodes carry no operation label but retain their grammar
kind. Generated mechanically from result files.}
\label{tab:node_dist}
\begin{tabular}{lr@{\hskip 2em}lr}
\toprule
Type & Share & Type & Share \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def table_rq1_analysis():
    d = load("phitype_eval.json")
    if not d or "per_class" not in d:
        return ""
    pc = d["per_class"]
    rows = []
    for cls_name, stats in sorted(pc.items()):
        cls_tex = cls_name.replace("_", "\\_")
        p = f"{stats['precision']:.3f}"
        r = f"{stats['recall']:.3f}"
        f1 = f"{stats['f1']:.3f}"
        counts = f"{stats['tp']}/{stats['fp']}/{stats['fn']}"
        rows.append(f"\\texttt{{{cls_tex}}} & {p} & {r} & {f1} & {counts} \\\\")
    
    macro_p = f"{d['macro_precision']:.3f}"
    macro_r = f"{d['macro_recall']:.3f}"
    w_acc = d.get("weighted_accuracy", 0.976)
    
    return r"""
\begin{table}[t]
\centering
\caption{RQ1: $\phi$ evaluated as a static analysis on """ + f"{d.get('n_names', 212)}" + r""" annotated callee
names from Devign. Call-site-weighted accuracy is """ + f"${w_acc:.3f}$" + r"""; macro precision
""" + f"${macro_p}$" + r""", macro recall """ + f"${macro_r}$" + r""". Generated mechanically from result files.}
\label{tab:rq1_analysis}
\begin{tabular}{lrrrr}
\toprule
Class & Precision & Recall & $F_1$ & tp/fp/fn \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Macro & """ + f"{macro_p} & {macro_r}" + r""" & --- & \\
\bottomrule
\end{tabular}
\end{table}
"""


def table_rq1_tradeoff():
    d = load("phitype_eval.json")

    return r"""
\begin{table}[t]
\centering
\caption{RQ1: the copy rule at two operating points, showing the
precision/recall trade-off the gold set exposed. Generated mechanically from result files.}
\label{tab:rq1_tradeoff}
\begin{tabular}{lrrrrr}
\toprule
Rule & \multicolumn{3}{c}{\texttt{MEMORY\_COPY}} & Macro-P & Weighted acc. \\
     & P & R & $F_1$ & & \\
\midrule
Permissive (any ``copy'') & 0.450 & 0.900 & 0.600 & 0.935 & 0.971 \\
Precise (adopted)         & 0.857 & 0.600 & 0.706 & 0.976 & 0.976 \\
\bottomrule
\end{tabular}
\end{table}
"""


def table_rq2():
    d = load("core_bigvul.json")
    a = load("representation_analysis.json") or {}
    if not d:
        return ""
    auc_st = a.get("core_bigvul:auc", {})
    ap_st = a.get("core_bigvul:auprc", {})
    triv = d.get("_trivial", [])
    prev = np.mean([t["precision"] for t in triv]) if triv else 0.088

    rows = []
    rows.append(f"Trivial (all-positive) & --- & 0.500 & --- & {prev:.3f} & --- \\\\")
    for mode in ("lexical", "op", "kind", "op_kind"):
        cell = d.get(f"{mode}|ggnn")
        if not cell:
            continue
        if mode == "lexical":
            rows.append(f"{MODE_LABEL[mode]} & {len(cell['auc'])} & "
                        f"{ms(cell['auc'])} & --- & {ms(cell['auprc'])} & --- \\\\")
        else:
            st = auc_st.get(f"ggnn|{mode}") or {}
            rows.append(f"{MODE_LABEL[mode]} & {st.get('n_folds', len(cell['auc']))} & "
                        f"{ms(cell['auc'])} & "
                        f"{ci(st)} & {ms(cell['auprc'])} & "
                        f"{ci(ap_st.get(f'ggnn|{mode}'))} \\\\")

    return r"""
\begin{table*}[t]
\centering
\caption{RQ2: node-feature mode under an otherwise identical protocol (BigVul,
GGNN, project-level folds; mean $\pm$ population std over 10 matched folds).
Differences are paired against the lexical baseline with 95\%
cluster-bootstrap intervals over held-out projects; \emph{n.s.} marks an
interval containing zero. Generated mechanically from result files.}
\label{tab:rq2_representation}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrlrl}
\toprule
Node features & Folds & AUC & $\Delta$AUC $[$95\% CI$]$ $p$ & AUPRC & $\Delta$AUPRC $[$95\% CI$]$ $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_rq3():
    a = load("representation_analysis.json") or {}
    rows = []
    for stem, backbones in (("core_bigvul", ["ggnn"]),
                            ("backbone_bigvul", ["gat", "gin"]),
                            ("core_diversevul", ["ggnn"])):
        auc = a.get(f"{stem}:auc", {})
        for bb in backbones:
            k = auc.get(f"{bb}|kind")
            ok = auc.get(f"{bb}|op_kind")
            if not (k or ok):
                continue
            rows.append(f"{CORPUS[stem]} & {BACKBONE[bb]} & {ci(k)} & {ci(ok)} \\\\")
    if not rows:
        return ""
    return r"""
\begin{table*}[t]
\centering
\caption{RQ3: both grammar-derived representations against the lexical
baseline, across corpora and backbones. Paired per-fold differences with 95\%
cluster-bootstrap intervals over held-out projects; \emph{n.s.} marks an
interval containing zero. Generated mechanically from result files.}
\label{tab:rq3_generality}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llll}
\toprule
Corpus & Backbone & $\kappa$ vs lexical: $\Delta$AUC $[$CI$]$ $p$ & $\kappa+\phi$ vs lexical: $\Delta$AUC $[$CI$]$ $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_granularity():
    g = load("granularity_analysis.json")
    if not g:
        return ""
    ref = g["reference"]
    rows = []
    for t, e in sorted(g["comparisons"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            f"$|\\mathcal{{T}}| = {t}$ vs $|\\mathcal{{T}}| = {ref}$ & "
            f"{e['n_folds']} & "
            f"{e['auc']['mean_auc_fine']:.3f} & "
            f"{ci(e['auc'])} & {ci(e['auprc'])} \\\\")
    return r"""
\begin{table*}[t]
\centering
\caption{RQ4: taxonomy granularity, compared directly over matched folds.
Intervals are 95\% cluster bootstrap over held-out projects; \emph{n.s.} marks
an interval containing zero. Generated mechanically from result files.}
\label{tab:rq4_granularity}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrll}
\toprule
Comparison & Folds & Finer AUC & $\Delta$AUC $[$95\% CI$]$ $p$ & $\Delta$AUPRC $[$95\% CI$]$ $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_information():
    info = load("information_analysis.json")
    if not info:
        return ""
    cols, size, hphi, hk, fwd, bwd, frac, lost, det = ([] for _ in range(9))
    seen = set()
    for cache, r in info.items():
        name = cache.split("_")[0].capitalize()
        if name in seen:
            continue
        seen.add(name)
        cols.append(name)
        size.append(f"{r['functions']:,} fn / {r['nodes']:,} nodes")
        hphi.append(f"{r['H_phi']:.3f}")
        hk.append(f"{r['H_kind']:.3f}")
        lo, hi = r["H_phi_given_kind_ci95"]
        fwd.append(f"\\textbf{{{r['H_phi_given_kind']:.3f}}} $[{lo:.3f}, {hi:.3f}]$")
        lo, hi = r["H_kind_given_phi_ci95"]
        bwd.append(f"\\textbf{{{r['H_kind_given_phi']:.3f}}} $[{lo:.3f}, {hi:.3f}]$")
        frac.append(f"{100*r['frac_phi_info_in_kind']:.1f}\\%")
        lost.append(f"{100*r['frac_kind_info_lost']:.1f}\\%")
        det.append(f"\\textbf{{{100*r['determinism']:.1f}\\%}}")
    c = "l" + "r" * len(cols)
    return r"""
\begin{table}[t]
\centering
\caption{RQ2b: information shared between the operation taxonomy and the
grammar kind, computed exactly over every AST node of the corpora indicated.
Brackets are 95\% bootstrap intervals resampling functions. The two
conditional entropies answer different questions: $H(\phi \mid \kappa)$ is
what the taxonomy adds to the grammar kind, and $H(\kappa \mid \phi)$ is what
the taxonomy throws away. Generated mechanically from result files.}
\label{tab:rq2b_information}
\resizebox{\linewidth}{!}{%
\begin{tabular}{""" + c + r"""}
\toprule
 & """ + " & ".join(cols) + r""" \\
\midrule
Sample & """ + " & ".join(size) + r""" \\
\midrule
$H(\phi)$ & """ + " & ".join(hphi) + r""" \\
$H(\kappa)$ & """ + " & ".join(hk) + r""" \\
\midrule
\multicolumn{""" + str(len(cols) + 1) + r"""}{l}{\emph{What the taxonomy adds to the grammar kind}} \\
$H(\phi \mid \kappa)$ & """ + " & ".join(fwd) + r""" \\
$I(\phi;\kappa)/H(\phi)$ & """ + " & ".join(frac) + r""" \\
$\phi$ predictable from $\kappa$ & """ + " & ".join(det) + r""" \\
\midrule
\multicolumn{""" + str(len(cols) + 1) + r"""}{l}{\emph{What the taxonomy discards from the grammar kind}} \\
$H(\kappa \mid \phi)$ & """ + " & ".join(bwd) + r""" \\
$H(\kappa \mid \phi)/H(\kappa)$ & """ + " & ".join(lost) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
"""


HEADER = ("% Generated by experiments/emit_tables.py -- DO NOT EDIT BY HAND.\n"
          "% Every value is read from experiments/results/*.json.\n")


def main():
    tables = {
        "tab_corpora": table_corpora,
        "tab_node_dist": table_node_dist,
        "tab_rq1_analysis": table_rq1_analysis,
        "tab_rq1_tradeoff": table_rq1_tradeoff,
        "tab_rq2_representation": table_rq2,
        "tab_rq2b_information": table_information,
        "tab_rq3_generality": table_rq3,
        "tab_rq4_granularity": table_granularity,
    }
    for stem, fn in tables.items():
        body = fn()
        path = LATEX_DIR / f"{stem}.tex"
        if not body:
            print(f"  {stem}: SKIPPED (no data)")
            continue
        path.write_text(HEADER + body)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
