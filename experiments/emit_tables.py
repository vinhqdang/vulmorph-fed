"""
Emit the manuscript's result tables mechanically from the result JSONs.

Every number in the paper's tables is produced here. Hand-typing them is what
allowed a stale value to survive a re-run and, in one case, a non-significant
result (p = 0.053, CI crossing zero) to be typeset as significant. Nothing in
this file rounds a confidence bound toward zero, and any cell without backing
data is emitted as "---", never guessed.

Output: manuscript/latex/tables_generated.tex, \input by the experiment
section.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "manuscript" / "latex" / "tables_generated.tex"

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
    """
    Signed fixed-point number for LaTeX. A value that is negative but rounds to
    zero keeps its minus sign: writing a lower bound of -0.0002 as "+0.000" is
    what turned an interval containing zero into an apparently significant one.
    """
    s = f"{abs(x):.{prec}f}"
    return ("-" if x < 0 else "+") + s


def ci(stat, prec=3):
    """
    Point estimate with interval and p-value, entirely inside math mode.

    An interval that excludes zero under only some bootstrap seeds is marked
    borderline rather than significant: whether it excludes zero is then a
    property of the resampling draw, not of the data.
    """
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


def table_rq2():
    d = load("core_bigvul.json")
    a = load("representation_analysis.json") or {}
    if not d:
        return ""
    auc_st = a.get("core_bigvul:auc", {})
    ap_st = a.get("core_bigvul:auprc", {})
    triv = d.get("_trivial", [])
    prev = np.mean([t["precision"] for t in triv]) if triv else float("nan")

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
            # The comparison uses only folds both this mode and the baseline
            # completed, so the paired count is what belongs in the table.
            rows.append(f"{MODE_LABEL[mode]} & {st.get('n_folds', '---')} & "
                        f"{ms(cell['auc'])} & "
                        f"{ci(st)} & {ms(cell['auprc'])} & "
                        f"{ci(ap_st.get(f'ggnn|{mode}'))} \\\\")

    return r"""
\begin{table*}[t]
\centering
\caption{RQ2: node-feature mode under an otherwise identical protocol (BigVul,
GGNN, project-level folds; mean $\pm$ population std over the folds each mode
completed). Differences are paired against the lexical baseline on the folds
both completed --- that paired count is the ``Folds'' column --- with 95\%
cluster-bootstrap intervals over held-out projects; \emph{n.s.} marks an
interval containing zero. Generated mechanically from the result files.}
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
interval containing zero. Generated mechanically from the result files.}
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
    """
    RQ4 as a matched comparison: each finer granularity against the coarsest
    one on the folds both runs completed, where fold i holds out the same
    repositories in both (verified in analyze_granularity.py).
    """
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
\caption{RQ4: taxonomy granularity, compared directly rather than through
separate baselines. Fold $i$ holds out the same repositories in every
granularity run, so each row is a paired comparison in which granularity is
the only quantity that differs. Intervals are 95\% cluster bootstrap over
held-out projects; \emph{n.s.} marks an interval containing zero. The
$|\mathcal{T}|=32$ run completed one fold, so its interval is correspondingly
wide and we draw no conclusion from it. Generated mechanically from the result
files.}
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
the taxonomy throws away. Generated mechanically from the result files.}
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
        "tab_rq2_representation": table_rq2,
        "tab_rq2b_information": table_information,
        "tab_rq3_generality": table_rq3,
        "tab_rq4_granularity": table_granularity,
    }
    for stem, fn in tables.items():
        body = fn()
        path = OUT.parent / f"{stem}.tex"
        if not body:
            print(f"  {stem}: SKIPPED (no data)")
            continue
        path.write_text(HEADER + body)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
