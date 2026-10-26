r"""
Emit the manuscript's result tables and numeric macros from the result JSONs.

Every number in the paper's tables, and every number the prose quotes, comes
from here: tables are written to manuscript/latex/tab_*.tex, and prose numbers
are LaTeX macros in manuscript/latex/numbers.tex (\num...). Nothing in this
file rounds a confidence bound toward zero, and a cell without backing data is
emitted as "---", never guessed.
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
    "anon": "Anonymised token",
    "op": "Operation type $\\phi_8$",
    "kind": "Grammar kind $\\kappa$",
    "op_kind": "Operation $+$ kind ($\\kappa{+}\\phi$)",
    "lex_kind": "Lexical $+$ kind",
    "anon_kind": "Anonymised $+$ kind",
    "kind_low": "Kind via 9-dim bottleneck",
}
BACKBONE = {"ggnn": "GGNN", "gat": "GAT", "gin": "GIN", "tf": "Transformer"}
CORPUS = {"bigvul": "BigVul", "diversevul": "DiverseVul", "devign": "Devign"}
NUMS = {}


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def merged(corpus):
    from experiments.analyze_representation import merge_corpus
    cells, folds, cfgs, _ = merge_corpus(corpus)
    return cells, folds, cfgs


def load_analysis():
    out = {}
    for c in ("bigvul", "diversevul"):
        d = load(f"analysis_{c}.json")
        if d:
            out[c] = d
    return out


def ms(vals, prec=3):
    if not vals:
        return "---"
    return f"{np.mean(vals):.{prec}f} $\\pm$ {np.std(vals):.{prec}f}"


def sf(x, prec=3):
    return ("-" if x < 0 else "+") + f"{abs(x):.{prec}f}"


def pfmt(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def ci(stat, prec=3, wilcoxon=True):
    """Point estimate, interval, bootstrap p and Wilcoxon p."""
    if not stat:
        return "---"
    lo, hi, d = stat["lo"], stat["hi"], stat["diff"]
    stab = stat.get("seed_stability")
    mark = ""
    if not (lo > 0 or hi < 0):
        mark = "\\,n.s."
    elif stab is not None and stab == stab and stab < 0.9:
        mark = "$^{\\dagger}$"
    out = (f"${sf(d, prec)}$ $[{sf(lo, prec)}, {sf(hi, prec)}]$ "
           f"{pfmt(stat['p_two_sided'])}{mark}")
    if wilcoxon:
        out += f" / {pfmt(stat['wilcoxon_p'])}"
    return out


def get(a, corpus, metric, key):
    return (a.get(corpus, {}).get(metric, {}) or {}).get(key)


def macro(name, value):
    NUMS[name] = value


def cellmean(cells, key, metric="auc"):
    c = cells.get(key)
    return float(np.mean(c[metric])) if c else None


# ── corpora ──────────────────────────────────────────────────────────────────

def table_corpora():
    d = load("corpus_stats.json")
    if not d:
        return ""
    rows = []
    for key in ("bigvul", "diversevul", "devign"):
        c = d.get(key)
        if not c:
            continue
        tp = f"{c['test_prev_pooled']:.3f}" if "test_prev_pooled" in c else "---"
        rows.append(
            f"{CORPUS[key]} & {c['n']:,} & {c['vulnerable']:,} & {c['prev']:.3f} & "
            f"{tp} & {c['projects']:,} & {c['top2_share']:.1f}\\% & "
            f"{c['nodes']:.0f} & {100*c['trunc_rate']:.1f} / "
            f"{100*c['trunc_rate_vuln']:.1f} / {100*c['trunc_rate_benign']:.1f} \\\\")
        macro(f"{key}N", f"{c['n']:,}")
        macro(f"{key}Vuln", f"{c['vulnerable']:,}")
        macro(f"{key}Prev", f"{c['prev']:.3f}")
        macro(f"{key}Projects", f"{c['projects']:,}")
        macro(f"{key}TruncPct", f"{100*c['trunc_rate']:.1f}")
        macro(f"{key}TruncVulnPct", f"{100*c['trunc_rate_vuln']:.1f}")
        macro(f"{key}TruncBenignPct", f"{100*c['trunc_rate_benign']:.1f}")
        macro(f"{key}TruncP", pfmt(c["trunc_fisher_p"]))
        macro(f"{key}TopTwoShare", f"{c['top2_share']:.0f}")
        macro(f"{key}Nodes", f"{c['nodes']:.0f}")
        macro(f"{key}Edges", f"{c['edges']:.0f}")
        if "test_prev_pooled" in c:
            macro(f"{key}TestPrev", f"{c['test_prev_pooled']:.3f}")
            macro(f"{key}TrainOnlyN", f"{c['train_only_n']:,}")
            macro(f"{key}TrainOnlyPrev", f"{c['train_only_prev']:.3f}")
            macro(f"{key}TopTestShareMax",
                  f"{100*max(f['top_share'] for f in c['folds']):.0f}")
            macro(f"{key}TopTestProject",
                  max(c['folds'], key=lambda f: f['top_share'])['top_project'])
    return r"""
\begin{table*}[t]
\centering
\caption{Corpus properties measured on our uniform random samples
(10{,}000 functions drawn per corpus, then de-duplicated). \emph{Prevalence} is
the fraction of vulnerable functions in the whole sample; \emph{Test prev.} is
the pooled fraction over the held-out folds. They differ because the largest
repositories, which have a lower share of vulnerable functions, are held
train-only and never appear in a test fold (Section~\ref{sec:protocol}).
\emph{Truncated} is the percentage of functions with more than 200 named AST
nodes, overall / vulnerable / benign.}
\label{tab:corpora}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Corpus & Functions & Vulnerable & Prevalence & Test prev. & Projects & Top-2 share & Nodes (mean) & Truncated (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_node_dist():
    d = load("corpus_stats.json")
    if not d or "_node_dist_bigvul" not in d:
        return ""
    dist = d["_node_dist_bigvul"]
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    mid = (len(items) + 1) // 2
    left, right = items[:mid], items[mid:]

    def nm(n):
        t = n.replace("_", "\\_")
        return f"$\\bot$ (\\texttt{{{t}}})" if n == "UNKNOWN" else f"\\texttt{{{t}}}"
    rows = []
    for i in range(mid):
        l = f"{nm(left[i][0])} & {left[i][1]:.2f}\\%"
        r = (f"{nm(right[i][0])} & {right[i][1]:.2f}\\%" if i < len(right)
             else " & ")
        rows.append(f"{l} & {r} \\\\")
    macro("bvUnknownPct", f"{dist.get('UNKNOWN', 0):.0f}")
    macro("bvMemoryPct", f"{dist.get('MEMORY_ACCESS', 0):.2f}")
    return r"""
\begin{table}[t]
\centering
\caption{Node-type distribution under $\phi_8$ (BigVul sample). $\bot$
(\texttt{UNKNOWN}) nodes carry no operation label but keep their grammar kind.}
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


# ── RQ1 ──────────────────────────────────────────────────────────────────────

def table_rq1():
    d = load("phitype_validation.json")
    if not d:
        return ""
    pc = d["pooled"]["per_class"]
    rows = []
    for cls, v in pc.items():
        t = cls.replace("_", "\\_")

        def f(x, ci_):
            if x is None:
                return "---"
            return f"{x:.2f} $[{ci_[0]:.2f},{ci_[1]:.2f}]$"
        rows.append(f"\\texttt{{{t}}} & {f(v['precision'], v['p_ci'] or (0, 0))} & "
                    f"{f(v['recall'], v['r_ci'] or (0, 0))} & "
                    f"{v['tp']}/{v['fp']}/{v['fn']} \\\\")
    cr = []
    for c in ("devign", "bigvul", "diversevul"):
        v = d["corpora"].get(c)
        if not v:
            continue
        mb = v["memop_binary"]
        cr.append(f"{CORPUS[c]} & {v['n_names']} & {v['agreement']:.2f} & "
                  f"{v['kappa']:.2f} & {v['macro_precision']:.2f} & "
                  f"{v['macro_recall']:.2f} & {mb['precision']:.2f} / "
                  f"{mb['recall']:.2f} \\\\")
        macro(f"{c}Kappa", f"{v['kappa']:.2f}")
        macro(f"{c}Agree", f"{v['agreement']:.2f}")
        macro(f"{c}NNames", str(v["n_names"]))
        macro(f"{c}MacroP", f"{v['macro_precision']:.2f}")
        macro(f"{c}MacroR", f"{v['macro_recall']:.2f}")
        macro(f"{c}Adjudicated", str(v["adjudicated"]))
        macro(f"{c}Unresolved", str(v["unresolved_three_way"]))
        macro(f"{c}MemopP", f"{mb['precision']:.2f}")
        macro(f"{c}MemopR", f"{mb['recall']:.2f}")
    po = d["pooled"]
    macro("pooledKappa", f"{po['kappa']:.2f}")
    macro("pooledAgree", f"{po['agreement']:.2f}")
    macro("pooledNNames", str(po["n_names"]))
    macro("pooledMacroP", f"{po['macro_precision']:.2f}")
    macro("pooledMacroR", f"{po['macro_recall']:.2f}")
    pm = d["pooled_memop"]
    macro("pooledMemopP", f"{pm['precision']:.2f}")
    macro("pooledMemopR", f"{pm['recall']:.2f}")
    macro("devignOrigKappa", f"{d['corpora']['devign'].get('kappa_original_vs_consensus', float('nan')):.2f}")
    mincls = min((v for v in pc.values() if v["precision"] is not None),
                 key=lambda v: v["tp"] + v["fn"])
    macro("minClassPositives", str(mincls["tp"] + mincls["fn"]))
    cp = pc.get("MEMORY_COPY")
    if cp:
        macro("copyPrecision", f"{cp['precision']:.2f}")
    return r"""
\begin{table}[t]
\centering
\caption{RQ1: $\phi$ as a static analysis, scored against the consensus of two
blind annotators and a third pass on disagreements, pooled over the three
corpora. Brackets are 95\% Wilson intervals; tp/fp/fn are counts of distinct
callee names. The last row scores the question $\phi_8$ actually answers
(memory or string operation vs. not).}
\label{tab:rq1_analysis}
\begin{tabular}{lrrr}
\toprule
Class & Precision & Recall & tp/fp/fn \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Macro & """ + f"{po['macro_precision']:.2f} & {po['macro_recall']:.2f}" + r""" & \\
Memory operation (any of the 9) & """ + f"{pm['precision']:.2f} & {pm['recall']:.2f}" + r""" & """ + f"{pm['tp']}/{pm['fp']}/{pm['fn']}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{RQ1: annotation agreement and per-corpus scores. Agreement and
Cohen's $\kappa_{C}$ are between the two blind annotators before adjudication;
precision and recall are macro-averaged over classes against the consensus; the
last column is precision / recall for ``memory or string operation''.}
\label{tab:rq1_agreement}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Corpus & Names & Agreement & $\kappa_{C}$ & Macro P & Macro R & Memory-op P / R \\
\midrule
""" + "\n".join(cr) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


# ── RQ2 / RQ3 ────────────────────────────────────────────────────────────────

def ssm(cell):
    v = [x for x in cell.get("auc_ss", []) if x == x]
    return f"{np.mean(v):.3f}" if v else "---"


def table_rq2(a):
    corpus = "bigvul"
    cells, _, _ = merged(corpus)
    if not cells:
        return ""
    rows = []
    prevs = (a.get(corpus) or {}).get("_trivial_prevalence") or []
    rows.append(f"Trivial (all-positive) & --- & 0.500 & --- & 0.500 & "
                f"{np.mean(prevs):.3f} \\\\" if prevs else "")
    sz = load("bigvul_size.json")
    if sz:
        c = sz["size|rule"]
        rows.append(f"Function size only (rule) & {len(c['auc'])} & {ms(c['auc'])} "
                    f"& --- & 0.500 & {ms(c['auprc'])} \\\\")
        macro("bvSizeAuc", f"{np.mean(c['auc']):.3f}")
        macro("bvSizeAuprc", f"{np.mean(c['auprc']):.3f}")
    szd = load("diversevul_size.json")
    if szd:
        macro("dvSizeAuc", f"{np.mean(szd['size|rule']['auc']):.3f}")
    for mode in ("lexical", "anon", "op", "kind", "op_kind", "lex_kind",
                 "anon_kind"):
        cell = cells.get(f"{mode}|ggnn")
        if not cell:
            continue
        st = get(a, corpus, "auc", f"ggnn|{mode}~lexical")
        sp = get(a, corpus, "auprc", f"ggnn|{mode}~lexical")
        if mode == "lexical":
            rows.append(f"{MODE_LABEL[mode]} & {len(cell['auc'])} & {ms(cell['auc'])} "
                        f"& --- & {ssm(cell)} & {ms(cell['auprc'])} \\\\")
        else:
            rows.append(f"{MODE_LABEL[mode]} & {len(cell['auc'])} & {ms(cell['auc'])} "
                        f"& {ci(st)} & {ssm(cell)} & {ms(cell['auprc'])} \\\\")
            sst = get(a, corpus, "auc_ss", f"ggnn|{mode}~lexical")
            if sst:
                mm = {"kind": "Kind", "op": "Op", "op_kind": "OpKind", "lex_kind": "LexKind",
                      "anon": "Anon", "anon_kind": "AnonKind"}[mode]
                macro(f"bvGgnn{mm}SsDiff", sf(sst["diff"]))
                macro(f"bvGgnn{mm}SsLo", sf(sst["lo"]))
                macro(f"bvGgnn{mm}SsHi", sf(sst["hi"]))
        if st:
            m = {"kind": "Kind", "op": "Op", "op_kind": "OpKind", "lex_kind": "LexKind",
                 "anon": "Anon", "anon_kind": "AnonKind"}[mode]
            macro(f"bvGgnn{m}AucDiff", sf(st["diff"]))
            macro(f"bvGgnn{m}AucLo", sf(st["lo"]))
            macro(f"bvGgnn{m}AucHi", sf(st["hi"]))
            macro(f"bvGgnn{m}AucP", pfmt(st["p_two_sided"]))
            macro(f"bvGgnn{m}AucWilcoxon", pfmt(st["wilcoxon_p"]))
            macro(f"bvGgnn{m}AucWins", f"{st['wins']}/{st['n_folds']}")
            if "without_android" in st:
                w = st["without_android"]
                macro(f"bvGgnn{m}AucNoAndroid",
                      f"{sf(w['diff'])} [{sf(w['lo'])}, {sf(w['hi'])}]")
        if sp:
            m = {"kind": "Kind", "op": "Op", "op_kind": "OpKind", "lex_kind": "LexKind",
                 "anon": "Anon", "anon_kind": "AnonKind"}.get(mode)
            if m:
                macro(f"bvGgnn{m}AuprcDiff", sf(sp["diff"]))
                macro(f"bvGgnn{m}AuprcLo", sf(sp["lo"]))
                macro(f"bvGgnn{m}AuprcHi", sf(sp["hi"]))
        macro(f"bvGgnn{mode.replace('_','').title().replace('Op','Op')}AucMean",
              f"{np.mean(cell['auc']):.3f}")
        macro(f"bvGgnn{mode.replace('_','').title()}AucStd", f"{np.std(cell['auc']):.3f}")
    return r"""
\begin{table*}[t]
\centering
\caption{RQ2: node views on BigVul, GGNN, 10 matched project-level folds
(mean $\pm$ population std over folds). Differences are paired against the
lexical baseline: each $\Delta$ cell reads mean per-fold difference, 95\%
cluster-bootstrap interval over held-out projects, bootstrap $p$ / Wilcoxon
signed-rank $p$ over folds; \emph{n.s.} marks an interval containing zero.
\emph{AUC in size bins} is the AUC computed within five bins of function size
(0.5 = no information beyond length). AUPRC is reported without paired tests.}
\label{tab:rq2_representation}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrlrr}
\toprule
Node view & Folds & AUC & $\Delta$AUC [95\% CI] $p_{\text{boot}}$ / $p_{\text{W}}$ & AUC in size bins & AUPRC \\
\midrule
""" + "\n".join(r for r in rows if r) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_direct(a):
    pairs = [("op_kind", "kind", "$\\kappa{+}\\phi$ vs $\\kappa$"),
             ("kind", "lex_kind", "$\\kappa$ vs lexical$+\\kappa$"),
             ("kind", "anon_kind", "$\\kappa$ vs anonymised$+\\kappa$"),
             ("anon_kind", "lex_kind", "anonymised$+\\kappa$ vs lexical$+\\kappa$"),
             ("lex_kind", "lexical", "lexical$+\\kappa$ vs lexical"),
             ("anon", "lexical", "anonymised vs lexical"),
             ("kind", "op", "$\\kappa$ vs $\\phi_8$"),
             ("kind_low", "kind", "9-dim bottleneck $\\kappa$ vs $\\kappa$")]
    rows = []
    for ka, kb, label in pairs:
        cs = []
        for corpus in ("bigvul", "diversevul"):
            s = get(a, corpus, "auc", f"ggnn|{ka}~{kb}")
            cs.append(ci(s) if s else "---")
            if s:
                tag = {"op_kind~kind": "OpKindVsKind", "kind~lex_kind": "KindVsLexKind",
                       "kind~anon_kind": "KindVsAnonKind",
                       "anon_kind~lex_kind": "AnonKindVsLexKind",
                       "lex_kind~lexical": "LexKindVsLex", "anon~lexical": "AnonVsLex",
                       "kind~op": "KindVsOp", "kind_low~kind": "LowVsKind"}[f"{ka}~{kb}"]
                pre = "bv" if corpus == "bigvul" else "dv"
                macro(f"{pre}{tag}Diff", sf(s["diff"]))
                macro(f"{pre}{tag}Lo", sf(s["lo"]))
                macro(f"{pre}{tag}Hi", sf(s["hi"]))
                macro(f"{pre}{tag}P", pfmt(s["p_two_sided"]))
                macro(f"{pre}{tag}Wilcoxon", pfmt(s["wilcoxon_p"]))
        rows.append(f"{label} & {cs[0]} & {cs[1]} \\\\")
    if not any("---" != r for r in rows):
        return ""
    return r"""
\begin{table*}[t]
\centering
\caption{Direct paired contrasts between node views (GGNN, 10 matched folds):
$\Delta$AUC of the first view minus the second, with 95\% cluster-bootstrap
interval, bootstrap $p$ / Wilcoxon $p$. These are the comparisons a reader
needs to judge whether one view is better than another; a difference between
two views' gaps to the lexical baseline is not a test of the difference.}
\label{tab:direct}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lll}
\toprule
Contrast & BigVul & DiverseVul \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_generality(a):
    rows = []
    for corpus in ("bigvul", "diversevul"):
        cells, _, _ = merged(corpus)
        for bb in ("ggnn", "gat", "gin", "tf"):
            if f"lexical|{bb}" not in cells:
                continue
            vals = []
            for mode in ("lexical", "lex_kind", "kind", "op_kind"):
                m = cellmean(cells, f"{mode}|{bb}")
                vals.append(f"{m:.3f}" if m is not None else "---")
            k = get(a, corpus, "auc", f"{bb}|kind~lexical")
            lk = get(a, corpus, "auc", f"{bb}|lex_kind~lexical")
            rows.append(f"{CORPUS[corpus]} & {BACKBONE[bb]} & " + " & ".join(vals) +
                        f" & {ci(k, wilcoxon=False) if k else '---'} "
                        f"& {ci(lk, wilcoxon=False) if lk else '---'} \\\\")
            pre = "bv" if corpus == "bigvul" else "dv"
            if k:
                macro(f"{pre}{bb.title()}KindDiff", sf(k["diff"]))
                macro(f"{pre}{bb.title()}KindLo", sf(k["lo"]))
                macro(f"{pre}{bb.title()}KindHi", sf(k["hi"]))
    if not rows:
        return ""
    return r"""
\begin{table*}[t]
\centering
\caption{RQ3: mean AUC over 10 matched folds for four node views under every
backbone and corpus, and the paired gain of $\kappa$ and of lexical$+\kappa$
over the lexical view (point estimate, 95\% cluster-bootstrap interval,
bootstrap $p$; \emph{n.s.}: interval contains zero).}
\label{tab:rq3_generality}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrll}
\toprule
Corpus & Backbone & Lexical & Lex$+\kappa$ & $\kappa$ & $\kappa{+}\phi$ & $\kappa$ vs lexical & Lex$+\kappa$ vs lexical \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_ablation(a):
    corpus = "bigvul"
    cells, _, _ = merged(corpus)
    rows = []
    base = cellmean(cells, "kind|ggnn")
    if base is None:
        return ""
    rows.append(f"All edges (AST+SIB+DU), $\\kappa$ & {len(cells['kind|ggnn']['auc'])} & "
                f"{base:.3f} & --- \\\\")
    for v, label in (("ast_sib", "no def-use edges"), ("ast_du", "no sibling edges"),
                     ("ast", "AST edges only")):
        k = f"kind|ggnn@{v}"
        s = get(a, corpus, "auc", f"ggnn|kind@{v}~kind")
        if k in cells:
            rows.append(f"\\quad {label} & {len(cells[k]['auc'])} & "
                        f"{cellmean(cells, k):.3f} & {ci(s) if s else '---'} \\\\")
            if s:
                tag = {"ast_sib": "NoDu", "ast_du": "NoSib", "ast": "AstOnly"}[v]
                macro(f"edge{tag}Diff", sf(s["diff"]))
                macro(f"edge{tag}Lo", sf(s["lo"]))
                macro(f"edge{tag}Hi", sf(s["hi"]))
    for t in (16, 32):
        k = f"op_kind|ggnn@t{t}"
        s = get(a, corpus, "auc", f"ggnn|op_kind@t{t}~op_kind")
        if k in cells:
            rows.append(f"$\\kappa{{+}}\\phi$ with $|\\mathcal{{T}}|={t}$ vs $8$ & "
                        f"{len(cells[k]['auc'])} & {cellmean(cells, k):.3f} & "
                        f"{ci(s) if s else '---'} \\\\")
            if s:
                macro(f"granT{t}Diff", sf(s["diff"]))
                macro(f"granT{t}Lo", sf(s["lo"]))
                macro(f"granT{t}Hi", sf(s["hi"]))
                macro(f"granT{t}Folds", str(s["n_folds"]))
    return r"""
\begin{table}[t]
\centering
\caption{Ablations on BigVul (GGNN). First block: removing edge types from the
graph under the $\kappa$ view. Second block: refining the operation alphabet
under the $\kappa{+}\phi$ view, compared against the same view at
$|\mathcal{T}|=8$ on the same folds. $\Delta$AUC is paired, with 95\%
cluster-bootstrap interval and bootstrap $p$.}
\label{tab:ablation}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrl}
\toprule
Variant & Folds & AUC & $\Delta$AUC vs reference [95\% CI] $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def table_baselines(a):
    out = []
    for corpus in ("bigvul", "diversevul"):
        cells, _, _ = merged(corpus)
        pre = load(f"{corpus}_unixcoder.json")
        if pre and "unixcoder|probe" in pre:
            cells["unixcoder|probe"] = pre["unixcoder|probe"]
        out.append((corpus, cells))
    spec = [
        ("lexical|ggnn", "GGNN on code text (token-only baseline)"),
        ("lex_kind|ggnn", "Devign/ReVeal-style GGNN: text $+$ node type"),
        ("anon_kind|ggnn", "GGNN, anonymised tokens $+$ node type"),
        ("lexical|tf", "Transformer over node sequence, text"),
        ("kind|tf", "Transformer over node sequence, $\\kappa$"),
        ("unixcoder|probe", "Pretrained UniXcoder, frozen, linear probe (raw source)"),
        ("unixnorm|probe", "Pretrained UniXcoder, frozen, linear probe (format-normalised)"),
        ("linevul|ft", "LineVul-style: fine-tuned CodeBERT (raw source)"),
        ("linevulnorm|ft", "LineVul-style: fine-tuned CodeBERT (format-normalised)"),
        ("size|rule", "No code read: function size (node count)"),
        ("trail|rule", "No code read: trailing-whitespace rule"),
        ("format|gbm", "No code read: gradient boosting on formatting statistics"),
        ("kind|ggnn", "GGNN, grammar kind $\\kappa$ (ours)"),
        ("op_kind|ggnn", "GGNN, $\\kappa{+}\\phi$ (ours)"),
    ]
    rows = []
    for key, label in spec:
        vals = []
        for corpus, cells in out:
            c = cells.get(key)
            vals.append(f"{ms(c['auc'])} & {ms(c['auprc'])}" if c else "--- & ---")
        if all(v == "--- & ---" for v in vals):
            continue
        rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
        if key == "unixcoder|probe":
            for corpus, cells in out:
                c = cells.get(key)
                if c:
                    macro(("bv" if corpus == "bigvul" else "dv") + "UnixAuc",
                          f"{np.mean(c['auc']):.3f}")
    return r"""
\begin{table*}[t]
\centering
\caption{Comparison with existing approaches re-implemented under the same
sample, folds and metrics (mean $\pm$ std over 10 project-level folds). These
are re-implementations of the \emph{input representation} of each approach
inside one training protocol, not reproductions of the published systems or
their published numbers. UniXcoder rows use a frozen encoder and a linear
probe. \emph{Format-normalised} source is the parser's token stream with
whitespace and comments removed (Section~\ref{sec:artefact}). The last three
rows score functions without reading what the code says.}
\label{tab:baselines}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{BigVul} & \multicolumn{2}{c}{DiverseVul} \\
System & AUC & AUPRC & AUC & AUPRC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


def table_perproject(a):
    r = (a.get("bigvul") or {}).get("_per_project")
    if not r:
        return ""
    rows = []
    for x in r[:8]:
        g = lambda k: f"{x[k]:.2f}" if k in x else "---"
        rows.append(f"{x['project'].split(':')[-1]} & {x['n']:,} & {x['n_vuln']} & "
                    f"{g('lexical')} & {g('lex_kind')} & {g('kind')} & {g('op_kind')} \\\\")
    return r"""
\begin{table}[t]
\centering
\caption{Per-project AUC (BigVul, GGNN) for the held-out projects with at
least 10 vulnerable functions, pooled over the folds in which they are tested.
It shows that the pooled result is not carried by a single repository.}
\label{tab:perproject}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Project & Functions & Vuln. & Lexical & Lex$+\kappa$ & $\kappa$ & $\kappa{+}\phi$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def table_hparams():
    cfg = None
    for p in sorted(RESULTS.glob("bigvul_core_f*.json")):
        cfg = json.load(open(p)).get("_config")
        break
    if not cfg:
        return ""
    macro("cfgN", f"{cfg['n']:,}")
    macro("cfgEpochs", str(cfg["epochs"]))
    macro("cfgDedupe", f"{cfg.get('n_functions_after_dedupe', 0):,}")
    return r"""
\begin{table}[t]
\centering
\caption{Hyper-parameters. All values are fixed in advance and shared by every
compared system; none is tuned on any fold.}
\label{tab:hparams}
\begin{tabular}{lp{0.62\linewidth}}
\toprule
Item & Value \\
\midrule
Embedding width / hidden width & """ + f"{cfg['embed_dim']} / {cfg['hidden_dim']}" + r""" \\
Message-passing layers & """ + f"{cfg['num_layers']}" + r""" (GGNN: 2 gated steps) \\
Transformer & 2 layers, 4 heads, feed-forward $2\times$ width, dropout 0.1 \\
Readout and head & mean $\Vert$ max pooling; MLP $2h\to h/2\to 1$, dropout 0.3 \\
Optimiser & Adam, learning rate """ + f"{cfg['lr']}" + r""" \\
Batch size & """ + f"{cfg['batch_size']}" + r""" \\
Epochs & """ + f"{cfg['epochs']}" + r""" (fixed; no early stopping, no validation split) \\
Loss & binary cross-entropy, positive weight $=\min(n_{\text{neg}}/n_{\text{pos}}, 20)$ \\
Seeds & initialisation and shuffling seed $1000+k$ for fold $k$, shared by all systems \\
Lexical dictionary & up to 9{,}999 most frequent training-fold tokens with count $\ge 2$, plus \texttt{<UNK>} \\
Node cap & 200 named AST nodes (pre-order) \\
Sample & """ + f"{cfg['n']:,}" + r""" functions per corpus, uniform at random, seed """ + f"{cfg['sample_seed']}" + r""" \\
Folds & """ + f"{cfg['n_splits']}$\\times${cfg['n_repeats']}" + r""" = 10 project-level folds \\
Pretrained reference & UniXcoder-base, 256 tokens, frozen; logistic regression $C=0.1$, balanced \\
\bottomrule
\end{tabular}
\end{table}
"""


def table_information():
    info = load("information_analysis.json")
    if not info:
        return ""
    cols, size, hphi, hk, fwd, bwd, frac, lost, det = ([] for _ in range(9))
    for cache, r in sorted(info.items()):
        name = CORPUS.get(cache.split("_")[0], cache.split("_")[0])
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
        k = cache.split("_")[0]
        macro(f"{k}HPhiGivenKind", f"{r['H_phi_given_kind']:.2f}")
        macro(f"{k}HKindGivenPhi", f"{r['H_kind_given_phi']:.2f}")
        macro(f"{k}Determinism", f"{100*r['determinism']:.1f}")
        macro(f"{k}KindLost", f"{100*r['frac_kind_info_lost']:.0f}")
        macro(f"{k}PhiInKind", f"{100*r['frac_phi_info_in_kind']:.0f}")
    dets = [100 * info[c]["determinism"] for c in info]
    macro("determinismMin", f"{min(dets):.1f}")
    macro("determinismMax", f"{max(dets):.1f}")
    widths = [info[c]["H_kind_given_phi_ci95"][1] - info[c]["H_kind_given_phi_ci95"][0] for c in info]
    macro("kindGivenPhiCiWidthMax", f"{max(widths):.2f}")
    hs = [info[c]["H_phi_given_kind"] for c in info]
    macro("hPhiGivenKindMax", f"{max(hs):.2f}")
    macro("hPhiGivenKindMin", f"{min(hs):.2f}")
    c = "l" + "r" * len(cols)
    n = str(len(cols) + 1)
    return r"""
\begin{table}[t]
\centering
\caption{RQ2b: information shared between the operation taxonomy and the
grammar kind, computed exactly over every AST node of the corpora indicated.
Brackets are 95\% bootstrap intervals resampling functions.
$H(\phi \mid \kappa)$ is what the taxonomy adds to the grammar kind, and
$H(\kappa \mid \phi)$ is what the taxonomy throws away.}
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
\multicolumn{""" + n + r"""}{l}{\emph{What the taxonomy adds to the grammar kind}} \\
$H(\phi \mid \kappa)$ & """ + " & ".join(fwd) + r""" \\
$I(\phi;\kappa)/H(\phi)$ & """ + " & ".join(frac) + r""" \\
$\phi$ predictable from $\kappa$ & """ + " & ".join(det) + r""" \\
\midrule
\multicolumn{""" + n + r"""}{l}{\emph{What the taxonomy discards from the grammar kind}} \\
$H(\kappa \mid \phi)$ & """ + " & ".join(bwd) + r""" \\
$H(\kappa \mid \phi)/H(\kappa)$ & """ + " & ".join(lost) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
"""


CAMEL = {"lexical": "Lex", "anon": "Anon", "op": "Op", "kind": "Kind",
         "op_kind": "OpKind", "lex_kind": "LexKind", "anon_kind": "AnonKind",
         "kind_low": "KindLow"}


def generic_macros(a):
    """m<Corpus><Backbone><Mode><Stat>: mean AUC and paired contrasts for every
    cell, so prose can quote any of them without a hand-typed number."""
    for corpus, C in (("bigvul", "Bv"), ("diversevul", "Dv")):
        cells, _, _ = merged(corpus)
        for key, cell in cells.items():
            mode, bb = key.split("|")
            mode = mode.split("@")[0]
            if mode not in CAMEL:
                continue
            tag = f"m{C}{bb.title()}{CAMEL[mode]}"
            macro(f"{tag}Auc", f"{np.mean(cell['auc']):.3f}")
            macro(f"{tag}Auprc", f"{np.mean(cell['auprc']):.3f}")
            v = [x for x in cell.get("auc_ss", []) if x == x]
            if v:
                macro(f"{tag}Ss", f"{np.mean(v):.3f}")
        for mkey, st in (a.get(corpus, {}).get("auc", {}) or {}).items():
            if "~" not in mkey or "@" in mkey:
                continue
            bb, rest = mkey.split("|")
            lhs, rhs = rest.split("~")
            names = {**CAMEL, "size": "Size", "unixcoder": "Unix", "trail": "Trail",
                     "format": "Format", "linevul": "Linevul",
                     "linevulnorm": "LinevulNorm", "unixnorm": "UnixNorm"}
            if lhs not in CAMEL or rhs not in names:
                continue
            tag = f"c{C}{bb.title()}{CAMEL[lhs]}V{names[rhs]}"
            macro(f"{tag}D", sf(st["diff"]))
            macro(f"{tag}Lo", sf(st["lo"]))
            macro(f"{tag}Hi", sf(st["hi"]))
            macro(f"{tag}P", pfmt(st["p_two_sided"]))
            macro(f"{tag}W", pfmt(st["wilcoxon_p"]))
            macro(f"{tag}H", pfmt(st["p_holm"]))
            macro(f"{tag}WH", pfmt(st["wilcoxon_p_holm"]))
            macro(f"{tag}Wblk", pfmt(st["wilcoxon_p_blocks"]))
            macro(f"{tag}Wins", f"{st['wins']}/{st['n_folds']}")
            if "without_android" in st:
                w = st["without_android"]
                macro(f"{tag}NoAnd", f"{sf(w['diff'])} [{sf(w['lo'])}, {sf(w['hi'])}]")
        for mkey, st in (a.get(corpus, {}).get("auc", {}) or {}).items():
            if not mkey.startswith("ref|"):
                continue
            lhs, rhs = mkey[4:].split("~")
            nm = {"linevul": "Linevul", "linevulnorm": "LinevulNorm", "unixcoder": "Unix",
                  "unixnorm": "UnixNorm", "format": "Format", "trail": "Trail", "size": "Size"}
            tag = f"x{C}{nm[lhs]}V{nm[rhs]}"
            macro(f"{tag}D", sf(st["diff"]))
            macro(f"{tag}Lo", sf(st["lo"]))
            macro(f"{tag}Hi", sf(st["hi"]))
            macro(f"{tag}P", pfmt(st["p_two_sided"]))
            macro(f"{tag}W", pfmt(st["wilcoxon_p"]))
            macro(f"{tag}Wins", f"{st['wins']}/{st['n_folds']}")
        for mkey, st in (a.get(corpus, {}).get("auc_ss", {}) or {}).items():
            if "~" not in mkey or "@" in mkey:
                continue
            bb, rest = mkey.split("|")
            lhs, rhs = rest.split("~")
            if lhs in CAMEL and rhs in CAMEL:
                tag = f"c{C}{bb.title()}{CAMEL[lhs]}V{CAMEL[rhs]}"
                macro(f"{tag}SsD", sf(st["diff"]))
                macro(f"{tag}SsLo", sf(st["lo"]))
                macro(f"{tag}SsHi", sf(st["hi"]))


REFS = {"size|rule": "Size", "trail|rule": "Trail", "format|gbm": "Format",
        "linevul|ft": "Linevul", "linevulnorm|ft": "LinevulNorm",
        "unixcoder|probe": "Unix", "unixnorm|probe": "UnixNorm"}


def reference_macros():
    import gzip
    for corpus, C in (("bigvul", "Bv"), ("diversevul", "Dv")):
        cells, _, _ = merged(corpus)
        for key, tag in REFS.items():
            c = cells.get(key)
            if c:
                macro(f"r{C}{tag}Auc", f"{np.mean(c['auc']):.3f}")
                macro(f"r{C}{tag}Auprc", f"{np.mean(c['auprc']):.3f}")
                macro(f"r{C}{tag}Folds", str(len(c["auc"])))
        p = Path(__file__).parent.parent / "colab_data" / f"{corpus}.json.gz"
        if p.exists():
            d = json.load(gzip.open(p, "rt"))
            y = np.asarray(d["y"])
            has = np.array([any(l != l.rstrip() for l in s_.split("\n"))
                            for s_ in d["codes"]])
            macro(f"art{C}TrailVuln", f"{100*has[y == 1].mean():.0f}")
            macro(f"art{C}TrailBenign", f"{100*has[y == 0].mean():.0f}")
            hasn = np.array([any(l != l.rstrip() for l in s_.split("\n"))
                             for s_ in d["codes_norm"]])
            macro(f"art{C}TrailNormAny", f"{100*hasn.mean():.0f}")


def range_macros(a):
    """Ranges quoted in the abstract, introduction and conclusion."""
    deltas, lex = [], []
    for corpus in ("bigvul", "diversevul"):
        cells, _, _ = merged(corpus)
        for key, c in cells.items():
            if key.startswith("lexical|") and "@" not in key:
                lex.append(float(np.mean(c["auc"])))
        for mkey, st in (a.get(corpus, {}).get("auc", {}) or {}).items():
            bb, rest = mkey.split("|")
            if "@" in rest or not rest.endswith("~lexical"):
                continue
            if rest.split("~")[0] in ("kind", "op_kind", "anon", "anon_kind"):
                deltas.append(st["diff"])
    if deltas:
        macro("deltaMin", f"{min(deltas):.2f}")
        macro("deltaMax", f"{max(deltas):.2f}")
    ss = []
    for corpus in ("bigvul", "diversevul"):
        cells, _, _ = merged(corpus)
        for key, c in cells.items():
            if "@" in key or key.split("|")[1] in ("rule", "probe"):
                continue
            v = [x for x in c.get("auc_ss", []) if x == x]
            if v:
                ss.append(float(np.mean(v)))
    if ss:
        macro("ssMin", f"{min(ss):.2f}")
        macro("ssMax", f"{max(ss):.2f}")
    if lex:
        macro("lexAucMin", f"{min(lex):.2f}")
        macro("lexAucMax", f"{max(lex):.2f}")


def architecture_macros(a):
    for corpus, pre in (("bigvul", "bv"), ("diversevul", "dv")):
        sp = (a.get(corpus) or {}).get("_architecture_spread") or {}
        for mode, tag in (("lexical", "Lex"), ("kind", "Kind"), ("lex_kind", "LexKind"),
                          ("op_kind", "OpKind")):
            if mode in sp:
                macro(f"{pre}Spread{tag}", f"{sp[mode]['range']:.3f}")


HEADER = ("% Generated by experiments/emit_tables.py -- DO NOT EDIT BY HAND.\n"
          "% Every value is read from experiments/results/*.json.\n")


def main():
    a = load_analysis()
    architecture_macros(a)
    generic_macros(a)
    range_macros(a)
    reference_macros()
    tables = {
        "tab_corpora": table_corpora,
        "tab_node_dist": table_node_dist,
        "tab_rq1_analysis": table_rq1,
        "tab_rq2b_information": table_information,
        "tab_hparams": table_hparams,
        "tab_rq2_representation": lambda: table_rq2(a),
        "tab_direct": lambda: table_direct(a),
        "tab_rq3_generality": lambda: table_generality(a),
        "tab_ablation": lambda: table_ablation(a),
        "tab_baselines": lambda: table_baselines(a),
        "tab_perproject": lambda: table_perproject(a),
    }
    for stem, fn in tables.items():
        try:
            body = fn()
        except Exception as e:
            print(f"  {stem}: FAILED ({type(e).__name__}: {e})")
            continue
        path = LATEX_DIR / f"{stem}.tex"
        if not body:
            print(f"  {stem}: SKIPPED (no data)")
            continue
        path.write_text(HEADER + body)
        print(f"  wrote {path.name}")
    lines = [HEADER.strip()]
    for k, v in sorted(NUMS.items()):
        digits = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four",
                                         "Five", "Six", "Seven", "Eight", "Nine"]))
        base = "".join(digits.get(ch, ch) for ch in k if ch.isalnum())
        # generic per-cell macros (m/c + Bv|Dv ...) are used without prefix
        name = base if (base[:3] in ("mBv", "mDv", "cBv", "cDv", "xBv", "xDv", "rBv", "rDv")) else "num" + base
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (LATEX_DIR / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote numbers.tex ({len(NUMS)} macros)")


if __name__ == "__main__":
    main()
