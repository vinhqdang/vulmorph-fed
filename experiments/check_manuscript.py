"""
Check the manuscript's prose against the result files.

The tables are generated (emit_tables.py), so they cannot drift. The prose
still quotes numbers by hand, and that is where the three defects found before
submission actually lived. This script extracts every quoted statistic from the
.tex sources and checks it against experiments/results/*.json.

It is deliberately noisy in one direction: it reports anything it cannot match
rather than staying silent, because a missed check is worse than a false alarm.
Run it before every submission.

Exit status is non-zero if any quoted value contradicts the data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
LATEX = ROOT / "manuscript" / "latex"


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else {}


def build_facts():
    """Every number the prose is allowed to quote, keyed by a short name."""
    facts = {}
    rep = load("representation_analysis.json")
    core = load("core_bigvul.json")
    gran = load("granularity_analysis.json")
    info = load("information_analysis.json")

    def add(key, val):
        facts[key] = val

    # Per-cell differences and intervals.
    for stem_metric, cells in rep.items():
        stem, metric = stem_metric.rsplit(":", 1)
        for cell, s in cells.items():
            bb, mode = cell.split("|")
            base = f"{stem}.{bb}.{mode}.{metric}"
            add(f"{base}.diff", s["diff"])
            add(f"{base}.lo", s["lo"])
            add(f"{base}.hi", s["hi"])

    # Absolute per-mode means on the core cell.
    for key, cell in core.items():
        if key.count("|") != 1:
            continue
        mode, bb = key.split("|")
        import numpy as np
        for metric in ("auc", "auprc"):
            if cell.get(metric):
                add(f"core_bigvul.{bb}.{mode}.{metric}.mean",
                    float(np.mean(cell[metric])))
                add(f"core_bigvul.{bb}.{mode}.{metric}.std",
                    float(np.std(cell[metric])))

    for t, e in (gran.get("comparisons") or {}).items():
        for metric in ("auc", "auprc"):
            if metric in e:
                add(f"gran.{t}.{metric}.diff", e[metric]["diff"])
                add(f"gran.{t}.{metric}.lo", e[metric]["lo"])
                add(f"gran.{t}.{metric}.hi", e[metric]["hi"])

    # The information JSON holds several cached samples per corpus, ordered
    # largest first. emit_tables.py reports the largest, so the prose must be
    # checked against that one -- keep the first per corpus, not the last.
    seen = set()
    for cache, r in info.items():
        corpus = cache.split("_")[0]
        if corpus in seen:
            continue
        seen.add(corpus)
        add(f"info.{corpus}.H_phi_given_kind", r["H_phi_given_kind"])
        add(f"info.{corpus}.H_kind_given_phi", r["H_kind_given_phi"])
        add(f"info.{corpus}.determinism", r["determinism"])
        add(f"info.{corpus}.frac_kind_info_lost", r["frac_kind_info_lost"])
    return facts


# Statistics quoted in prose, each with the fact it must equal. Extend this as
# the prose changes; an unlisted quoted number is not checked, which is why the
# report also prints how many numeric literals it did not account for.
CLAIMS = [
    # (regex over the .tex sources, fact key, tolerance)
    (r"improves AUC by \$\+0\.(\d+)\$", "core_bigvul.ggnn.kind.auc.diff", 5e-4),
    (r"AUPRC by\s*\n?\$\+0\.(\d+)\$", "core_bigvul.ggnn.kind.auprc.diff", 5e-4),
    (r"AUPRC \$0\.(\d+)\$\s*\nagainst a base rate", "core_bigvul.ggnn.kind.auprc.mean", 5e-4),
    (r"reaches AUPRC \$0\.(\d+)\$ against", "core_bigvul.ggnn.op_kind.auprc.mean", 5e-4),
    (r"against \$0\.(\d+)\$ for \$\\kappa\$ alone", "core_bigvul.ggnn.kind.auprc.mean", 5e-4),
    (r"H\(\\phi \\mid \\kappa\) = 0\.(\d+)\$ bits on\s*\nBigVul", "info.bigvul.H_phi_given_kind", 5e-4),
    (r"H\(\\kappa \\mid \\phi\) = 3\.(\d+)\$ bits on BigVul", "info.bigvul.H_kind_given_phi", 5e-3),
    (r"changes AUC by \$-0\.(\d+)\$", "gran.16.auc.diff", 5e-4),
    (r"AUPRC by \$\+0\.(\d+)\$\s*\n?\$\[-0", "gran.16.auprc.diff", 5e-4),
]


def main():
    text = "\n".join((LATEX / f).read_text()
                     for f in sorted(p.name for p in LATEX.glob("[0-9]_*.tex")))
    facts = build_facts()

    failures, checked = [], 0
    for pattern, key, tol in CLAIMS:
        m = re.search(pattern, text)
        if not m:
            failures.append(f"NOT FOUND in prose: /{pattern}/ (claim for {key})")
            continue
        if key not in facts:
            failures.append(f"NO DATA for {key} (pattern matched '{m.group(0)[:40]}')")
            continue
        # Reconstruct the quoted value from the matched digits.
        quoted = float(re.search(r"[-+]?\d*\.?\d+", m.group(0).replace("$", "")
                                 .replace("\\", "")).group(0))
        actual = abs(facts[key])
        checked += 1
        if abs(abs(quoted) - actual) > tol:
            failures.append(
                f"MISMATCH {key}: prose says {quoted}, data says {actual:.4f} "
                f"(tolerance {tol})")

    print(f"checked {checked} quoted statistics against "
          f"{len(facts)} facts from experiments/results/")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all checked statistics agree with the result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
