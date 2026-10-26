"""
Check that the manuscript cannot disagree with its data.

Prose numbers are LaTeX macros generated from the result files
(manuscript/latex/numbers.tex), so a quoted statistic cannot drift. This script
verifies the three remaining ways that can fail:

  1. a macro used in the text is not defined (a result is missing),
  2. a generated macro is never used (stale or dead output; reported only),
  3. a hand-typed statistic from an earlier version of the paper is still in
     the prose (a small list of known-stale strings).

Exit status is non-zero for 1 or 3.
"""

import re
import sys
from pathlib import Path

LATEX = Path(__file__).parent.parent / "manuscript" / "latex"
SOURCES = [p for p in sorted(LATEX.glob("*.tex"))
           if p.name[0].isdigit() and p.name not in ("0_abstract.tex",)] + \
          [LATEX / "0_abstract.tex"]

STALE = [
    r"98\.2\\?%", r"\\pm 0\.003", r"\+0\.049", r"\+0\.106",
    r"p \\approx 0\.9", r"212 call sites", r"8\{,\}000 functions",
    r"taken in corpus order", r"10 matched folds both conditions completed",
]


def main():
    text = "\n".join(p.read_text() for p in SOURCES)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}",
                             (LATEX / "numbers.tex").read_text()))
    used = set(re.findall(r"\\((?:num|[mc](?:Bv|Dv))\w+)", text))
    missing = sorted(used - defined)
    unused = sorted(d for d in defined if d not in used)
    bad = []
    for pat in STALE:
        for m in re.finditer(pat, text):
            line = text[:m.start()].count("\n") + 1
            bad.append(f"stale text /{pat}/ near line {line} of the joined sources")

    print(f"{len(used)} generated macros used, {len(defined)} defined, "
          f"{len(unused)} unused")
    if missing:
        print("UNDEFINED macros (result missing or typo):")
        for m in missing:
            print("  ", m)
    for b in bad:
        print("  ", b)
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
