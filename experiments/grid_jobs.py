"""
Emit the experiment grid as independent shell jobs (one result file each), in
priority order, for `xargs -P`. Every job covers two folds so that the grid
balances across workers; analyze_representation.py merges the files and
verifies that all of them used the same fold assignment.

  python experiments/grid_jobs.py | xargs -P 4 -I{} sh -c "{}"
"""

PAIRS = ["0,1", "2,3", "4,5", "6,7", "8,9"]
R = "python experiments/run_representation.py"


def job(corpus, tag, modes, bbs, pair, extra=""):
    f = pair.replace(",", "")
    out = f"results/{corpus}_{tag}_f{f}.json"
    log = f"experiments/logs/{corpus}_{tag}_f{f}.log"
    return (f"{R} --dataset {corpus} --modes {modes} --backbones {bbs} "
            f"--folds {pair} {extra} --output {out} > {log} 2>&1")


def main():
    jobs = []
    for p in PAIRS:
        jobs.append(job("bigvul", "core", "lexical,kind,op_kind,lex_kind,anon_kind", "ggnn", p))
        jobs.append(job("diversevul", "core", "lexical,kind,op_kind,lex_kind,anon_kind", "ggnn", p))
    for p in PAIRS:
        jobs.append(job("bigvul", "bb", "lexical,kind,op_kind", "gat,gin", p))
        jobs.append(job("diversevul", "bb", "lexical,kind,op_kind", "gat,gin", p))
    for p in PAIRS:
        jobs.append(job("bigvul", "extra", "op,anon,kind_low", "ggnn", p))
    for p in PAIRS:
        jobs.append(job("bigvul", "tax16", "op_kind", "ggnn", p, "--taxonomy_size 16"))
    for es in ("ast", "ast_sib"):
        for p in PAIRS:
            jobs.append(job("bigvul", f"edge-{es}", "kind", "ggnn", p, f"--edge_set {es}"))
    print("\n".join(jobs))


if __name__ == "__main__":
    main()
