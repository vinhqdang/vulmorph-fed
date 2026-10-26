"""
Export the exact sample and folds of the experiments for the GPU fine-tuning
run: function source, label, project, and for each of the ten folds the train
and test indices (test indices in the order used by run_representation.py, so
the stored scores line up with every other system's).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gzip
import json

from data.loaders.sampled import load_sampled, CORPORA
from data.loaders.real_datasets import dedupe_functions
from data.normalise import normalise_code
from utils.protocol import project_group_kfold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--sample_seed", type=int, default=0)
    args = p.parse_args()
    data, _, _ = load_sampled(args.dataset, args.n, args.sample_seed, 8)
    data = dedupe_functions(data)
    from datasets import load_dataset
    hf, code_key, _, _ = CORPORA[args.dataset]
    ds = load_dataset(hf, split="train")
    codes = [ds[d.src_index][code_key] or "" for d in data]
    folds = [{"train": tr, "test": te, "info": info}
             for tr, te, info in project_group_kfold(data, 5, 2)]
    out = Path("colab_data") / f"{args.dataset}.json.gz"
    with gzip.open(out, "wt") as f:
        json.dump({"dataset": args.dataset, "codes": codes,
                   "codes_norm": [normalise_code(c) for c in codes],
                   "y": [float(d.y[0]) for d in data],
                   "project": [d.project for d in data],
                   "folds": folds}, f)
    print(f"{args.dataset}: {len(codes)} functions, {len(folds)} folds -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
