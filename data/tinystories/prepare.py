#!/usr/bin/env python3
"""
TinyStories: short children's stories, the corpus ngxson/fly-llm-hf was trained on.

  python data/tinystories/prepare.py --out data/tinystories --limit 10000     # 10 MB downloaded
  python data/tinystories/prepare.py --out data/tinystories --split train --shards 2   # 250 MB each

The default reads the published `validation` file — 10 MB for ~22,000 stories, which is the cheapest
way to get a real sample onto a laptop. Our own held-out split is cut from whatever is loaded
(`--val-frac`), so the published split boundary carries no meaning here. Use `--split train` on a
machine with room for it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.text.prepare import add_arguments, collect, prepare  # noqa: E402

REPO = "roneneldan/TinyStories"
VALIDATION = "data/validation-00000-of-00001-869c898b519ad725.parquet"
TRAIN_SHARDS = [
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
]


def main():
    ap = add_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--split", default="validation", choices=["validation", "train"],
                    help="validation is one 10 MB file; train is 4 shards of 250 MB")
    ap.add_argument("--shards", type=int, default=1, help="how many train shards to read")
    ap.set_defaults(out="data/tinystories", vocab=1024)
    args = ap.parse_args()

    files = [VALIDATION] if args.split == "validation" else TRAIN_SHARDS[:max(1, args.shards)]
    want, texts, used = args.limit, [], 0
    for f in files:
        if want and len(texts) >= want:
            break
        args.hf_file = f"{REPO}:{f}"
        args.limit = want - len(texts) if want else 0
        texts += collect(args)[0]
        used += 1
    args.limit = want
    prepare(args, texts, f"{REPO} ({args.split}, {used} file(s))")


if __name__ == "__main__":
    main()
