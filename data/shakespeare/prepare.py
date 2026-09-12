#!/usr/bin/env python3
"""
Tiny Shakespeare, the nanoGPT dev set: 1.1 MB, one file, trains on a laptop in minutes.

  python data/shakespeare/prepare.py --out data/shakespeare --vocab 512

An example is one speech (the text is split on blank lines). Use it to check the whole pipeline end
to end before spending a GPU hour on anything real.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.text.prepare import add_arguments, collect, prepare  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = add_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.set_defaults(out="data/shakespeare", vocab=512, val_frac=0.05)
    args = ap.parse_args()
    if not (args.local or args.url or args.hf_file):
        args.url = URL
    texts, source = collect(args)
    prepare(args, texts, source)


if __name__ == "__main__":
    main()
