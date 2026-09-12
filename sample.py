#!/usr/bin/env python3
"""
Sample from a trained nanoFLY checkpoint.

  python sample.py --ckpt runs/A/ckpt.pt --n 5
  python sample.py --ckpt runs/B/ckpt.pt --news "Janelia published MaleCNS v1.0"

For the encoder-decoder model the post is encoded once and held on the olfactory neurons for the
whole reply. To record the activity of every neuron for the player, use `record.py` in the replyfly
repository instead.
"""
import argparse
import warnings

import torch

from nanofly.generate import generate_one, load_for_generation, post_vector

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", default="", help="defaults to the path recorded in the checkpoint")
    ap.add_argument("--news", action="append", default=[], help="the post to answer (repeatable)")
    ap.add_argument("--news-file", default="", help="file with one post per line")
    ap.add_argument("--prompt", default="", help="start of the text")
    ap.add_argument("--n", type=int, default=1, help="samples per post")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="also print the top-k at every step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device.startswith("mps"):
        raise SystemExit("torch has no sparse CSR matmul on MPS; use --device cpu")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, tok, encoder, ck, _, _ = load_for_generation(args.ckpt, args.graph, device)

    posts = list(args.news)
    if args.news_file:
        with open(args.news_file, encoding="utf-8") as f:
            posts += [l.strip() for l in f if l.strip()]
    if not posts:
        posts = [""]

    for post in posts:
        vec = post_vector(encoder, model, post, device)
        if post:
            print(f"\n{post}")
        for _ in range(args.n):
            text, steps, _ = generate_one(model, tok, vec, args.prompt, device,
                                          max_new=args.max_new, temperature=args.temperature,
                                          top_k=args.top_k)
            print(f"  fly: {text}")
            if args.verbose:
                for s in steps:
                    top = " ".join(f"{t!r}:{p:.2f}" for t, p in s["topk"])
                    print(f"    {s['phase']:5s} {s['input']!r} -> {s['output']!r}  {top}")


if __name__ == "__main__":
    main()
