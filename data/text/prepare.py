#!/usr/bin/env python3
"""
Turn any corpus of plain texts into a prepared directory nanoFLY can train on.

The other prepare scripts are thin wrappers around this one; use it directly for anything else on
the Hub or on disk.

  # one file from a Hub dataset, stopping after 5,000 records (only the prefix is downloaded)
  python data/text/prepare.py --hf-file dichspace/darulm:rulm_gazeta_0.jsonl.zst \
      --limit 5000 --chunk 320 --vocab 4096 --out data/darulm-gazeta

  # a local file
  python data/text/prepare.py --local mycorpus.jsonl --field text --out data/mine

Formats are taken from the extension: `.txt` (one example per paragraph), `.jsonl` / `.jsonl.zst`
(one JSON object per line, `--field`), `.parquet` (`--field` column). `--chunk N` cuts a long
document into windows of N tokens — a book is not one training example.

What is written: `train.bin` / `val.bin` (uint16 ids), `train_offsets.npy` / `val_offsets.npy`,
`tokenizer.json`, `meta.json`.
"""
import argparse
import io
import json
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanofly.data import CHAR_ALPHABET, split_texts, write_prepared  # noqa: E402


def opener():
    """python.org builds on macOS ship without a CA store; use certifi's bundle when it is there."""
    try:
        import certifi
    except ImportError:
        return urllib.request.build_opener()
    ctx = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def hub_url(spec, repo_type="dataset"):
    """`repo_id:path/in/repo` -> a resolve URL."""
    repo, _, path = spec.partition(":")
    if not path:
        raise SystemExit("--hf-file wants repo_id:path/in/repo")
    kind = "" if repo_type == "model" else f"{repo_type}s/"
    return f"https://huggingface.co/{kind}{repo}/resolve/main/{path}", path


def open_stream(url):
    return opener().open(url, timeout=120)


def iter_texts(name, stream, field, limit):
    """Yield texts from an open binary stream, stopping at `limit` records."""
    n = 0
    if name.endswith(".zst"):
        import zstandard
        stream = zstandard.ZstdDecompressor().stream_reader(stream)
    if name.endswith(".jsonl") or name.endswith(".jsonl.zst") or name.endswith(".json"):
        for line in io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            t = str(json.loads(line).get(field, "") or "").strip()
            if not t:
                continue
            yield t
            n += 1
            if limit and n >= limit:
                return
    elif name.endswith(".txt") or name.endswith(".txt.zst"):
        text = io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8").read()
        parts = [p.strip() for p in text.split("\n\n")]
        parts = [p for p in parts if p] or [l for l in text.split("\n") if l.strip()]
        for t in parts:
            yield t
            n += 1
            if limit and n >= limit:
                return
    else:
        raise SystemExit(f"unsupported format: {name} (.txt, .jsonl, .jsonl.zst or .parquet)")


def read_parquet(path, field, limit):
    import pyarrow.parquet as pq
    out = []
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=8192, columns=[field]):
        for t in batch.column(0).to_pylist():
            t = str(t or "").strip()
            if t:
                out.append(t)
            if limit and len(out) >= limit:
                return out
    return out


def collect(args):
    """-> (texts, a label for meta.json)"""
    if args.hf_file:
        url, name = hub_url(args.hf_file, args.repo_type)
        label = args.hf_file
        if name.endswith(".parquet"):
            from huggingface_hub import hf_hub_download
            repo, _, path = args.hf_file.partition(":")
            local = hf_hub_download(repo, path, repo_type=args.repo_type)
            return read_parquet(local, args.field, args.limit), label
        print(f"streaming {url}")
        with open_stream(url) as s:
            return list(iter_texts(name, s, args.field, args.limit)), label
    if args.url:
        name = args.url.split("?")[0]
        print(f"downloading {args.url}")
        with open_stream(args.url) as s:
            return list(iter_texts(name, s, args.field, args.limit)), args.url
    if args.local:
        if args.local.endswith(".parquet"):
            return read_parquet(args.local, args.field, args.limit), args.local
        with open(args.local, "rb") as s:
            return list(iter_texts(args.local, s, args.field, args.limit)), args.local
    raise SystemExit("pass one of --hf-file, --url or --local")


def add_arguments(ap):
    ap.add_argument("--out", required=True, help="directory to write the prepared data into")
    ap.add_argument("--hf-file", default="", help="repo_id:path/in/repo on the Hub")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--url", default="")
    ap.add_argument("--local", default="")
    ap.add_argument("--field", default="text", help="JSON key or parquet column holding the text")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many documents")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-limit", type=int, default=0, help="cap the validation split")
    ap.add_argument("--chunk", type=int, default=0,
                    help="split a document into windows of this many tokens (0 = keep whole)")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--vocab-type", default="bpe", choices=["bpe", "char"])
    ap.add_argument("--alphabet", default=CHAR_ALPHABET)
    ap.add_argument("--tokenizer", default="", help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--tok-limit", type=int, default=300_000, help="documents the tokenizer is trained on")
    ap.add_argument("--force", action="store_true", help="retrain the tokenizer even if one is there")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def prepare(args, texts, source):
    print(f"{len(texts):,} documents, {sum(len(t) for t in texts) / 1e6:.1f}M characters")
    train, val = split_texts(texts, args.val_frac, args.seed)
    if args.val_limit:
        val = val[:args.val_limit]
    return write_prepared(args.out, train, val, vocab=args.vocab, kind=args.vocab_type,
                          alphabet=args.alphabet, tokenizer=args.tokenizer or None,
                          chunk=args.chunk, tok_limit=args.tok_limit, source=source, force=args.force)


def main():
    ap = add_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    texts, source = collect(args)
    if not texts:
        raise SystemExit("no texts found; check --field and the format")
    prepare(args, texts, source)


if __name__ == "__main__":
    main()
