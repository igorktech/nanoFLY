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
import hashlib
import random
import tempfile
import io
import json
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanofly.data import CHAR_ALPHABET, iter_spool, write_prepared  # noqa: E402


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
    with io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8") as text_stream:
        if name.endswith(".jsonl") or name.endswith(".jsonl.zst") or name.endswith(".json"):
            for line in text_stream:
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
            paragraph = []
            for line in text_stream:
                if line.strip():
                    paragraph.append(line.rstrip("\r\n"))
                elif paragraph:
                    yield "\n".join(paragraph).strip()
                    paragraph = []
                    n += 1
                    if limit and n >= limit:
                        return
            if paragraph:
                yield "\n".join(paragraph).strip()
        else:
            raise SystemExit(f"unsupported format: {name} (.txt, .jsonl, .jsonl.zst or .parquet)")


def read_parquet(path, field, limit):
    import pyarrow.parquet as pq
    n = 0
    with pq.ParquetFile(path) as f:
        for batch in f.iter_batches(batch_size=256, columns=[field]):
            for t in batch.column(0).to_pylist():
                t = str(t or "").strip()
                if t:
                    yield t
                    n += 1
                    if limit and n >= limit:
                        return


def source_texts(spec, args):
    field, limit = spec.get("field", args.field), spec.get("limit", 0)
    if "local" in spec:
        name = spec["local"]
        if name.endswith(".parquet"):
            yield from read_parquet(name, field, limit)
        else:
            with open(name, "rb") as stream:
                yield from iter_texts(name, stream, field, limit)
    else:
        if "hf_file" in spec:
            url, name = hub_url(spec["hf_file"], args.repo_type)
            if name.endswith(".parquet"):
                from huggingface_hub import hf_hub_download
                repo, _, path = spec["hf_file"].partition(":")
                local = hf_hub_download(repo, path, repo_type=args.repo_type)
                yield from read_parquet(local, field, limit)
                return
        else:
            url = spec["url"]
            name = url.split("?")[0]
        print(f"streaming {url}")
        with open_stream(url) as stream:
            yield from iter_texts(name, stream, field, limit)


def collect(args):
    """Return a lazy document stream and its source manifest. Limits count documents."""
    specs = []
    for key in ("local", "url", "hf_file"):
        values = getattr(args, key)
        if isinstance(values, str):  # also accept the small dataset wrappers
            values = [values] if values else []
        specs.extend({key: value} for value in values)
    if getattr(args, "mix", ""):
        if specs:
            raise ValueError("use --mix or source flags, not both")
        with open(args.mix, encoding="utf-8") as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError("--mix must contain a JSON list of source objects")
    if not specs:
        raise ValueError("pass --hf-file, --url, --local (repeatable), or --mix")
    for spec in specs:
        if sum(key in spec for key in ("local", "url", "hf_file")) != 1:
            raise ValueError("each source needs exactly one of local, url, hf_file")
        weight = spec.get("weight", 1)
        if not isinstance(weight, (int, float)) or not 0 < weight < float("inf"):
            raise ValueError("source weights must be finite and positive")
        if not isinstance(spec.get("limit", 0), int) or spec.get("limit", 0) < 0:
            raise ValueError("source limits must be nonnegative integers")

    def documents():
        streams = [iter(source_texts(spec, args)) for spec in specs]
        active = list(range(len(streams)))
        rng = random.Random(args.seed)
        n = 0
        try:
            while active and (not args.limit or n < args.limit):
                # Weights describe document sampling without replacement. Exhausted sources leave
                # the mixture; --limit controls the overall budget, per-source limits cap each input.
                i = rng.choices(active, weights=[specs[j].get("weight", 1) for j in active])[0]
                try:
                    text = next(streams[i])
                except StopIteration:
                    active.remove(i)
                    continue
                yield text
                n += 1
        finally:
            for stream in streams:
                stream.close()
    return documents(), specs


def add_arguments(ap):
    ap.add_argument("--out", required=True, help="directory to write the prepared data into")
    ap.add_argument("--hf-file", action="append", default=[], help="repo_id:path/in/repo on the Hub")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--local", action="append", default=[])
    ap.add_argument("--mix", default="", help="JSON list of sources with optional weight, field, limit")
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
    ap.add_argument("--tok-chars", type=int, default=20_000_000, help="maximum characters in the BPE sample")
    ap.add_argument("--force", action="store_true", help="retrain the tokenizer even if one is there")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def prepare(args, texts, source):
    if not 0 <= args.val_frac < 1 or args.limit < 0 or args.val_limit < 0:
        raise ValueError("require 0 <= val-frac < 1 and nonnegative limits")
    os.makedirs(args.out, exist_ok=True)
    # Split before chunking. Identical documents always go to the same split, regardless of source.
    # Validation is spooled while training text flows into write_prepared's tokenizer reservoir.
    with tempfile.TemporaryDirectory(prefix=".split-", dir=args.out) as tmp:
        val_path = os.path.join(tmp, "val.jsonl")
        def training_texts():
            n_val = 0
            with open(val_path, "w", encoding="utf-8") as f:
                for text in texts:
                    digest = hashlib.sha256(f"{args.seed}\0{text}".encode()).digest()
                    held_out = int.from_bytes(digest[:8], "big") / 2**64 < args.val_frac
                    if held_out:
                        if not args.val_limit or n_val < args.val_limit:
                            f.write(json.dumps(text, ensure_ascii=False) + "\n")
                            n_val += 1
                    else:
                        yield text
        return write_prepared(args.out, training_texts(), iter_spool(val_path), vocab=args.vocab,
                              kind=args.vocab_type, alphabet=args.alphabet,
                              tokenizer=args.tokenizer or None, chunk=args.chunk,
                              tok_limit=args.tok_limit, tok_chars=args.tok_chars, seed=args.seed,
                              source=source, force=args.force)


def main():
    ap = add_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    texts, source = collect(args)
    prepare(args, texts, source)


if __name__ == "__main__":
    main()
