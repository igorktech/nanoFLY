"""Data for nanoFLY: tokenizers, the token stream the training loop reads, and dataset preparation.

Two on-disk forms, both reaching the loop as a `TokenStream`:

* a **prepared directory** — `train.bin` / `val.bin` (flat uint16 token ids) plus
  `train_offsets.npy` / `val_offsets.npy` (int64, `n + 1` entries) and `meta.json`. Written once by
  `data/<name>/prepare.py`, read back as a memmap, so a corpus that does not fit in memory still
  trains. This is the nanoGPT layout with one addition: offsets, because an example here is a whole
  document and the loop batches documents, not a sliding window.
* a **JSONL or text file** — `{"news": "the post", "text": "the reply"}` per line, or one text per
  line. Tokenised in memory at start-up; this is the path for post/reply pairs, which are few.

The loop only ever needs three things from a stream: how many examples, how long each one is, and a
row of ids it can assign into a padded batch. `TokenStream` is exactly that and nothing more.
"""
import hashlib
import json
import math
import os
import random
import tempfile
import warnings

import numpy as np
import torch

from nanofly.model import BOS, EOS, PAD, SPECIAL_TOKENS

CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,'?!-"
MAX_VOCAB = 2 ** 16          # ids are stored as uint16 in the prepared files


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------- tokenizer

def char_tokenizer(alphabet):
    """Character vocabulary: every output is a whole character, no truncated bytes."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + ["<unk>"] + list(dict.fromkeys(alphabet)))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(""), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return tok


def train_bpe(texts, vocab_size):
    """Byte-level BPE, the same recipe for every corpus: any byte is representable, so a model
    trained on one language can at least read another without <unk>, and `<pad>/<s>/</s>` keep
    ids 0/1/2 (the model, the exported config and every checkpoint assume that)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    assert [tok.token_to_id(s) for s in SPECIAL_TOKENS] == [PAD, BOS, EOS]
    return tok


def get_tokenizer(path, texts, vocab_size, kind="bpe", alphabet=CHAR_ALPHABET):
    """Load `path` if it exists, otherwise train on `texts` and save it there."""
    from tokenizers import Tokenizer
    if path and os.path.exists(path):
        tok = Tokenizer.from_file(path)
        validate_tokenizer(tok)
        return tok
    if kind == "char":
        tok = char_tokenizer(alphabet)
        unknown = sorted({c for t in texts for c in t.lower() if tok.token_to_id(c) is None})
        if unknown:
            print(f"{len(unknown)} characters outside the alphabet become <unk>: {''.join(unknown[:40])}")
    else:
        tok = train_bpe(texts, vocab_size)
    if path:
        tok.save(path)
    return tok


def validate_tokenizer(tok):
    if [tok.token_to_id(t) for t in SPECIAL_TOKENS] != [PAD, BOS, EOS]:
        raise ValueError("tokenizer must use <pad>=0, <s>=1, </s>=2")
    ids = list(tok.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= min(MAX_VOCAB, tok.get_vocab_size()):
        raise ValueError("tokenizer ids must be dense and fit in uint16")


def data_sha256(path):
    """Content identity, including token IDs and offsets, independent of the dataset's location."""
    names = ("train.bin", "val.bin", "train_offsets.npy", "val_offsets.npy")
    paths = [os.path.join(path, n) for n in names] if os.path.isdir(path) else [path]
    h = hashlib.sha256()
    for filename in paths:
        h.update(file_sha256(filename).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------ token stream

class TokenStream:
    """Token ids of every example, flat, plus where each one starts.

    `ids` is uint16 (a memmap for prepared data, an array otherwise) and `offsets` is int64 with
    `len(self) + 1` entries, so example `i` is `ids[offsets[i]:offsets[i + 1]]`. `news` holds the
    post text per example when there is one; `news_idx` is filled in by the training script once the
    posts have been encoded.
    """

    def __init__(self, ids, offsets, news=None, max_len=0):
        self.ids = ids
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.news = news
        self.max_len = int(max_len)
        self.news_idx = None
        self._lengths = None

    # -- construction

    @classmethod
    def from_dir(cls, d, split, max_len=0):
        path = os.path.join(d, f"{split}.bin")
        ids = np.memmap(path, dtype=np.uint16, mode="r") if os.path.getsize(path) else np.empty(0, np.uint16)
        offsets = np.load(os.path.join(d, f"{split}_offsets.npy"), mmap_mode="r")
        if (offsets.ndim != 1 or offsets.dtype != np.int64 or not len(offsets)
                or offsets[0] != 0 or offsets[-1] != len(ids)
                or np.any(np.diff(offsets) < 2)):
            raise ValueError(f"invalid {split} offsets: expected documents of at least two tokens")
        return cls(ids, offsets, max_len=max_len)

    @classmethod
    def from_lists(cls, seqs, news=None, max_len=0):
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in seqs], out=offsets[1:])
        ids = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs]) if seqs else np.zeros(0, np.int64)
        if len(ids) and ids.max() >= MAX_VOCAB:
            raise ValueError(f"token id {ids.max()} does not fit in uint16")
        return cls(ids.astype(np.uint16), offsets, news=news, max_len=max_len)

    # -- what the loop uses

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        if self.max_len and e - s > self.max_len:
            # a copy, always: `ids` may be a read-only memmap, and the last token must be EOS
            out = np.array(self.ids[s:s + self.max_len], dtype=np.uint16)
            out[-1] = EOS
            return out
        return self.ids[s:e]

    @property
    def lengths(self):
        if self._lengths is None:
            n = np.diff(self.offsets)
            self._lengths = np.minimum(n, self.max_len) if self.max_len else n
        return self._lengths

    @property
    def n_tokens(self):
        """Tokens the loss is computed on: every example predicts its own length minus one."""
        return int(np.maximum(self.lengths - 1, 0).sum())

    # -- slicing

    def head(self, n):
        """The first `n` examples, sharing the same ids (no copy)."""
        n = min(n, len(self))
        out = TokenStream(self.ids, self.offsets[:n + 1], max_len=self.max_len)
        out.news = self.news[:n] if self.news is not None else None
        return out

    def subset(self, idx):
        """A copy holding only the chosen examples, in that order."""
        idx = list(idx)
        seqs = [np.asarray(self[i]) for i in idx]
        news = [self.news[i] for i in idx] if self.news is not None else None
        return TokenStream.from_lists(seqs, news=news, max_len=self.max_len)


# ------------------------------------------------------------------------------------- preparation

def _encode_documents(tok, texts, chunk=0):
    """Encode one document at a time; zero chunk keeps its full recurrent context."""
    if chunk and chunk < 3:
        raise ValueError("chunk must be 0 or at least 3 (BOS + text + EOS)")
    for text in texts:
        ids = tok.encode(text).ids
        if not ids:
            continue
        if chunk:
            for c in range(0, len(ids), chunk - 2):
                yield [BOS] + ids[c:c + chunk - 2] + [EOS]
        else:
            yield [BOS] + ids + [EOS]


def iter_spool(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_prepared(out, texts_train, texts_val, vocab=1024, kind="bpe", alphabet=CHAR_ALPHABET,
                   tokenizer=None, chunk=0, tok_limit=300_000, source="", force=False,
                   tok_chars=20_000_000, seed=0):
    """Spool training text to disk, fit BPE on a bounded reservoir, then stream uint16 IDs.

    RAM is bounded by the tokenizer sample and the largest individual document. Temporary disk
    holds the training text; neither the corpus nor its growing offsets are collected in RAM.
    """
    if tok_limit < 1 or tok_chars < 1 or (chunk and chunk < 3):
        raise ValueError("positive tokenizer limits and chunk=0 or chunk>=3 required")
    os.makedirs(out, exist_ok=True)
    tok_path = tokenizer or os.path.join(out, "tokenizer.json")
    if force and not tokenizer and os.path.exists(tok_path):
        os.remove(tok_path)
    # Fixed-size snippets make the reservoir bounded even for arbitrarily long documents.
    snippet_chars = min(2048, tok_chars)
    sample_size = min(tok_limit, max(1, tok_chars // snippet_chars))
    rng = random.Random(seed)
    sample, chars, n_docs = [], 0, 0
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=out) as tmp:
        spool = os.path.join(tmp, "train.jsonl")
        with open(spool, "w", encoding="utf-8") as f:
            for text in texts_train:
                if not text.strip():
                    continue
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
                chars += len(text)
                n_docs += 1
                j = rng.randrange(n_docs)
                if len(sample) < sample_size or j < sample_size:
                    start = rng.randrange(max(1, len(text) - snippet_chars + 1))
                    snippet = text[start:start + snippet_chars]
                    if len(sample) < sample_size:
                        sample.append(snippet)
                    else:
                        sample[j] = snippet
        if not n_docs:
            raise ValueError("no training documents; check the field, limits, and validation fraction")
        tok = get_tokenizer(tok_path, sample, vocab, kind, alphabet)
        validate_tokenizer(tok)
        sample_chars = sum(map(len, sample))
        del sample
        if os.path.abspath(tok_path) != os.path.abspath(os.path.join(out, "tokenizer.json")):
            tok.save(os.path.join(out, "tokenizer.json"))
        counts = {}
        for split, texts in (("train", iter_spool(spool)), ("val", texts_val)):
            raw_offsets = os.path.join(tmp, "offsets.bin")
            n, total = 0, 0
            with open(os.path.join(out, f"{split}.bin"), "wb") as f, open(raw_offsets, "wb") as off:
                off.write(np.int64(0).tobytes())
                for seq in _encode_documents(tok, texts, chunk):
                    f.write(np.asarray(seq, dtype=np.uint16).tobytes())
                    total += len(seq)
                    n += 1
                    off.write(np.int64(total).tobytes())
            offsets = np.memmap(raw_offsets, dtype=np.int64, mode="r")
            np.save(os.path.join(out, f"{split}_offsets.npy"), offsets)
            del offsets
            counts[split] = n, total
            print(f"  {split}: {n:,} examples, {total:,} tokens")
    meta = {
        "source": source, "vocab_size": tok.get_vocab_size(), "tokenizer": "tokenizer.json",
        "tokenizer_kind": kind, "tokenizer_sha256": file_sha256(os.path.join(out, "tokenizer.json")),
        "chunk": chunk, "seed": seed, "tokenizer_sample_chars": sample_chars,
        "n_train": counts["train"][0], "n_val": counts["val"][0],
        "tokens_train": counts["train"][1], "tokens_val": counts["val"][1],
        "chars_per_token": round(chars / max(counts["train"][1], 1), 2),
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"  vocabulary {meta['vocab_size']}, {meta['chars_per_token']} characters per token -> {out}")
    return meta


def split_texts(texts, val_frac, seed=0):
    """Seeded split into train and val."""
    if not texts or val_frac <= 0:
        return texts, []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    n_val = int(len(texts) * val_frac)
    return [texts[i] for i in perm[n_val:]], [texts[i] for i in perm[:n_val]]


# ------------------------------------------------------------------------------------------ input

def read_examples(path, text_field="text", news_field="news", limit=0):
    """One example per line: a JSON object with `text` (and optionally `news`), or a plain line."""
    texts, news = [], []
    is_json = path.endswith(".jsonl") or path.endswith(".json")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_json:
                obj = json.loads(line)
                t = str(obj.get(text_field, "")).strip()
                if not t:
                    continue
                texts.append(t)
                news.append(str(obj.get(news_field, "") or "").strip())
            else:
                texts.append(line)
                news.append("")
            if limit and len(texts) >= limit:
                break
    return texts, news


def load_data(args):
    """`--data` is either a prepared directory or a JSONL/text file.

    Returns `(train, val, tokenizer, tokenizer_path)`. The tokenizer always ends up in `--out` as
    `tokenizer.json`, so a checkpoint and its vocabulary travel together.
    """
    out_tok = os.path.join(args.out, "tokenizer.json")
    if os.path.isdir(args.data):
        for f in ("train.bin", "train_offsets.npy", "val.bin", "val_offsets.npy"):
            if not os.path.exists(os.path.join(args.data, f)):
                raise SystemExit(f"{args.data} is not a prepared directory: {f} is missing "
                                 "(run data/<name>/prepare.py first)")
        tok_path = args.tokenizer or os.path.join(args.data, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise SystemExit(f"tokenizer not found: {tok_path}")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        validate_tokenizer(tok)
        meta_path = os.path.join(args.data, "meta.json")
        if not os.path.exists(meta_path):
            raise ValueError("prepared data needs meta.json with tokenizer_sha256; prepare it again")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("tokenizer_sha256") or file_sha256(tok_path) != meta["tokenizer_sha256"]:
            raise ValueError("prepared data tokenizer mismatch: token IDs belong to another tokenizer; "
                             "re-prepare the corpus with the checkpoint tokenizer")
        train = TokenStream.from_dir(args.data, "train", args.max_len)
        val = TokenStream.from_dir(args.data, "val", args.max_len)
        if args.limit:
            train = train.head(args.limit)
        if args.val_limit:
            val = val.head(args.val_limit)
        if args.val_frac and len(val):
            print("  prepared data has its own validation split, --val-frac ignored")
    elif os.path.isfile(args.data):
        texts, news = read_examples(args.data, args.text_field, args.news_field, args.limit)
        print(f"examples: {len(texts):,}")
        tok_path = args.tokenizer or out_tok
        tok = get_tokenizer(tok_path, texts, args.vocab, args.vocab_type, args.alphabet)
        validate_tokenizer(tok)
        seqs = [[BOS] + e.ids + [EOS] for e in tok.encode_batch(texts)]
        n_val = int(len(seqs) * args.val_frac) if len(seqs) > 50 else 0
        perm = np.random.default_rng(args.seed).permutation(len(seqs))
        val_i, train_i = perm[:n_val].tolist(), perm[n_val:].tolist()
        train = TokenStream.from_lists([seqs[i] for i in train_i], [news[i] for i in train_i], args.max_len)
        val = TokenStream.from_lists([seqs[i] for i in val_i], [news[i] for i in val_i], args.max_len)
    else:
        raise SystemExit(f"--data {args.data}: expected a prepared directory or a .jsonl/.txt file")

    validate_tokenizer(tok)
    if not len(train):
        raise ValueError("training split is empty")
    for name, stream in (("train", train), ("val", val)):
        if args.max_len:
            original = np.diff(stream.offsets)
            lost = int(np.maximum(original - args.max_len, 0).sum())
            if lost:
                warnings.warn(f"{name}: explicit --max-len={args.max_len} truncates "
                              f"{lost:,} tokens; use --max-len 0 to keep full documents "
                              "or prepare with --chunk to retain all text")
    os.makedirs(args.out, exist_ok=True)
    if os.path.abspath(tok_path) != os.path.abspath(out_tok):
        tok.save(out_tok)
    print(f"training tokens: {train.n_tokens:,} in {len(train):,} examples, "
          f"vocabulary {tok.get_vocab_size()}, validation {len(val):,}")
    return train, val, tok, out_tok


def news_embeddings(news, encoder_name, cache_dir, device="cpu"):
    """Post text -> one embedding per unique post, plus the per-example index into that table."""
    from nanofly.encoders import load_news_encoder
    uniq = sorted(set(news))
    if not encoder_name or encoder_name == "none" or uniq == [""]:
        return None, None, 0
    key = hashlib.md5(("\n".join(uniq) + encoder_name).encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"news_{key}.npy")
    if os.path.exists(cache):
        emb = np.load(cache)
    else:
        enc = load_news_encoder(encoder_name, device)
        print(f"encoding {len(uniq):,} unique posts with {encoder_name}")
        emb = enc.encode(uniq)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache, emb)
    index = {s: i for i, s in enumerate(uniq)}
    return emb, np.array([index[s] for s in news]), emb.shape[1]


# ----------------------------------------------------------------------------------------- batching

def make_batches(lengths, batch_size, seed):
    """Examples of similar length land in the same batch, so padding stays small; the jitter keeps
    the buckets from being identical every epoch, and the batch order is shuffled."""
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] + rng.random() * 3)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def batch_context(idx, news_emb, news_idx, vis_cur, device):
    """The post channel for one batch: the same constant current for every token of an example."""
    news = torch.from_numpy(news_emb[news_idx[idx]]).to(device) if news_emb is not None else None
    vis = torch.from_numpy(vis_cur[news_idx[idx]]).to(device) if vis_cur is not None else None
    return news, vis


def pad_batch(stream, idx, seq_len):
    """`[B, L]` of ids, left-aligned and padded, with `L - 1` an exact multiple of the TBPTT window."""
    rows = [stream[i] for i in idx]
    L = max(len(r) for r in rows)
    L = int(math.ceil((L - 1) / seq_len) * seq_len) + 1
    out = np.full((len(rows), L), PAD, dtype=np.int64)
    for r, row in enumerate(rows):
        out[r, :len(row)] = row
    return torch.from_numpy(out)
