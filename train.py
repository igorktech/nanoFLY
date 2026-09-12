#!/usr/bin/env python3
"""
Train fly-LM on "post -> reply" pairs.

Data: JSONL, one example per line: {"news": "the post", "text": "a short reply"}.
A .txt file also works (one line = one text, no posts).

Examples:
  python train.py --graph graph/graph.npz --data pairs.jsonl --out runs/gains
  python train.py --graph graph/graph.npz --data pairs.jsonl --out runs/edges --mode edges --readout dn+motor
  python train.py --graph graph_synth/graph.npz --data demo.jsonl --out /tmp/t --news-encoder hash --max-steps 20
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from nanofly.encoders import load_news_encoder
from nanofly.model import (BOS, EOS, PAD, SPECIAL_TOKENS, FlyConfig, FlyLM, load_graph,
                           save_checkpoint)

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")


def read_data(path, text_field, news_field, limit):
    texts, news = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl") or path.endswith(".json"):
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


CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,'?!-"


def char_tokenizer(alphabet):
    """Character vocabulary: every output is a whole character, no truncated bytes."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + ["<unk>"] + list(dict.fromkeys(alphabet)))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(""), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return tok


def get_tokenizer(path, texts, vocab_size, kind="bpe", alphabet=CHAR_ALPHABET):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if path and os.path.exists(path):
        return Tokenizer.from_file(path)
    if kind == "char":
        tok = char_tokenizer(alphabet)
        unknown = sorted({c for t in texts for c in t.lower() if tok.token_to_id(c) is None})
        if unknown:
            print(f"{len(unknown)} characters outside the alphabet become <unk>: {''.join(unknown[:40])}")
        if path:
            tok.save(path)
        return tok
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    assert [tok.token_to_id(s) for s in SPECIAL_TOKENS] == [PAD, BOS, EOS]
    if path:
        tok.save(path)
    return tok


def vision_currents(news, graph, mode, args):
    """The post arrives as a picture on the photoreceptors instead of an embedding."""
    from nanofly.vision import TextRetina, embedding_pattern, eye_layout, retina_resolution
    lay = eye_layout(graph)
    if lay is None:
        raise SystemExit("too few photoreceptors in the graph for visual input")
    print(f"retina: {len(lay['idx'])} receptors, {retina_resolution(lay['xy'], lay['side'])}")
    uniq = sorted(set(news))
    index = {s: i for i, s in enumerate(uniq)}
    if mode == "code":
        from nanofly.encoders import load_news_encoder
        enc = load_news_encoder(args.news_encoder if args.news_encoder != "none" else "hash", "cpu")
        emb = enc.encode(uniq)
        cur = np.stack([embedding_pattern(e, lay["xy"]) for e in emb])
    else:
        r = TextRetina(lay, mode=mode, width=args.vision_width, height=args.vision_height,
                       zoom=args.vision_zoom, scale=args.vision_scale)
        cur = np.stack([r.current(t, 0, 1)[0] for t in uniq])
    cur = (cur * args.vision_scale).astype(np.float32)
    return cur, np.array([index[s] for s in news]), lay


def news_embeddings(news, encoder_name, cache_dir, device):
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
        np.save(cache, emb)
    index = {s: i for i, s in enumerate(uniq)}
    return emb, np.array([index[s] for s in news]), emb.shape[1]


def make_batches(lengths, batch_size, seed):
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] + rng.random() * 3)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def pad_batch(seqs, idx, seq_len):
    L = max(len(seqs[i]) for i in idx)
    L = int(math.ceil((L - 1) / seq_len) * seq_len) + 1
    out = np.full((len(idx), L), PAD, dtype=np.int64)
    for r, i in enumerate(idx):
        out[r, :len(seqs[i])] = seqs[i]
    return torch.from_numpy(out)


def run_batch(model, ids, news, seq_len, device, train=True, opt=None, sched=None, clip=1.0, vision=None):
    ids = ids.to(device)
    B, L = ids.shape
    state = model.init_state(B, device)
    total, count = 0.0, 0
    for c in range(0, L - 1, seq_len):
        inp = ids[:, c:c + seq_len]
        tgt = ids[:, c + 1:c + seq_len + 1].clone()
        tgt[tgt == PAD] = -100
        if (tgt != -100).sum() == 0:
            break
        with torch.set_grad_enabled(train):
            logits, state, _ = model(inp, news, state, vision=vision)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        n_tok = int((tgt != -100).sum())
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
        state = (state[0].detach(), state[1])
        total += loss.item() * n_tok
        count += n_tok
    return total, count


def batch_context(idx, news_emb, news_idx, vis_cur, device):
    news = torch.from_numpy(news_emb[news_idx[idx]]).to(device) if news_emb is not None else None
    vis = torch.from_numpy(vis_cur[news_idx[idx]]).to(device) if vis_cur is not None else None
    return news, vis


def evaluate(model, seqs, news_idx, news_emb, idx_list, args, device, vis_cur=None):
    model.eval()
    total, count = 0.0, 0
    for b in make_batches([len(seqs[i]) for i in idx_list], args.batch, 0):
        idx = [idx_list[i] for i in b]
        news, vis = batch_context(idx, news_emb, news_idx, vis_cur, device)
        t, c = run_batch(model, pad_batch(seqs, idx, args.seq), news, args.seq, device, train=False, vision=vis)
        total, count = total + t, count + c
    model.train()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--news-field", default="news")
    ap.add_argument("--tokenizer", default="", help="existing tokenizer.json; without it a BPE is trained on the texts")
    ap.add_argument("--vocab", type=int, default=2048, help="vocabulary size for bpe")
    ap.add_argument("--vocab-type", default="bpe", choices=["bpe", "char"])
    ap.add_argument("--alphabet", default=CHAR_ALPHABET, help="alphabet for --vocab-type char")
    ap.add_argument("--arch", default="encoder-decoder", choices=["decoder", "encoder-decoder"],
                    help="decoder: tokens only, like ngxson/fly-llm-hf. "
                         "encoder-decoder: the post is encoded and fed into the ORNs")
    ap.add_argument("--news-encoder", default="intfloat/multilingual-e5-small",
                    help="a sentence-transformers model, 'hash' for tests or 'none'")
    ap.add_argument("--mode", default="gains", choices=["gains", "edges"])
    ap.add_argument("--readout", default="all", help="all | dn | motor | dn+motor | dn+motor+ascending")
    ap.add_argument("--token-input", default="cb_sensory",
                    help="where the token delay line goes: a superclass list (cb_sensory), a group name "
                         "(orn), or 'sensory' for every sensory neuron of the CNS")
    ap.add_argument("--rank", type=int, default=256, help="readout rank when it is large (0 = full)")
    ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--delay", type=int, default=8)
    ap.add_argument("--d-emb", type=int, default=256)
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--modulatory-sign", type=float, default=0.0)
    ap.add_argument("--news-group", default="orn")
    ap.add_argument("--news-mode", default="direct", choices=["direct", "glomeruli"],
                    help="glomeruli: the post becomes a pattern over glomeruli, ORNs of one type share a value")
    ap.add_argument("--vision", default="off", choices=["off", "banner", "glyph", "code"],
                    help="feed the post as a picture on the photoreceptors instead of an ORN embedding")
    ap.add_argument("--vision-width", type=int, default=128)
    ap.add_argument("--vision-height", type=int, default=32)
    ap.add_argument("--vision-zoom", type=float, default=1.0)
    ap.add_argument("--vision-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=32, help="TBPTT window length")
    ap.add_argument("--max-len", type=int, default=160, help="truncate an example to this many tokens")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-edges", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N batches (for smoke tests)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.arch == "decoder":
        args.news_encoder = "none"
    elif args.news_encoder == "none" and args.vision == "off":
        raise SystemExit("--arch encoder-decoder needs an encoder: pass --news-encoder "
                         "(or hash for a smoke test), or use --arch decoder")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    texts, news = read_data(args.data, args.text_field, args.news_field, args.limit)
    print(f"examples: {len(texts):,}")
    tok_path = args.tokenizer or os.path.join(args.out, "tokenizer.json")
    tok = get_tokenizer(tok_path, texts, args.vocab, args.vocab_type, args.alphabet)
    if os.path.abspath(tok_path) != os.path.abspath(os.path.join(args.out, "tokenizer.json")):
        shutil.copy(tok_path, os.path.join(args.out, "tokenizer.json"))
    seqs = [[BOS] + e.ids[:args.max_len - 2] + [EOS] for e in tok.encode_batch(texts)]
    n_tokens = sum(len(s) - 1 for s in seqs)
    chars = sum(len(t) for t in texts)
    print(f"training tokens: {n_tokens:,}, vocabulary {tok.get_vocab_size()}, "
          f"characters per token {chars / max(n_tokens, 1):.2f}")

    graph = load_graph(args.graph)
    vis_cur = None
    if args.vision != "off":
        vis_cur, news_idx, _ = vision_currents(news, graph, args.vision, args)
        news_emb, news_dim = None, 0
    else:
        news_emb, news_idx, news_dim = news_embeddings(news, args.news_encoder, args.out, "cpu")
    cfg = FlyConfig(vocab_size=tok.get_vocab_size(), d_emb=args.d_emb, delay=args.delay, ticks=args.ticks,
                    mode=args.mode, token_input=args.token_input, readout=args.readout,
                    readout_rank=args.rank, news_dim=news_dim,
                    news_group=args.news_group, news_mode=args.news_mode, min_syn=args.min_syn, modulatory_sign=args.modulatory_sign,
                    vision=args.vision, seed=args.seed)
    model = FlyLM(cfg, graph).to(device)
    del graph
    print(model.describe())

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(seqs))
    n_val = int(len(seqs) * args.val_frac) if len(seqs) > 50 else 0
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()

    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    edge_params = [p for n, p in model.named_parameters() if n == "logw"]
    other = [p for n, p in model.named_parameters() if not n.startswith("head.") and n != "logw"]
    groups = [{"params": other, "lr": args.lr, "weight_decay": 0.0},
              {"params": head_params, "lr": args.lr_head, "weight_decay": args.wd}]
    if edge_params:
        groups.append({"params": edge_params, "lr": args.lr_edges, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    chunks_per_epoch = sum(math.ceil((min(len(seqs[i]), args.max_len) - 1) / args.seq)
                           for i in train_idx) / max(args.batch, 1)
    total_steps = max(1, int(chunks_per_epoch * args.epochs))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    extra = {"tokenizer": "tokenizer.json", "news_encoder": args.news_encoder if news_dim else "none",
             "arch": "encoder_decoder" if news_dim else "decoder", "vision": args.vision,
             "graph": os.path.abspath(args.graph), "args": vars(args)}

    step, t0, seen, run_loss, run_tok = 0, time.time(), 0, 0.0, 0
    best = float("inf")
    for epoch in range(args.epochs):
        for b in make_batches([len(seqs[i]) for i in train_idx], args.batch, args.seed + epoch):
            idx = [train_idx[i] for i in b]
            nb, vb = batch_context(idx, news_emb, news_idx, vis_cur, device)
            loss_sum, n_tok = run_batch(model, pad_batch(seqs, idx, args.seq), nb, args.seq, device,
                                        train=True, opt=opt, sched=sched, clip=args.clip, vision=vb)
            step += 1
            seen += n_tok
            run_loss += loss_sum
            run_tok += n_tok
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"epoch {epoch} step {step} loss {run_loss / max(run_tok, 1):.3f} "
                      f"{seen / dt:,.0f} tok/s  rho {math.exp(model.log_rho.item()):.3f}", flush=True)
                run_loss, run_tok = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                break
        if val_idx:
            vl = evaluate(model, seqs, news_idx, news_emb, val_idx, args, device, vis_cur)
            print(f"epoch {epoch}: val loss {vl:.3f}, ppl {math.exp(min(vl, 20)):.1f}")
            if vl < best:
                best = vl
                extra["val_loss"] = float(vl)
                extra["epoch"] = epoch
                save_checkpoint(os.path.join(args.out, "ckpt.pt"), model, extra)
        else:
            save_checkpoint(os.path.join(args.out, "ckpt.pt"), model, extra)
        if args.max_steps and step >= args.max_steps:
            break
    save_checkpoint(os.path.join(args.out, "last.pt"), model, extra)
    print(f"done in {time.time() - t0:.0f} s, checkpoint {os.path.join(args.out, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
