#!/usr/bin/env python3
"""
Train nanoFLY: a connectome reservoir language model.

Two shapes of data:
  a prepared directory   data/<name>/prepare.py wrote train.bin / val.bin / tokenizer.json
  a JSONL or text file   {"news": "the post", "text": "a short reply"} per line, or one text per line

Examples:
  python data/tinystories/prepare.py --out data/tinystories --limit 10000
  python train.py --graph graph_cb/graph.npz --data data/tinystories --out runs/A --arch decoder
  python train.py --graph graph_cb/graph.npz --data data/pairs/pairs.jsonl --out runs/B \
      --arch encoder-decoder --init-from runs/A/ckpt.pt
"""
import argparse
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from nanofly.data import (CHAR_ALPHABET, batch_context, file_sha256, load_data, make_batches,
                          news_embeddings, pad_batch)
from nanofly.model import PAD, FlyConfig, FlyLM, load_graph, save_checkpoint

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")


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


def run_batch(model, ids, news, seq_len, device, train=True, opt=None, sched=None, clip=1.0, vision=None):
    """One batch, truncated backpropagation through time: the reservoir state carries across the
    `seq_len` windows, the optimiser steps on each one, and only the state crosses the boundary."""
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


def evaluate(model, stream, news_emb, args, device, vis_cur=None):
    model.eval()
    total, count = 0.0, 0
    for b in make_batches(stream.lengths, args.batch, 0):
        news, vis = batch_context(b, news_emb, stream.news_idx, vis_cur, device)
        t, c = run_batch(model, pad_batch(stream, b, args.seq), news, args.seq, device,
                         train=False, vision=vis)
        total, count = total + t, count + c
    model.train()
    return total / max(count, 1)


# Flags that describe the model rather than the run. With --init-from they default to the values in
# the checkpoint, so a fine-tune cannot silently build a different brain than the one it loads.
ARCH_FLAGS = ["d_emb", "delay", "ticks", "mode", "readout", "rank", "token_input", "news_group",
              "min_syn", "modulatory_sign", "vocab", "vocab_type"]
ARCH_DEFAULTS = {"d_emb": 256, "delay": 8, "ticks": 2, "mode": "gains", "readout": "all", "rank": 256,
                 "token_input": "cb_sensory,visual_projection", "news_group": "orn", "min_syn": 1,
                 "modulatory_sign": 0.0, "vocab": 1024, "vocab_type": "bpe"}
# name in the checkpoint's cfg, when it differs from the flag
CFG_NAME = {"rank": "readout_rank", "vocab": "vocab_size"}


def resolve_arch(args, ck_cfg):
    """Fill the architecture flags the user did not pass: from the checkpoint when fine-tuning,
    from the defaults otherwise."""
    for flag in ARCH_FLAGS:
        if getattr(args, flag) is not None:
            continue
        value = ARCH_DEFAULTS[flag]
        if ck_cfg is not None:
            value = ck_cfg.get(CFG_NAME.get(flag, flag), value)
        setattr(args, flag, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--data", required=True, help="a prepared directory or a .jsonl/.txt file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default="",
                    help="checkpoint to start from; the architecture and tokenizer come from it")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--news-field", default="news")
    ap.add_argument("--tokenizer", default="", help="existing tokenizer.json; defaults to the one "
                                                    "next to --data or --init-from")
    ap.add_argument("--vocab", type=int, default=None, help="vocabulary size when training a bpe")
    ap.add_argument("--vocab-type", default=None, choices=["bpe", "char"])
    ap.add_argument("--alphabet", default=CHAR_ALPHABET, help="alphabet for --vocab-type char")
    ap.add_argument("--arch", default="encoder-decoder", choices=["decoder", "encoder-decoder"],
                    help="decoder: tokens only, like ngxson/fly-llm-hf. "
                         "encoder-decoder: the post is encoded and fed into the ORNs")
    ap.add_argument("--news-encoder", default="minishlab/potion-base-8M",
                    help="a sentence-transformers model, 'hash' for tests or 'none'")
    ap.add_argument("--mode", default=None, choices=["gains", "edges"])
    ap.add_argument("--readout", default=None, help="all | dn | motor | dn+motor | dn+motor+ascending")
    ap.add_argument("--token-input", default=None,
                    help="where the token delay line goes: a superclass list "
                         "(cb_sensory,visual_projection), a group name, or 'sensory' for the whole CNS")
    ap.add_argument("--rank", type=int, default=None, help="readout rank when it is large (0 = full)")
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--delay", type=int, default=None)
    ap.add_argument("--d-emb", type=int, default=None)
    ap.add_argument("--min-syn", type=int, default=None)
    ap.add_argument("--modulatory-sign", type=float, default=None)
    ap.add_argument("--news-group", default=None,
                    help="the post channel; these neurons are kept out of the token input in every "
                         "architecture, so a decoder and an encoder-decoder share one input layout")
    ap.add_argument("--news-mode", default="glomeruli", choices=["direct", "glomeruli"],
                    help="glomeruli: the post becomes a pattern over glomeruli, ORNs of one type share a value")
    ap.add_argument("--vision", default="off", choices=["off", "banner", "glyph", "code"],
                    help="experimental: feed the post as a picture on the photoreceptors")
    ap.add_argument("--vision-width", type=int, default=128)
    ap.add_argument("--vision-height", type=int, default=32)
    ap.add_argument("--vision-zoom", type=float, default=1.0)
    ap.add_argument("--vision-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=32, help="TBPTT window length")
    ap.add_argument("--max-len", type=int, default=320, help="truncate an example to this many tokens")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-edges", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-limit", type=int, default=0, help="use at most this many validation examples")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="validate and checkpoint every N steps, not only at the end of an epoch")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N batches (for smoke tests)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="torch init and the order of the data; the "
                                                        "neuron layout follows the checkpoint when --init-from is used")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device.startswith("mps"):
        raise SystemExit("torch has no sparse CSR matmul on MPS; use --device cpu")
    if args.arch == "decoder":
        args.news_encoder = "none"
    elif args.news_encoder == "none" and args.vision == "off":
        raise SystemExit("--arch encoder-decoder needs an encoder: pass --news-encoder "
                         "(or hash for a smoke test), or use --arch decoder")

    init_ck = None
    if args.init_from:
        init_ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if not args.tokenizer:
            side = os.path.join(os.path.dirname(os.path.abspath(args.init_from)),
                                init_ck.get("tokenizer", "tokenizer.json"))
            if os.path.exists(side):
                args.tokenizer = side
    resolve_arch(args, init_ck["cfg"] if init_ck else None)

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train, val, tok, tok_path = load_data(args)
    tok_sha = file_sha256(tok_path)

    graph = load_graph(args.graph)
    vis_cur = None
    news_emb, news_dim = None, 0
    posts = (train.news or []) + (val.news or []) if train.news is not None else []
    if args.vision != "off":
        if not posts:
            raise SystemExit("--vision needs posts: use a .jsonl file with a 'news' field")
        vis_cur, idx, _ = vision_currents(posts, graph, args.vision, args)
    elif posts:
        news_emb, idx, news_dim = news_embeddings(posts, args.news_encoder, args.out, "cpu")
    else:
        idx = None
    if idx is not None:
        train.news_idx, val.news_idx = idx[:len(train)], idx[len(train):]

    cfg = FlyConfig(vocab_size=tok.get_vocab_size(), d_emb=args.d_emb, delay=args.delay, ticks=args.ticks,
                    mode=args.mode, token_input=args.token_input, readout=args.readout,
                    readout_rank=args.rank, news_dim=news_dim, news_group=args.news_group,
                    news_mode=args.news_mode, min_syn=args.min_syn,
                    modulatory_sign=args.modulatory_sign, vision=args.vision,
                    seed=init_ck["cfg"]["seed"] if init_ck else args.seed)
    model = FlyLM(cfg, graph, token_idx=init_ck.get("token_idx") if init_ck else None).to(device)
    del graph
    print(model.describe())

    if init_ck is not None:
        from nanofly.model import init_from_checkpoint
        report = init_from_checkpoint(model, init_ck, tok_sha)
        print(f"init from {args.init_from}: loaded {len(report['loaded'])} tensors "
              f"({', '.join(report['loaded'][:6])}{', …' if len(report['loaded']) > 6 else ''})")
        if report["fresh"]:
            print(f"  fresh: {', '.join(report['fresh'])}")
        if report["ignored"]:
            print(f"  ignored: {', '.join(report['ignored'])}")

    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    edge_params = [p for n, p in model.named_parameters() if n == "logw"]
    other = [p for n, p in model.named_parameters() if not n.startswith("head.") and n != "logw"]
    groups = [{"params": other, "lr": args.lr, "weight_decay": 0.0},
              {"params": head_params, "lr": args.lr_head, "weight_decay": args.wd}]
    if edge_params:
        groups.append({"params": edge_params, "lr": args.lr_edges, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    chunks_per_epoch = float(np.ceil(np.maximum(train.lengths - 1, 1) / args.seq).sum()) / max(args.batch, 1)
    total_steps = max(1, int(chunks_per_epoch * args.epochs))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    extra = {"tokenizer": "tokenizer.json", "tokenizer_sha": tok_sha,
             "news_encoder": args.news_encoder if news_dim else "none",
             "arch": "encoder_decoder" if news_dim else "decoder", "vision": args.vision,
             "graph": os.path.abspath(args.graph), "init_from": os.path.abspath(args.init_from)
             if args.init_from else "", "args": vars(args)}
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    best = float("inf")

    def checkpoint(epoch, step):
        nonlocal best
        if not len(val):
            save_checkpoint(ckpt_path, model, extra)
            return
        vl = evaluate(model, val, news_emb, args, device, vis_cur)
        mark = ""
        if vl < best:
            best, mark = vl, "  *"
            extra["val_loss"], extra["epoch"], extra["step"] = float(vl), epoch, step
            save_checkpoint(ckpt_path, model, extra)
        print(f"epoch {epoch} step {step}: val loss {vl:.3f}, ppl {math.exp(min(vl, 20)):.1f}{mark}", flush=True)

    step, t0, seen, run_loss, run_tok = 0, time.time(), 0, 0.0, 0
    stop = False
    for epoch in range(args.epochs):
        for b in make_batches(train.lengths, args.batch, args.seed + epoch):
            nb, vb = batch_context(b, news_emb, train.news_idx, vis_cur, device)
            loss_sum, n_tok = run_batch(model, pad_batch(train, b, args.seq), nb, args.seq, device,
                                        train=True, opt=opt, sched=sched, clip=args.clip, vision=vb)
            step += 1
            seen += n_tok
            run_loss += loss_sum
            run_tok += n_tok
            if step % args.log_every == 0:
                dt = time.time() - t0
                done = step / max(total_steps, 1)
                eta = (dt / max(done, 1e-9) - dt) / 60
                print(f"epoch {epoch} step {step}/{total_steps} loss {run_loss / max(run_tok, 1):.3f} "
                      f"{seen / dt:,.0f} tok/s  rho {math.exp(model.log_rho.item()):.3f}  eta {eta:.0f}m",
                      flush=True)
                run_loss, run_tok = 0.0, 0
            if args.eval_every and step % args.eval_every == 0:
                checkpoint(epoch, step)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        checkpoint(epoch, step)
        if stop:
            break
    save_checkpoint(os.path.join(args.out, "last.pt"), model, extra)
    print(f"done in {time.time() - t0:.0f} s, checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
