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
import random
import signal
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from nanofly.data import (CHAR_ALPHABET, batch_context, data_sha256, file_sha256, load_data, make_batches,
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


def run_batch(model, ids, news, seq_len, device, train=True, opt=None, sched=None, clip=1.0, vision=None,
              state=None, start=0, on_update=None):
    """One batch, truncated backpropagation through time: the reservoir state carries across the
    `seq_len` windows, the optimiser steps on each one, and only the state crosses the boundary."""
    ids = ids.to(device)
    B, L = ids.shape
    state = model.init_state(B, device) if state is None else state
    total, count = 0.0, 0
    for c in range(start, L - 1, seq_len):
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
        if train and on_update is not None:
            if not on_update(c + seq_len, state, loss.item() * n_tok, n_tok):
                break
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


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_updates(lengths, batch, seq, seed, epochs):
    """Count actual nonempty TBPTT windows, including partial document batches."""
    return sum((int(max(lengths[b])) - 2) // seq + 1
               for epoch in range(epochs) for b in make_batches(lengths, batch, seed + epoch))


def resume_arguments(ap, args, ck):
    """Restore the training recipe; only paths and operational controls may change on resume."""
    runtime = {"graph", "data", "out", "device", "resume", "max_steps", "log_every",
               "eval_every", "save_every"}
    explicit = {ap._option_string_actions[arg.split("=")[0]].dest
                for arg in sys.argv[1:] if arg.split("=")[0] in ap._option_string_actions}
    if "training_state" not in ck:
        ap.error("this is a weights-only checkpoint; use --init-from, not --resume")
    for key, value in ck["args"].items():
        if key in {"log_every", "eval_every", "save_every"} and key not in explicit:
            setattr(args, key, value)
        if key in runtime or key == "init_from":
            continue
        if key in explicit and getattr(args, key) != value:
            ap.error(f"--resume cannot change {key}; use --init-from for a new training recipe")
        setattr(args, key, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--data", required=True, help="a prepared directory or a .jsonl/.txt file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default="",
                    help="checkpoint to start from; the architecture and tokenizer come from it")
    ap.add_argument("--resume", default="", help="resume latest.pt, including optimizer and TBPTT cursor")
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
    ap.add_argument("--max-len", type=int, default=0, help="0 keeps full documents; positive values explicitly truncate")
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
                    help="validate every N optimizer updates (also at epoch end)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="save resumable latest.pt every N optimizer updates; 0 disables periodic saves")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop at N total optimizer updates; 0 runs all epochs")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="torch init and the order of the data; the "
                                                        "neuron layout follows the checkpoint when --init-from is used")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.resume and args.init_from:
        ap.error("--resume and --init-from are mutually exclusive")
    init_ck = None
    checkpoint_source = args.resume or args.init_from
    if checkpoint_source:
        init_ck = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
        if args.resume:
            resume_arguments(ap, args, init_ck)
        if args.resume or not args.tokenizer:
            args.tokenizer = os.path.join(os.path.dirname(os.path.abspath(checkpoint_source)),
                                          init_ck.get("tokenizer", "tokenizer.json"))
            if not os.path.exists(args.tokenizer):
                ap.error(f"checkpoint tokenizer is missing: {args.tokenizer}")
    resolve_arch(args, init_ck["cfg"] if init_ck else None)
    if min(args.batch, args.seq, args.epochs, args.log_every) < 1:
        ap.error("batch, seq, epochs and log-every must be positive")
    if args.max_len and args.max_len < 3:
        ap.error("max-len must be 0 or at least 3")
    if min(args.warmup, args.max_steps, args.save_every, args.eval_every, args.limit, args.val_limit) < 0:
        ap.error("limits and intervals must be nonnegative")
    if not 0 <= args.val_frac < 1:
        ap.error("val-frac must be in [0, 1)")

    if args.device.startswith("mps"):
        raise SystemExit("torch has no sparse CSR matmul on MPS; use --device cpu")
    if args.arch == "decoder":
        args.news_encoder = "none"
    elif args.news_encoder == "none" and args.vision == "off":
        raise SystemExit("--arch encoder-decoder needs an encoder: pass --news-encoder "
                         "(or hash for a smoke test), or use --arch decoder")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train, val, tok, tok_path = load_data(args)
    tok_sha = file_sha256(tok_path)
    print("checking dataset content identity...", flush=True)
    dataset_sha = data_sha256(args.data)
    if args.resume and dataset_sha != init_ck["training_state"]["data_sha"]:
        raise ValueError("--resume dataset content changed; use the original data or --init-from")

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
    if args.resume:
        saved_cfg = FlyConfig.from_dict(init_ck["cfg"])
        if cfg.news_dim != saved_cfg.news_dim:
            raise ValueError("--resume post encoder dimensions changed")
        cfg = saved_cfg
    model = FlyLM(cfg, graph, token_idx=init_ck.get("token_idx") if init_ck else None).to(device)
    del graph
    print(model.describe())

    if init_ck is not None:
        from nanofly.model import init_from_checkpoint
        report = init_from_checkpoint(model, init_ck, tok_sha)
        print(f"init from {checkpoint_source}: loaded {len(report['loaded'])} tensors "
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
    total_steps = optimizer_updates(train.lengths, args.batch, args.seq, args.seed, args.epochs)

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
    latest_path = os.path.join(args.out, "latest.pt")
    best, last_eval_step = float("inf"), -1
    progress = {"epoch": 0, "batch_index": 0, "window_start": 0, "state": None,
                "batch_step": 0, "optimizer_step": 0, "tokens_seen": 0}
    if args.resume:
        saved = init_ck["training_state"]
        if saved["total_steps"] != total_steps:
            raise ValueError("--resume schedule horizon changed")
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        progress = dict(saved["progress"])
        if progress["state"] is not None:
            progress["state"] = tuple(t.to(device) for t in progress["state"])
        best, last_eval_step = saved["best"], saved["last_eval_step"]
        restore_rng(saved["rng"])
        del saved
        print(f"resumed at update {progress['optimizer_step']}, epoch {progress['epoch']}, "
              f"batch {progress['batch_index']}, token offset {progress['window_start']}")
    # Release the CPU checkpoint copy (including Adam tensors) after restoring it.
    init_ck = None

    def weights_extra():
        return {**extra, "epoch": progress["epoch"], "step": progress["optimizer_step"],
                "batch_step": progress["batch_step"], "tokens_seen": progress["tokens_seen"]}

    def save_latest():
        cursor = dict(progress)
        if cursor["state"] is not None:
            cursor["state"] = tuple(t.detach().cpu() for t in cursor["state"])
        save_checkpoint(latest_path, model, {**weights_extra(), "training_state": {
            "version": 1, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "rng": rng_state(), "progress": cursor, "best": best,
            "last_eval_step": last_eval_step, "total_steps": total_steps, "data_sha": dataset_sha}})

    def validate():
        nonlocal best, last_eval_step
        update = progress["optimizer_step"]
        if last_eval_step == update:
            return
        last_eval_step = update
        if not len(val):
            save_checkpoint(ckpt_path, model, weights_extra())
            return
        vl = evaluate(model, val, news_emb, args, device, vis_cur)
        mark = ""
        if vl < best:
            best, mark = vl, "  *"
            save_checkpoint(ckpt_path, model, {**weights_extra(), "val_loss": float(vl)})
        print(f"epoch {progress['epoch']} update {update}: val loss {vl:.3f}, "
              f"ppl {math.exp(min(vl, 20)):.1f}{mark}", flush=True)

    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True  # Finish the current optimizer update before serializing its cursor.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    t0 = time.time()
    start_update, start_tokens = progress["optimizer_step"], progress["tokens_seen"]
    run_loss, run_tok = 0.0, 0
    stopping = stopping or bool(args.max_steps and start_update >= args.max_steps)
    # Ensure there is a recovery point even before the first periodic save.
    save_latest()
    for epoch in range(progress["epoch"], args.epochs):
        if stopping:
            break
        batches = make_batches(train.lengths, args.batch, args.seed + epoch)
        for batch_index in range(progress["batch_index"], len(batches)):
            b = batches[batch_index]
            nb, vb = batch_context(b, news_emb, train.news_idx, vis_cur, device)
            ids = pad_batch(train, b, args.seq)

            def on_update(next_window, state, loss_sum, n_tok):
                nonlocal stopping, run_loss, run_tok
                progress["optimizer_step"] += 1
                progress["tokens_seen"] += n_tok
                run_loss += loss_sum
                run_tok += n_tok
                if next_window >= ids.shape[1] - 1:
                    progress["batch_step"] += 1
                    progress.update(batch_index=batch_index + 1, window_start=0, state=None)
                    if batch_index + 1 == len(batches):
                        progress.update(epoch=epoch + 1, batch_index=0)
                else:
                    progress.update(batch_index=batch_index, window_start=next_window, state=state)
                update = progress["optimizer_step"]
                if update % args.log_every == 0:
                    dt = time.time() - t0
                    target = min(total_steps, args.max_steps) if args.max_steps else total_steps
                    eta = dt / max(update - start_update, 1) * max(target - update, 0) / 60
                    print(f"epoch {epoch} update {update}/{total_steps} batch {progress['batch_step']} "
                          f"tokens {progress['tokens_seen']:,} loss {run_loss / max(run_tok, 1):.3f} "
                          f"{(progress['tokens_seen'] - start_tokens) / max(dt, 1e-9):,.0f} tok/s "
                          f"rho {math.exp(model.log_rho.item()):.3f} eta {eta:.0f}m", flush=True)
                    run_loss, run_tok = 0.0, 0
                eval_due = args.eval_every and update % args.eval_every == 0
                if eval_due or (args.save_every and update % args.save_every == 0):
                    save_latest()
                if eval_due:
                    validate()
                    save_latest()
                if args.max_steps and update >= args.max_steps:
                    stopping = True
                return not stopping

            run_batch(model, ids, nb, args.seq, device, opt=opt, sched=sched,
                      clip=args.clip, vision=vb, state=progress["state"],
                      start=progress["window_start"], on_update=on_update)
            if stopping:
                break
        # Save before validation too: a long validation pass must not delay recovery.
        save_latest()
        validate()
        save_latest()
        if stopping:
            break
    save_latest()
    save_checkpoint(os.path.join(args.out, "last.pt"), model, weights_extra())
    print(f"done in {time.time() - t0:.0f} s, best {ckpt_path}, resume {latest_path}")


if __name__ == "__main__":
    main()
