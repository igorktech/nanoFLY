#!/usr/bin/env python3
"""
Generate a reply to a post and record the activity of every neuron for the player.

Examples:
  python generate.py --ckpt runs/gains/ckpt.pt --news "Google released the full connectome of a fly"
  python generate.py --ckpt runs/gains/ckpt.pt --news-file headlines.txt --out player/runs

Each post gets a folder with the files player/index.html expects:
  run.json      text, generation steps, top-k probabilities, frame-to-token mapping
  frames.bin    int8 [frames x neurons], activity after every tick (x * 127)
  neurons.bin   float32 xyz [n,3], then uint8 superclass [n], then uint8 group flags [n]
  neurons.json  superclass names, meaning of the flags, DN indices
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import warnings

import numpy as np
import torch

from nanofly.encoders import load_news_encoder
from nanofly.model import BOS, EOS, FLAG_BITS, load_checkpoint, load_graph

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")

ATTRIBUTION = ("Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, "
               "Google Research. CC BY 4.0.")


def copy_shell(out_dir, graph_path):
    """The shell, if make_shell.py built one next to the graph."""
    src = os.path.dirname(os.path.abspath(graph_path))
    found = []
    for name in ("shell.json", "shell.bin"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, name))
            found.append(name)
    return len(found) == 2


def write_neurons(out_dir, graph, model):
    n = model.n
    pos = graph["pos"].astype(np.float32)
    sc = graph["superclass"].astype(np.uint8)
    flags = model.neuron_flags().astype(np.uint8)
    src = graph["pos_source"].astype(np.uint8) if "pos_source" in graph else np.zeros(n, np.uint8)
    hop = graph["hop"].astype(np.uint8) if "hop" in graph else np.full(n, 255, np.uint8)
    with open(os.path.join(out_dir, "neurons.bin"), "wb") as f:
        f.write(pos.tobytes())
        f.write(sc.tobytes())
        f.write(flags.tobytes())
        f.write(src.tobytes())
        f.write(hop.tobytes())
    dn = model.dn_indices()
    dn = dn[np.argsort(pos[dn, 0])] if len(dn) else dn
    meta = {
        "n": int(n),
        "superclass_names": [str(s) for s in graph["superclass_names"]],
        "flag_bits": FLAG_BITS,
        "pos_sources": ["soma", "root", "pos", "graph", "random", "synapse_mean"],
        "measured": int((src < 3).sum()),
        "has_hop": bool((hop != 255).any()),
        "dn": dn.astype(int).tolist(),
        "edges": int(model.cfg.graph_e),
        "attribution": ATTRIBUTION,
    }
    with open(os.path.join(out_dir, "neurons.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)


@torch.no_grad()
def generate_one(model, tok, news_vec, prompt, args, device, vision=None):
    ids = [BOS] + (tok.encode(prompt).ids if prompt else [])
    delay = model.cfg.delay
    state = model.init_state(1, device)
    frames, steps, generated = [], [], []
    pos = 0
    while pos < len(ids):
        tid = ids[pos]
        logits, state, fr = model(torch.tensor([[tid]], device=device), news_vec, state, record=True, vision=vision)
        f0 = len(frames)
        frames.extend(t[:, 0].clamp(-1, 1).mul(127).round().to(torch.int8).cpu() for t in fr)
        logits = logits[0, -1].float()
        probs = torch.softmax(logits / max(args.temperature, 1e-4), -1)
        top_p, top_i = probs.topk(args.show_topk)
        out = None
        phase = "read"
        if pos == len(ids) - 1:
            phase = "speak"
            if args.top_k > 0:
                kth = probs.topk(min(args.top_k, probs.numel())).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
            if nxt != EOS and len(generated) < args.max_new:
                ids.append(nxt)
                generated.append(nxt)
                out = nxt
        window = ids[max(0, pos - delay + 1):pos + 1]
        steps.append({
            "frame_start": f0, "frame_end": len(frames), "phase": phase,
            "window": [tok.decode([i]) if i > EOS else "" for i in window],
            "input": tok.decode([tid]) if tid > EOS else "",
            "output": tok.decode([out]) if out is not None else "",
            "text": tok.decode(generated),
            "topk": [[tok.decode([int(i)]), round(float(p), 4)] for p, i in zip(top_p, top_i)],
        })
        pos += 1
    return tok.decode(generated), steps, torch.stack(frames).numpy() if frames else np.zeros((0, model.n), np.int8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", default="", help="defaults to the path recorded in the checkpoint")
    ap.add_argument("--news", action="append", default=[], help="the post to answer (repeatable)")
    ap.add_argument("--news-file", default="", help="file with one post per line")
    ap.add_argument("--prompt", default="", help="start of the reply")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--show-topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="player/runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ck_dir = os.path.dirname(os.path.abspath(args.ckpt))
    ck_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    graph_path = args.graph or ck_meta.get("graph", "")
    graph = load_graph(graph_path)
    model, ck = load_checkpoint(args.ckpt, graph, device)
    model.eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ck_dir, ck.get("tokenizer", "tokenizer.json")))
    encoder = load_news_encoder(ck.get("news_encoder", "none"), "cpu") if model.cfg.news_dim else None
    vis_mode = ck.get("vision", "off")
    retina = None
    if vis_mode != "off":
        from nanofly.vision import TextRetina, embedding_pattern, eye_layout
        lay = eye_layout(graph)
        a = ck.get("args", {})
        retina = (lay, TextRetina(lay, mode=vis_mode if vis_mode != "code" else "banner",
                                  width=a.get("vision_width", 128), height=a.get("vision_height", 32),
                                  zoom=a.get("vision_zoom", 1.0), scale=a.get("vision_scale", 1.0)))
        if vis_mode == "code":
            from nanofly.encoders import load_news_encoder as _enc
            encoder = _enc(ck.get("news_encoder", "hash") or "hash", "cpu")

    headlines = list(args.news)
    if args.news_file:
        with open(args.news_file, encoding="utf-8") as f:
            headlines += [l.strip() for l in f if l.strip()]
    if not headlines:
        headlines = [""]

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    for h in headlines:
        news_vec = None
        if encoder is not None and h:
            news_vec = torch.from_numpy(encoder.encode([h])).to(device)
        elif model.cfg.news_dim:
            news_vec = torch.zeros(1, model.cfg.news_dim, device=device)
        vision = None
        if retina is not None:
            lay, r = retina
            if vis_mode == "code":
                from nanofly.vision import embedding_pattern
                vec = encoder.encode([h])[0] if h else np.zeros(384, np.float32)
                cur = embedding_pattern(vec, lay["xy"])
            else:
                cur = r.current(h, 0, 1)[0]
            vision = torch.from_numpy(cur[None].astype(np.float32)).to(device)
            news_vec = None
        t0 = time.time()
        text, steps, frames = generate_one(model, tok, news_vec, args.prompt, args, device, vision)
        dt = time.time() - t0
        slug = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.md5(h.encode()).hexdigest()[:6]
        run_dir = os.path.join(args.out, slug)
        os.makedirs(run_dir, exist_ok=True)
        write_neurons(run_dir, graph, model)
        has_shell = copy_shell(run_dir, graph_path)
        frames.astype(np.int8).tofile(os.path.join(run_dir, "frames.bin"))
        run = {
            "headline": h, "prompt": args.prompt, "text": text,
            "n_frames": int(frames.shape[0]), "n_neurons": int(model.n), "ticks": model.cfg.ticks,
            "temperature": args.temperature, "top_k": args.top_k,
            "model": {"mode": model.cfg.mode, "readout": model.cfg.readout, "edges": model.cfg.graph_e,
                      "delay": model.cfg.delay},
            "steps": steps, "attribution": ATTRIBUTION,
        }
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False)
        index.append({"dir": slug, "headline": h, "text": text})
        size = frames.nbytes / 1e6
        shell_note = "" if has_shell else ", no shell (build one with make_shell.py)"
        print(f"\n{h}\n  fly: {text}\n  {len(steps)} steps, {frames.shape[0]} frames, "
              f"{size:.1f} MB{shell_note}, {dt:.1f} s -> {run_dir}")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
