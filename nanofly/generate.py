"""Sampling from a trained model, one token at a time.

The reservoir is stateful, so there is no way around stepping: every token is one forward pass that
carries the state on. `record=True` additionally returns the activity of every neuron after every
tick, which is what the player in the replyfly repository turns into video; text generation itself
does not need it.
"""
import numpy as np
import torch

from nanofly.model import BOS, EOS


@torch.no_grad()
def generate_one(model, tok, news_vec, prompt, device, *, max_new=60, temperature=0.8, top_k=40,
                 show_topk=5, record=False, vision=None):
    """-> (text, steps, frames).

    `steps` has one entry per forward pass — the prompt is read first (`phase="read"`), then the
    model speaks — carrying the delay-line window, the token in, the token out, and the top-k
    distribution. `frames` is int8 `[n_frames, n_neurons]` when recording and empty otherwise.
    """
    ids = [BOS] + (tok.encode(prompt).ids if prompt else [])
    delay = model.cfg.delay
    state = model.init_state(1, device)
    frames, steps, generated = [], [], []
    pos = 0
    while pos < len(ids):
        tid = ids[pos]
        logits, state, fr = model(torch.tensor([[tid]], device=device), news_vec, state,
                                  record=record, vision=vision)
        f0 = len(frames)
        frames.extend(t[:, 0].clamp(-1, 1).mul(127).round().to(torch.int8).cpu() for t in fr)
        logits = logits[0, -1].float()
        probs = torch.softmax(logits / max(temperature, 1e-4), -1)
        top_p, top_i = probs.topk(min(show_topk, probs.numel()))
        out = None
        phase = "read"
        if pos == len(ids) - 1:
            phase = "speak"
            if top_k > 0:
                kth = probs.topk(min(top_k, probs.numel())).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
            if nxt != EOS and len(generated) < max_new:
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
    stacked = torch.stack(frames).numpy() if frames else np.zeros((0, model.n), np.int8)
    return tok.decode(generated), steps, stacked


def load_for_generation(ckpt, graph_path="", device="cpu"):
    """Checkpoint -> (model, tokenizer, post encoder, checkpoint dict, graph, graph path).

    The graph is not in the checkpoint — only the absolute path it was trained from — so a moved or
    renamed directory has to be pointed at explicitly.
    """
    import os
    from tokenizers import Tokenizer
    from nanofly.encoders import load_news_encoder
    from nanofly.model import load_checkpoint, load_graph

    ck_dir = os.path.dirname(os.path.abspath(ckpt))
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    path = graph_path or meta.get("graph", "")
    if not path or not os.path.exists(path):
        raise SystemExit(f"graph not found ({path or 'no path in the checkpoint'}); pass --graph")
    graph = load_graph(path)
    model, ck = load_checkpoint(ckpt, graph, device)
    model.eval()
    tok = Tokenizer.from_file(os.path.join(ck_dir, ck.get("tokenizer", "tokenizer.json")))
    encoder = load_news_encoder(ck.get("news_encoder", "none"), "cpu") if model.cfg.news_dim else None
    return model, tok, encoder, ck, graph, path


def post_vector(encoder, model, text, device):
    """The constant current the post becomes, or None for a decoder-only model."""
    if encoder is not None and text:
        return torch.from_numpy(encoder.encode([text])).to(device)
    if model.cfg.news_dim:
        return torch.zeros(1, model.cfg.news_dim, device=device)
    return None
