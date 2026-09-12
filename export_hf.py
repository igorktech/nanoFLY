#!/usr/bin/env python3
"""
Turn a training checkpoint into a self-contained Hugging Face repo folder.

  python export_hf.py --ckpt runs/decoder/ckpt.pt --out hf_out/replyfly-decoder
  python export_hf.py --ckpt runs/sniff/ckpt.pt   --out hf_out/replyfly-sniff \
      --push your-name/replyfly-sniff

What goes in the folder: config.json with auto_map, model.safetensors (learned weights *and* the
connectome buffers, so nothing has to be rebuilt at load time), the two code files, the tokenizer,
and a model card. The code is written once, in `hf/`, and shipped under two names: a checkpoint with
no post channel goes out as `modeling_fly.py` / `FlyForCausalLM`, the way ngxson/fly-llm-hf reads,
and one with the post channel as `modeling_replyfly.py` / `ReplyflyForConditionalGeneration`.
Loading then needs only transformers:

  m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

The connectome buffers are the anatomy: CSR indices plus signed, row-normalised synapse counts.
They are derived from MaleCNS v1.0 (CC BY 4.0), so the card carries the attribution and so does
config.connectome. For `--mode edges` checkpoints the published values are the *learned* strengths
with the anatomical mask and sign — the card says which.

Before writing anything the script runs the exported model and the training model on the same
tokens and compares logits; a mismatch aborts the export. Afterwards the written folder is loaded
back with `trust_remote_code=True` and checked again, so the renamed copy is tested, not assumed.
"""
import argparse
import json
import os
import shutil

import torch

import nanofly.hf
from nanofly.hf.configuration_replyfly import CONNECTOME, ReplyflyConfig
from nanofly.hf.modeling_replyfly import ReplyflyForCausalLM, ReplyflyForConditionalGeneration
from nanofly.model import FlyLM, load_checkpoint, load_graph

HERE = os.path.dirname(os.path.abspath(__file__))
# the shipped code is read from the installed package, not from a folder next to this script
HF_SRC = os.path.dirname(os.path.abspath(nanofly.hf.__file__))
SOURCE_FILES = ["configuration_replyfly.py", "modeling_replyfly.py"]
# How the shipped code is named. The simple model reads like ngxson's — FlyModel in modeling_fly.py —
# the conditional one keeps the project name. Same source either way, see write_code().
NAMES = {
    "decoder": {"prefix": "Fly", "module": "fly", "model_type": "fly"},
    "encoder_decoder": {"prefix": "Replyfly", "module": "replyfly", "model_type": "replyfly"},
}


def write_code(out_dir, variant):
    """Write the model code into the repo folder, renamed for the variant.

    `hf/` is the single source. For the decoder the `Replyfly` prefix becomes `Fly`, the module names
    follow, and the conditional tail (`_hash_embed`, `ForConditionalGeneration`) is cut, so the simple
    repo holds a simple model. A pure rename is easy to get wrong silently, which is why the written
    folder is loaded and re-checked before the export is called done.
    """
    names = NAMES[variant]
    written = []
    for src in SOURCE_FILES:
        with open(os.path.join(HF_SRC, src), encoding="utf-8") as f:
            text = f.read()
        dst = src
        if names["prefix"] != "Replyfly":
            text = text.replace("Replyfly", names["prefix"]).replace("replyfly", names["module"])
            # a top-level definition, so prose mentioning the marker cannot cut the file short
            cut = text.find("\ndef " + "_hash_embed(")
            if cut > 0:
                text = text[:cut].rstrip() + "\n"
            if "modeling" in src and f"class {names['prefix']}ForCausalLM" not in text:
                raise SystemExit(f"write_code cut {src} short: no {names['prefix']}ForCausalLM left")
            dst = src.replace("replyfly", names["module"])
        with open(os.path.join(out_dir, dst), "w", encoding="utf-8") as f:
            f.write(text)
        written.append(dst)
    return written


def check_folder(out_dir, hf, conf, seed=0, batch=2, length=6):
    """Load the folder the way a user would and run it against the model we just exported."""
    from transformers import AutoModelForCausalLM
    loaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(3, conf.vocab_size, (batch, length), generator=g)
    news = torch.randn(batch, conf.news_dim, generator=g) if conf.conditional else None
    if news is not None:
        news = torch.nn.functional.normalize(news, dim=-1)
    with torch.no_grad():
        a = hf(input_ids=tokens, news_embeds=news, use_cache=False).logits
        b = loaded(input_ids=tokens, news_embeds=news, use_cache=False).logits
    return (a - b).abs().max().item(), type(loaded).__name__


def git_remote():
    """Default source link for the card: this clone's origin, without the .git suffix."""
    import subprocess
    try:
        url = subprocess.check_output(["git", "-C", HERE, "config", "--get", "remote.origin.url"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        return url[:-4] if url.endswith(".git") else url
    except Exception:
        return ""


def build_config(model: FlyLM, ck: dict, arch: str, source_repo: str) -> ReplyflyConfig:
    cfg = model.cfg
    head_linear = isinstance(model.head[0], torch.nn.LayerNorm)
    encoder = ck.get("news_encoder", "none") if cfg.news_dim else "none"
    return ReplyflyConfig(
        arch=arch,
        vocab_size=cfg.vocab_size,
        d_emb=cfg.d_emb,
        delay=cfg.delay,
        ticks=cfg.ticks,
        mode=cfg.mode,
        token_input=cfg.token_input,
        n_neurons=model.n,
        n_edges=cfg.graph_e,
        n_token_input=len(model.token_idx),
        n_news_input=len(model.news_idx),
        readout=cfg.readout,
        readout_size=len(model.readout_idx),
        readout_rank=cfg.readout_rank,
        head_type="linear" if head_linear else "lowrank",
        news_dim=cfg.news_dim,
        news_group=cfg.news_group,
        n_reserved=getattr(model, "n_reserved", 0),
        news_mode=cfg.news_mode,
        news_glom=cfg.news_glom,
        news_encoder=encoder,
        news_prefix="query: " if "e5" in str(encoder).lower() else "",
        min_syn=cfg.min_syn,
        modulatory_sign=cfg.modulatory_sign,
        connectome=CONNECTOME,
        source_repo=source_repo,
        torch_dtype="float32",
    )


def transfer(model: FlyLM, hf, conf: ReplyflyConfig):
    """Copy learned parameters and freeze the connectome into persistent buffers."""
    b = hf.brain
    with torch.no_grad():
        b.emb.weight.copy_(model.emb.weight)
        b.in_proj.copy_(model.in_proj)
        b.bias.copy_(model.bias)
        b.gain.copy_(model.gain)
        b.log_rho.copy_(model.log_rho)
        b.leak_logit.copy_(model.leak_logit)
        if b.news_proj is not None:
            b.news_proj.weight.copy_(model.news_proj.weight)
            b.news_proj.bias.copy_(model.news_proj.bias)
            if conf.news_mode == "glomeruli":
                b.news_glom_of.copy_(model.news_glom_of)
        hf.head.load_state_dict(model.head.state_dict())

        idx = torch.int32 if max(model.n, conf.n_edges) < 2**31 else torch.int64
        b.crow = model.crow.to(idx)
        b.col = model.col.to(idx)
        b.w = model._values().detach().clone()
        b.token_idx = model.token_idx.long()
        b.news_idx = model.news_idx.long()
        b.readout_idx = model.readout_idx.long()
        bounds = [s for s, _ in model.group_bounds] + [model.group_bounds[-1][1]]
        b.bounds = torch.tensor(bounds, dtype=torch.long)


@torch.no_grad()
def check(model: FlyLM, hf, conf: ReplyflyConfig, seed=0, batch=2, length=6):
    """Same tokens through both models; logits must agree."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(3, conf.vocab_size, (batch, length), generator=g)
    news = torch.randn(batch, conf.news_dim, generator=g) if conf.conditional else None
    if news is not None:
        news = torch.nn.functional.normalize(news, dim=-1)
    ref, _, _ = model(tokens, news=news)
    out = hf(input_ids=tokens, news_embeds=news, use_cache=False).logits
    diff = (ref - out).abs().max().item()
    scale = ref.abs().max().item()

    # and once more through generate()-style stepping, to prove the cache carries the state
    cache, step_logits = None, []
    for t in range(length):
        o = hf(input_ids=tokens[:, t:t + 1], news_embeds=news, cache_params=cache, use_cache=True)
        cache = o.cache_params
        step_logits.append(o.logits)
    step_diff = (ref - torch.cat(step_logits, 1)).abs().max().item()
    return diff, step_diff, scale


CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- connectome
- drosophila
- reservoir
- nanofly
{extra_tags}---

# {name}

A language model whose recurrent layer is not learned but measured: it is the wiring of a fruit fly.
{n_neurons:,} neurons and {n_edges:,} connections from the **MaleCNS v1.0** connectome release run
as a rate model, {ticks} tick(s) of dynamics per token, and a linear head reads {readout_size:,}
neurons ({readout}) to produce logits.

{arch_line}

## Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
{usage}
```

## What is real here and what is not

* **Real**: which neuron talks to which, how many synapses, and the sign of the presynaptic
  transmitter (ACh +, GABA / glutamate / histamine −, following Shiu et al., Nature 2024).
  Modulatory and unknown transmitters use `modulatory_sign={modulatory_sign}`.
* **Not real**: this is a rate model with `tanh`, not spikes; synaptic strengths are proxied by
  synapse counts with rows normalised to unit total weight; membrane and synaptic time constants
  are a single learned leak per neuron. Nothing here was fitted to fly electrophysiology.
* **Learned**: {learned}.
* The fly did not learn English. The connectome supplies a fixed high-dimensional recurrent map;
  the head learns to read it.

## Honest baseline

A connectome model only means something against controls. The one that matters is the same graph
with the wiring shuffled at matched in-degree, trained identically. Unless that comparison is
published next to the model, treat "the fly writes" as a demo, not a result.

## Training

| | |
|---|---|
| architecture | `{arch}` |
| mode | `{mode}` ({learned_short}) |
| delay line | {delay} slots, {n_token_input:,} sensory neurons carry tokens |
| post channel | {reserved_line} |
| ticks per token | {ticks} |
| vocabulary | {vocab_size:,} ({tokenizer_kind}) |
| trainable parameters | {n_params_m:.1f}M |
| data | {data} |
| epochs | {epochs} |
| validation loss | {val_loss} |

## Attribution and licence

Connectome: {connectome} The published buffers are derived from that release, so keep the
attribution when you redistribute the weights. Model code: Apache-2.0. Source and training
pipeline: {source_repo}
"""

USAGE_DECODER = """ids = torch.tensor([[model.config.bos_token_id]])
out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.9, top_k=40)
print(tok.decode(out[0], skip_special_tokens=True))"""

USAGE_COND = """# the post is encoded once and injected into the olfactory neurons as a constant current
news = model.encode_posts("Google released the full connectome of a fly")
ids = torch.tensor([[model.config.bos_token_id]])
out = model.generate(ids, news_embeds=news, max_new_tokens=60, do_sample=True, temperature=0.9, top_k=40)
print(tok.decode(out[0], skip_special_tokens=True))"""


def _encoder_link(name):
    """Hub ids get a link; `hash` is the built-in test encoder and has nowhere to point."""
    return f"[`{name}`](https://huggingface.co/{name})" if "/" in str(name) else f"`{name}`"


def write_card(path, conf: ReplyflyConfig, ck: dict, hf, repo, name):
    args = ck.get("args", {}) or {}
    learned = ("the input projection, per-neuron gain, bias and leak, and the readout head"
               if conf.mode == "gains" else
               "the strength of every synapse (sign and mask frozen), plus input, gains, leak and head")
    arch_line = (
        "This is the **decoder-only** model: tokens in, tokens out, nothing conditions it."
        if conf.arch == "decoder" else
        f"This is the **encoder-decoder** model. The post being answered is encoded once by the frozen "
        f"sentence encoder {_encoder_link(conf.news_encoder)} and injected as a "
        f"constant current into {conf.n_news_input:,} olfactory neurons"
        + (f", pooled into {conf.news_glom} glomeruli" if conf.news_mode == "glomeruli" else "")
        + ", so the fly smells the post for the whole generation. The encoder is not part of these "
          "weights; `encode_posts()` downloads it on first use."
    )
    text = CARD.format(
        name=name, repo=repo, arch=conf.arch, mode=conf.mode, ticks=conf.ticks, delay=conf.delay,
        n_neurons=conf.n_neurons, n_edges=conf.n_edges, readout=conf.readout,
        readout_size=conf.readout_size, n_token_input=conf.n_token_input,
        vocab_size=conf.vocab_size, modulatory_sign=conf.modulatory_sign,
        tokenizer_kind=args.get("vocab_type", "bpe"), learned=learned,
        learned_short="connectome frozen" if conf.mode == "gains" else "synapse strengths trained",
        n_params_m=sum(p.numel() for p in hf.parameters()) / 1e6,
        data=os.path.basename(args.get("data", "not recorded")),
        epochs=args.get("epochs", "—"),
        val_loss=f"{ck['val_loss']:.3f}" if "val_loss" in ck else "not recorded",
        connectome=conf.connectome, source_repo=conf.source_repo or "not recorded",
        usage=USAGE_COND if conf.conditional else USAGE_DECODER,
        arch_line=arch_line,
        extra_tags="- replyfly\n" if conf.conditional else "",
        reserved_line=(f"{conf.n_news_input:,} olfactory neurons, {conf.news_glom} glomeruli"
                       if conf.conditional else
                       f"none — the {conf.n_reserved:,} olfactory neurons are held out of the token "
                       "input so the encoder-decoder variant can start from these weights"),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_tokenizer(out_dir, ck_dir, ck, conf):
    src = os.path.join(ck_dir, ck.get("tokenizer", "tokenizer.json"))
    if not os.path.exists(src):
        print(f"  ! tokenizer.json not found next to the checkpoint ({src}), skipping")
        return False
    shutil.copy(src, os.path.join(out_dir, "tokenizer.json"))
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>",
        "clean_up_tokenization_spaces": False,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({"pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"}, f, indent=1)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", default="", help="defaults to the path recorded in the checkpoint")
    ap.add_argument("--out", required=True, help="folder to write the Hugging Face repo into")
    ap.add_argument("--name", default="", help="model name for the card, defaults to the folder name")
    ap.add_argument("--repo", default="", help="repo id used in the card examples and by --push")
    ap.add_argument("--source-repo", default=git_remote(),
                    help="link to the training code, goes into the card and config")
    ap.add_argument("--tolerance", type=float, default=2e-4, help="max allowed logit difference")
    ap.add_argument("--push", default="", help="push to this repo id (needs huggingface_hub login)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ck_dir = os.path.dirname(os.path.abspath(args.ckpt))
    ck_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    graph_path = args.graph or ck_meta.get("graph", "")
    if not graph_path or not os.path.exists(graph_path):
        raise SystemExit(f"graph not found ({graph_path or 'no path in checkpoint'}); pass --graph")
    print(f"graph: {graph_path}")
    graph = load_graph(graph_path)
    model, ck = load_checkpoint(args.ckpt, graph, "cpu")
    model.eval()
    del graph

    vis = ck.get("vision", "off") or model.cfg.vision
    if vis != "off":
        raise SystemExit(f"this checkpoint was trained with --vision {vis}, and the shipped model has "
                         "no photoreceptor input: the export would silently drop the post channel")
    if str(ck.get("arch", "")).replace("-", "_") == "encoder_decoder" and not model.cfg.news_dim:
        raise SystemExit("encoder-decoder checkpoint with no post channel; nothing to export")
    arch = ck.get("arch") or ("encoder_decoder" if model.cfg.news_dim else "decoder")
    arch = "encoder_decoder" if arch.replace("-", "_") == "encoder_decoder" else "decoder"
    name = args.name or os.path.basename(os.path.abspath(args.out))
    repo = args.push or args.repo or name
    conf = build_config(model, ck, arch, args.source_repo)
    variant = "encoder_decoder" if conf.conditional else "decoder"
    prefix, module = NAMES[variant]["prefix"], NAMES[variant]["module"]
    conf.auto_map = {
        "AutoConfig": f"configuration_{module}.{prefix}Config",
        "AutoModel": f"modeling_{module}.{prefix}Model",
        "AutoModelForCausalLM": (f"modeling_{module}.{prefix}ForConditionalGeneration"
                                 if conf.conditional else f"modeling_{module}.{prefix}ForCausalLM"),
    }

    cls = ReplyflyForConditionalGeneration if conf.conditional else ReplyflyForCausalLM
    hf = cls(conf).eval()
    transfer(model, hf, conf)

    diff, step_diff, scale = check(model, hf, conf)
    print(f"check: max |Δlogit| = {diff:.2e} in one pass, {step_diff:.2e} stepping with the cache "
          f"(logit scale {scale:.2f})")
    if max(diff, step_diff) > args.tolerance:
        raise SystemExit("export does not reproduce the training model, nothing written")

    os.makedirs(args.out, exist_ok=True)
    hf.save_pretrained(args.out, safe_serialization=True)
    code_files = write_code(args.out, variant)
    cfg_path = os.path.join(args.out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg_json = json.load(f)
    # save_pretrained stamps the class attribute; the shipped class is the renamed one
    cfg_json["model_type"] = NAMES[variant]["model_type"]
    cfg_json["architectures"] = [conf.auto_map["AutoModelForCausalLM"].split(".")[-1]]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2, sort_keys=True)
    write_tokenizer(args.out, ck_dir, ck, conf)
    write_card(os.path.join(args.out, "README.md"), conf, ck, hf, repo, name)

    loaded_diff, loaded_cls = check_folder(args.out, hf, conf)
    print(f"folder check: {loaded_cls} from {', '.join(code_files)}, max |Δlogit| = {loaded_diff:.2e}")
    if loaded_diff > args.tolerance:
        raise SystemExit("the written folder does not reproduce the exported model")

    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
               if os.path.isfile(os.path.join(args.out, f)))
    print(f"written: {args.out}, {size / 1e6:.0f} MB")
    print(f"  {conf.n_neurons:,} neurons, {conf.n_edges:,} connections, "
          f"{sum(p.numel() for p in hf.parameters()) / 1e6:.1f}M trainable parameters")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push,
                          commit_message=f"nanofly {arch}, {conf.n_neurons:,} neurons")
        print(f"pushed: https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
