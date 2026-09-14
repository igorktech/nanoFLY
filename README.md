# nanoFLY

A language model whose transformer blocks are replaced by the wiring of a fruit fly. The recurrent
layer is not learned but measured: neurons and synapses from the **MaleCNS v1.0** connectome release
run as a rate model, one line per neuron, `ticks` times per token.

```
x_i <- (1 - a_i) x_i + a_i * tanh( rho * g_i * sum_j W_ij x_j + u_i + b_i )
```

`W_ij` is the sign of the presynaptic transmitter times the synapse count `j -> i`, rows normalised
to unit total weight. Tokens enter the sensory neurons of the head through a delay line — slot `j`
receives the embedding of token `t-j` — and a linear head reads a subset of neurons. There is no
attention and no positional encoding; order comes from the recurrence and from the delay line.

Two models come out of the same pipeline:

| | input | published as |
|---|---|---|
| **decoder** | tokens only | `FlyForCausalLM` in `modeling_fly.py` |
| **encoder-decoder** | tokens + the post, encoded once and held on the olfactory neurons as a constant current | `ReplyflyForConditionalGeneration` in `modeling_replyfly.py` |

The second starts from the weights of the first: the olfactory neurons are kept out of the token
input in **both**, so the two share an input layout and `--init-from` can carry a pretrained decoder
straight into the conditional model.

The player that turns a run into video lives in a separate repository,
[replyfly](https://github.com/igorktech/replyfly).

## Install

```bash
pip install -e ".[dev]"
```

`torch`, `numpy` and `tokenizers` are the only hard requirements. The extras: `hf` for the Hub
export, `encoder` for the post encoder, `data` to build the graph and read compressed corpora.
Training needs CUDA or CPU — torch has no sparse CSR matmul on MPS.

## Two minutes, no downloads

```bash
tests/smoke.sh
```

Builds a synthetic 20,000-neuron connectome, trains both models, exports both, loads them back the
way a user would and samples. Everything below is the same path on real data.

## The graph

```bash
python prepare_malecns.py --download --raw raw --out graph_cb \
    --keep-superclass cb_sensory,cb_intrinsic,visual_projection,descending_neuron,ascending_neuron
```

Downloads ~1.1 GB of official tables once and writes a 28 MB `graph.npz` — after that only that file
is needed, and it is what you copy to a rented GPU. The subset above is the central brain: 49,393
neurons and 9,055,280 signed edges, reproducing
[`ngxson/fly-llm-hf`](https://huggingface.co/ngxson/fly-llm-hf) to 59 edges out of nine million.
Drop `--keep-superclass` for the whole CNS — 166,700 neurons and 25,582,938 connections — and see
`summary.json` for what the filters kept.

## Data

Two shapes. A **prepared directory** (`train.bin` / `val.bin` of uint16 ids, offsets, tokenizer,
`meta.json`) for anything large, read back as a memmap; a **JSONL file** of `{"news": …, "text": …}`
for post/reply pairs, tokenised at start-up.

```bash
python data/shakespeare/prepare.py                       # 1.1 MB, the dev set
python data/tinystories/prepare.py --limit 10000         # 10 MB downloaded, 2.7M tokens
python data/text/prepare.py --hf-file some/dataset:file.jsonl.zst --field text \
    --limit 5000 --chunk 320 --vocab 4096 --out data/mine
```

`data/text/prepare.py` is the general one — the other two are wrappers. It reads `.txt`, `.jsonl`,
`.jsonl.zst` and `.parquet`, from a local path, a URL or a Hub file. JSONL and paragraph text are
streamed up to `--limit` documents; Hub parquet files are downloaded before their rows are streamed.
Preparation spools text to temporary files under `--out`, so allow disk space for the unpacked text
as well as the token files. RAM depends on the largest document and the bounded tokenizer sample,
not the total corpus size. BPE uses a seeded reservoir of snippets of up to 2,048 characters, capped
by both `--tok-limit` documents and `--tok-chars` characters (20 million by default).

Repeat `--local`, `--url`, or `--hf-file` to combine sources under one tokenizer. For a weighted
mixture, use `--mix mix.json` with a document budget (`--limit`):

```json
[
  {"local": "raw/news.jsonl.zst", "field": "text", "weight": 2},
  {"local": "raw/pikabu.jsonl.zst", "field": "text", "weight": 1, "limit": 500000}
]
```

```bash
python data/text/prepare.py --mix mix.json --limit 1000000 --vocab 4096 --out data/russian
```

Weights are relative probabilities per **document**, not token proportions. Sampling is without
replacement; exhausted sources drop out and the remaining weights are renormalized. With no global
limit, every source is consumed up to its own limit, so weights change order rather than final counts.
Paths in a manifest are relative to the working directory. A seeded content hash assigns documents
to train/validation before chunking, keeping exact duplicates in the same split. This does not remove
duplicates or detect near-duplicate news. `--val-limit` discards excess held-out documents.

By default whole documents are preserved. `--chunk N` makes independent examples of at most N tokens
including BOS/EOS, covering all body tokens but resetting context between chunks. During training,
`--max-len 0` (the default) keeps whole examples and carries state across TBPTT windows. A positive
`--max-len` explicitly truncates examples and prints a warning with the number of discarded tokens.
Chunking trades document context for smaller padded batches; TBPTT bounds the gradient window in
either case.

Prepared data must include `meta.json` with its tokenizer SHA-256. A different tokenizer, including
one inherited through `--init-from`, is rejected even when vocabulary sizes match. To fine-tune on
a new prepared corpus, prepare it with `--tokenizer runs/A/tokenizer.json` first.

## Train

```bash
# A: the decoder, pretrained on stories
python train.py --graph graph_cb/graph.npz --data data/tinystories --out runs/A \
    --arch decoder --epochs 20 --batch 64 --eval-every 500

# B: the encoder-decoder, starting from A's weights
python train.py --graph graph_cb/graph.npz --data data/pairs/pairs.jsonl --out runs/B \
    --arch encoder-decoder --init-from runs/A/ckpt.pt --epochs 3
```

`--init-from` takes the architecture, the tokenizer and the neuron layout from the checkpoint and
refuses to start if any of them would differ — a different graph, a different token population or a
different vocabulary is an error, not a silent mismatch. Only the post projection is new.

Training is truncated backpropagation through time: the reservoir state carries across `--seq`
windows and the optimiser steps on each one. `--warmup`, `--max-steps`, `--eval-every`, `--log-every`
and `--save-every` all count **optimizer updates**, including on resume. Logs separately report
completed document batches and predicted tokens. The scheduler horizon counts the actual TBPTT
windows in the length-bucketed batches, including partial batches, over `--epochs`; `--max-steps`
is an optional stopping point within that schedule, not a replacement schedule horizon.

`ckpt.pt` holds the best validation weights (latest evaluated weights when validation is empty).
`last.pt` holds weights at a clean exit. `latest.pt` is the resumable checkpoint, saved atomically
every `--save-every` updates (500 by default), at validation boundaries, and at a clean stop. It
includes Adam and scheduler state, RNG states, counters, dataset identity, the next data cursor,
and the recurrent state and token history when stopped inside a batch. SIGINT/SIGTERM request a
stop after the current optimizer update. A hard kill can lose work since the last completed save.

```bash
python train.py --graph graph_cb/graph.npz --data data/tinystories --out runs/A \
    --resume runs/A/latest.pt
```

Resume restores the original training recipe. Paths, device, stopping limit and logging/checkpoint
intervals may be changed; changing the model, batch/sequence lengths, epochs or learning-rate recipe
requires `--init-from` instead. Dataset contents are hashed at startup and must match on resume.
Logging/checkpoint intervals are inherited unless overridden. Omit `--max-steps` on resume to
finish the original epoch budget, or pass a new absolute optimizer-update stopping point.
Keep the run's tokenizer and best checkpoint with `latest.pt`. Exact numerical replay is tested on
CPU; CUDA reproducibility also depends on the kernels, hardware and runtime. Older weights-only
checkpoints remain usable with `--init-from`, but cannot reconstruct a previous optimizer state.

Run focused offline regressions with `python -m unittest discover -s tests -v`, and the complete
training/export/loading pipeline with `tests/smoke.sh`.

## Sample

```bash
python sample.py --ckpt runs/B/ckpt.pt --news "Janelia published MaleCNS v1.0"
```

## Publish

```bash
python export_hf.py --ckpt runs/A/ckpt.pt --out hf_out/nanofly-decoder --push you/nanofly-decoder
```

The folder is self-contained: `config.json` with `auto_map`, `model.safetensors` carrying the
learned weights **and** the connectome buffers, the model code, the tokenizer and a filled-in card.
Nothing is rebuilt at load time and nothing is downloaded except the repository itself.

The model code is written once, in `nanofly/hf/`, and shipped under two names: the decoder gets
`configuration_fly.py` / `modeling_fly.py` with the `Fly*` classes and `model_type: fly`, the
conditional model keeps `Replyfly*`. Both are generated from that one source at export time, so they
cannot drift apart. Before writing, the exported model is checked against the training model on the
same tokens; after writing, the folder is loaded back with `trust_remote_code=True` and checked
again, so the renamed copy is tested rather than assumed.

## Knobs

- `--arch decoder|encoder-decoder`.
- `--mode gains|edges`. In `gains` the graph is frozen and input, gains, leaks, bias and head are
  trained. In `edges` the strength of every synapse joins them, while the sign and the mask stay.
- `--readout all|dn|motor|dn+motor|dn+motor+ascending`. For `all` the head is low-rank (`--rank`).
- `--token-input` — which population the delay line writes into. `cb_sensory,visual_projection` is
  ngxson's 14,069 sensory-facing neurons; `sensory` takes every sensory neuron of the CNS at once,
  head, retina and legs, which is convenient and anatomically meaningless.
- `--news-group orn` — the post channel. These neurons are held out of the token input in every
  architecture, which is what makes `--init-from` possible.
- `--news-mode glomeruli` pools the post into a pattern over glomeruli, so ORNs of one type share a
  value, the way a smell actually arrives. `direct` gives every ORN its own current.
- `--delay`, `--ticks`, `--vocab-type char`, `--modulatory-sign` — see `train.py --help`.

## What is fly here and what is not

The fly part is the graph: who is wired to whom, how many synapses, and the sign of the
transmitter. Understanding the post is done by a sentence encoder; the dynamics are a `tanh` rate
model, not spikes; and in `gains` mode almost all parameters sit in the readout. Any conclusion
needs controls — the same graph with the wiring shuffled at matched degrees, and a dense baseline of
the same parameter count. Until those are run and published next to the weights, "the fly writes" is
a demo, not a result.

## Attribution

Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research.
CC BY 4.0 — the attribution travels with the published weights. Model code: Apache-2.0.
The reservoir idea follows Costi et al. 2025 and `ngxson/fly-llm-hf`; transmitter signs follow
Shiu et al., Nature 2024.
