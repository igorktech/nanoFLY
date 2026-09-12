# nanoFLY

A language model whose transformer blocks are replaced by the wiring of a fruit fly. The recurrent
layer is not learned but measured: neurons and synapses from the **MaleCNS v1.0** connectome release
run as a rate model, one line per neuron, `ticks` times per token.

```
x_i <- (1 - a_i) x_i + a_i * tanh( rho * g_i * sum_j W_ij x_j + u_i + b_i )
```

`W_ij` is the sign of the presynaptic transmitter times the synapse count `j -> i`, rows normalised
to unit total weight. Tokens enter the sensory neurons of the head through a delay line; the readout
is a linear head on a subset of neurons.

Two models come out of the same pipeline:

| | input | shipped as |
|---|---|---|
| **decoder** | tokens only | `FlyForCausalLM` in `modeling_fly.py` |
| **encoder-decoder** | tokens + the post, encoded once and fed to the olfactory neurons as a constant current | `ReplyflyForConditionalGeneration` in `modeling_replyfly.py` |

The player that turns a run into video lives in a separate repository,
[replyfly](https://github.com/igorktech/replyfly).

## Install

```bash
pip install -e ".[dev]"
```

`torch`, `numpy` and `tokenizers` are the only hard requirements. `transformers` is needed for the
Hub export, `sentence-transformers` for the post encoder, `pyarrow`/`pandas`/`scipy` to build the
graph from the release tables.

## Quick start, no connectome download

```bash
python prepare_malecns.py --synthetic --out graph_synth     # 20,000 fake neurons
python train.py --graph graph_synth/graph.npz --data data/pairs/pairs_demo.jsonl --out runs/t \
    --arch decoder --vocab 512 --d-emb 64 --delay 4 --max-steps 5
python export_hf.py --ckpt runs/t/ckpt.pt --out hf_out/demo
```

## The real graph

```bash
python prepare_malecns.py --download --raw raw --out graph_cb \
    --keep-superclass cb_sensory,cb_intrinsic,visual_projection,descending_neuron,ascending_neuron
```

Downloads ~1.1 GB of official tables once and writes a 28 MB `graph.npz`; after that only that file
is needed. The subset above is the central brain — 49,393 neurons and 9,055,280 signed edges, which
reproduces [`ngxson/fly-llm-hf`](https://huggingface.co/ngxson/fly-llm-hf) to 59 edges. Drop
`--keep-superclass` for the whole CNS (166,700 neurons, 25.6M connections).

## Export to the Hub

```bash
python export_hf.py --ckpt runs/A/ckpt.pt --out hf_out/nanofly-decoder --push you/nanofly-decoder
```

The folder is self-contained: `config.json` with `auto_map`, `model.safetensors` carrying the
learned weights **and** the connectome buffers, the model code, the tokenizer and a filled-in card.
Before writing, the exported model is checked against the training model on the same tokens; after
writing, the folder is loaded back with `trust_remote_code=True` and checked again.

## What is fly here and what is not

The fly part is the graph: who is wired to whom, how many synapses, and the sign of the
transmitter. The dynamics are a `tanh` rate model, not spikes. In `gains` mode almost all
parameters sit in the readout. Any conclusion needs controls: the same graph with the wiring
shuffled at matched degrees, and a dense baseline with the same parameter count.

## Attribution

Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research.
CC BY 4.0 — the attribution travels with the published weights. Model code: Apache-2.0.
The reservoir idea follows Costi et al. 2025 and `ngxson/fly-llm-hf`; transmitter signs follow
Shiu et al., Nature 2024.
