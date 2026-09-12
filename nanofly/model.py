"""
nanoFLY: a language model whose transformer blocks are replaced by recurrent dynamics on the
MaleCNS v1.0 connectome of a male fruit fly.

Dynamics (rate model, one line per neuron i):
    x_i <- (1 - a_i) x_i + a_i * tanh( rho * g_i * sum_j W_ij x_j + u_i + b_i )
W_ij = sign of the presynaptic transmitter of j * synapse count j->i, rows normalised to sum |W| = 1.

Modes:
  gains  the graph is frozen as a reservoir; input, gains g, leaks a, bias and readout are trained
  edges  additionally trains the strength of every synapse (sign and mask frozen, Dale's law)

Tokens enter the sensory neurons through a delay line of `delay` groups, the post is fed as a
constant current into `news_group` (olfactory ORNs by default), and the output is read from the
`readout` neurons (all, dn, motor, dn+motor, dn+motor+ascending).
"""
import hashlib
import math
import warnings
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn

# Transmitter sign: ACh +1, GABA and glutamate -1 (as in Shiu et al. 2024), histamine -1 (in the fly it
# opens chloride channels; otherwise the photoreceptors are cut off from the network), modulatory amines
# and unknowns get `modulatory_sign` (0 by default, which drops those edges).
NT_SIGN = {"ach": 1.0, "gaba": -1.0, "glu": -1.0, "his": -1.0}

PAD, BOS, EOS = 0, 1, 2
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]

FLAG_BITS = {"token_input": 1, "news_input": 2, "readout": 4, "descending": 8,
             "motor": 16, "mushroom_body": 32, "ascending": 64}


@dataclass
class FlyConfig:
    vocab_size: int = 2048
    d_emb: int = 256
    delay: int = 8
    ticks: int = 2
    mode: str = "gains"
    token_input: str = "cb_sensory"
    readout: str = "all"
    readout_rank: int = 256
    news_dim: int = 0
    news_group: str = "orn"
    news_mode: str = "direct"
    news_glom: int = 0
    vision: str = "off"
    vision_n: int = 0
    news_min_neurons: int = 64
    rho: float = 1.0
    leak_init: float = 0.5
    input_scale: float = 1.0
    min_syn: int = 1
    modulatory_sign: float = 0.0
    edge_chunk: int = 4_000_000
    seed: int = 0
    graph_n: int = 0
    graph_e: int = 0
    graph_sha: str = ""

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def load_graph(path):
    g = np.load(path)
    return {k: g[k] for k in g.files}


def neuron_signs(graph, modulatory_sign):
    names = [str(s) for s in graph["nt_names"]]
    table = np.array([NT_SIGN.get(nm, modulatory_sign) for nm in names], dtype=np.float32)
    return table[graph["nt"]]


def group(graph, name):
    key = f"grp_{name}"
    return graph[key].astype(np.int64) if key in graph else np.zeros(0, dtype=np.int64)


def superclass_idx(graph, spec):
    """Neurons whose superclass is in the comma-separated `spec` (case-insensitive)."""
    if "superclass_names" not in graph:
        return np.zeros(0, dtype=np.int64)
    names = [str(s).strip().lower() for s in graph["superclass_names"]]
    want = {s.strip().lower() for s in str(spec).split(",") if s.strip()}
    codes = [i for i, nm in enumerate(names) if nm in want]
    if not codes:
        return np.zeros(0, dtype=np.int64)
    return np.where(np.isin(graph["superclass"], codes))[0].astype(np.int64)


def token_population(graph, spec, min_neurons=64):
    """Where the delay line writes the tokens.

    `sensory` takes every sensory neuron of the CNS at once — head, retina and legs — which is
    convenient and anatomically meaningless. The default `cb_sensory` keeps the token stream on the
    sensory surface of the head, as in ngxson/fly-llm-hf, and leaves the retina to `--vision` and the
    ORNs to the post. Anything else is read as a superclass list first and as a group name second.
    """
    if spec in ("sensory", "all", ""):
        return group(graph, "sensory")
    idx = superclass_idx(graph, spec)
    if len(idx) == 0:
        idx = group(graph, spec)
    if len(idx) < min_neurons:
        fallback = group(graph, "sensory")
        print(f"token input '{spec}': only {len(idx)} neurons, falling back to all {len(fallback)} sensory")
        return fallback
    return idx


class _Struct:
    """Sparse matrix indices for the autograd function (references to the model buffers)."""
    __slots__ = ("n", "crow", "col", "crow_t", "col_t", "perm_t", "row", "chunk")


class SparseRecurrent(torch.autograd.Function):
    """y = W x, W in CSR (rows = postsynaptic neurons). Gradients for x and for the values of W."""

    @staticmethod
    def forward(ctx, values, x, S):
        W = torch.sparse_csr_tensor(S.crow, S.col, values, (S.n, S.n))
        y = W @ x
        ctx.save_for_backward(values, x)
        ctx.S = S
        return y

    @staticmethod
    def backward(ctx, gy):
        values, x = ctx.saved_tensors
        S = ctx.S
        gy = gy.contiguous()
        g_values = g_x = None
        if ctx.needs_input_grad[1]:
            Wt = torch.sparse_csr_tensor(S.crow_t, S.col_t, values[S.perm_t], (S.n, S.n))
            g_x = Wt @ gy
        if ctx.needs_input_grad[0]:
            g_values = _sddmm(S, values, gy, x)
        return g_values, g_x, None


_SAMPLED_OK = {}


def _sddmm(S, values, gy, x):
    """dL/dW_e = sum_b gy[post_e, b] * x[pre_e, b], only on edges that exist."""
    dev = values.device.type
    if _SAMPLED_OK.get(dev, True):
        try:
            base = torch.sparse_csr_tensor(S.crow, S.col, torch.zeros_like(values), (S.n, S.n))
            out = torch.sparse.sampled_addmm(base, gy, x.t().contiguous(), beta=0.0)
            _SAMPLED_OK[dev] = True
            return out.values()
        except (RuntimeError, NotImplementedError):
            _SAMPLED_OK[dev] = False
    g = torch.empty_like(values)
    for s in range(0, values.numel(), S.chunk):
        e = min(s + S.chunk, values.numel())
        g[s:e] = (gy.index_select(0, S.row[s:e]) * x.index_select(0, S.col[s:e].long())).sum(1)
    return g

class FlyLM(nn.Module):
    def __init__(self, cfg: FlyConfig, graph: dict, token_idx=None):
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        token_idx_override = token_idx
        n = len(graph["nt"])
        cfg.graph_n = n

        cfg.graph_sha = graph_sha(graph)
        sign = neuron_signs(graph, cfg.modulatory_sign)
        pre, post, cnt = graph["pre"].astype(np.int64), graph["post"].astype(np.int64), graph["count"].astype(np.float64)
        keep = cnt >= cfg.min_syn
        w = sign[pre] * cnt
        keep &= w != 0
        self.dropped_edges = int((~keep).sum())
        pre, post, w = pre[keep], post[keep], w[keep]
        rowsum = np.bincount(post, weights=np.abs(w), minlength=n)
        w = w / np.maximum(rowsum[post], 1e-12)
        order = np.lexsort((pre, post))
        pre, post, w = pre[order], post[order], w[order].astype(np.float32)
        e = len(w)
        cfg.graph_e = e
        crow = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=n))]).astype(np.int64)
        order_t = np.lexsort((post, pre))
        crow_t = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(np.int64)
        idx_dtype = torch.int32 if e < 2**31 and n < 2**31 else torch.int64
        self.register_buffer("crow", torch.from_numpy(crow).to(idx_dtype), persistent=False)
        self.register_buffer("col", torch.from_numpy(pre).to(idx_dtype), persistent=False)
        self.register_buffer("row", torch.from_numpy(post), persistent=False)
        self.register_buffer("crow_t", torch.from_numpy(crow_t).to(idx_dtype), persistent=False)
        self.register_buffer("col_t", torch.from_numpy(post[order_t]).to(idx_dtype), persistent=False)
        self.register_buffer("perm_t", torch.from_numpy(order_t), persistent=False)
        self.register_buffer("w0", torch.from_numpy(w), persistent=False)
        self.n = n

        if cfg.mode == "edges":
            self.register_buffer("edge_sign", torch.from_numpy(np.sign(w)), persistent=False)
            self.logw = nn.Parameter(torch.from_numpy(np.log(np.abs(w))))
        elif cfg.mode != "gains":
            raise ValueError(f"unknown mode {cfg.mode}")

        # The post channel is carved out of the sensory population whether or not this model uses
        # it, so a decoder and an encoder-decoder built on the same graph get the same token input
        # and the same `in_proj` layout — which is what lets one initialise the other. Anything
        # that consumes `rng` before the permutation below re-maps every row of `in_proj`, so do
        # not insert draws here without invalidating old checkpoints.
        sensory = token_population(graph, cfg.token_input)
        reserved = group(graph, cfg.news_group)
        if len(reserved) < cfg.news_min_neurons:
            k = max(cfg.news_min_neurons, len(sensory) // 7)
            reserved = np.sort(rng.choice(sensory, size=min(k, len(sensory) // 2), replace=False))
        news_idx = reserved if cfg.news_dim > 0 else np.zeros(0, dtype=np.int64)
        token_idx = np.setdiff1d(sensory, reserved)
        if len(token_idx) < cfg.delay:
            raise ValueError("too few sensory neurons for the token input")
        token_idx = rng.permutation(token_idx)
        if token_idx_override is not None:
            override = np.asarray(token_idx_override, dtype=np.int64)
            if not np.array_equal(np.sort(override), np.sort(token_idx)):
                raise ValueError("the stored token population is not the one this graph and config "
                                 "produce; --graph, --token-input, --news-group and the graph seed "
                                 "must match the checkpoint")
            token_idx = override
        self.n_reserved = len(reserved)
        bounds = np.linspace(0, len(token_idx), cfg.delay + 1).astype(int)
        self.group_bounds = [(int(bounds[j]), int(bounds[j + 1])) for j in range(cfg.delay)]

        ro = cfg.readout.split("+")
        names = {"dn": "descending", "motor": "motor", "ascending": "ascending"}
        if ro == ["all"]:
            readout_idx = np.arange(n)
        else:
            readout_idx = np.unique(np.concatenate([group(graph, names[r]) for r in ro]))
        if len(readout_idx) == 0:
            raise ValueError(f"readout group '{cfg.readout}' is empty")

        self.register_buffer("token_idx", torch.from_numpy(token_idx), persistent=False)
        self.register_buffer("news_idx", torch.from_numpy(news_idx), persistent=False)
        photo = group(graph, "photoreceptor")
        if cfg.vision != "off" and len(photo) == 0:
            raise ValueError("the graph has no photoreceptor group, visual input is impossible")
        cfg.vision_n = len(photo)
        self.register_buffer("photo_idx", torch.from_numpy(photo), persistent=False)
        self.register_buffer("readout_idx", torch.from_numpy(readout_idx), persistent=False)
        self._flags = self._make_flags(graph, token_idx, news_idx, readout_idx)
        self._dn = group(graph, "descending")

        d = cfg.d_emb
        self.emb = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.in_proj = nn.Parameter(torch.randn(len(token_idx), d) * (cfg.input_scale / math.sqrt(d)))
        self.news_proj = None
        self.news_scatter = None
        if cfg.news_dim > 0:
            if cfg.news_mode == "glomeruli":
                glom = graph["glom"].astype(np.int64) if "glom" in graph else np.full(n, -1)
                g_of_news = glom[news_idx]
                keep = g_of_news >= 0
                if keep.sum() < cfg.news_min_neurons:
                    raise ValueError("ORN glomeruli are not labelled in the graph, use --news-mode direct")
                used = np.unique(g_of_news[keep])
                remap = {int(g): i for i, g in enumerate(used)}
                self.register_buffer("news_idx", torch.from_numpy(news_idx[keep]), persistent=False)
                self.register_buffer("news_glom_of", torch.tensor([remap[int(g)] for g in g_of_news[keep]]),
                                     persistent=False)
                cfg.news_glom = len(used)
                self.news_proj = nn.Linear(cfg.news_dim, len(used))
                self.news_scatter = True
            else:
                self.news_proj = nn.Linear(cfg.news_dim, len(news_idx))
        self.bias = nn.Parameter(torch.zeros(n, 1))
        self.gain = nn.Parameter(torch.ones(n, 1))
        self.log_rho = nn.Parameter(torch.tensor(math.log(cfg.rho)))
        self.leak_logit = nn.Parameter(torch.full((n, 1), math.log(cfg.leak_init / (1 - cfg.leak_init))))
        n_ro = len(readout_idx)
        if cfg.readout_rank > 0 and n_ro * cfg.vocab_size > 8_000_000:
            self.head = nn.Sequential(nn.Linear(n_ro, cfg.readout_rank, bias=False),
                                      nn.LayerNorm(cfg.readout_rank), nn.Linear(cfg.readout_rank, cfg.vocab_size))
        else:
            self.head = nn.Sequential(nn.LayerNorm(n_ro), nn.Linear(n_ro, cfg.vocab_size))

    def _make_flags(self, graph, token_idx, news_idx, readout_idx):
        f = np.zeros(self.n, dtype=np.uint8)
        f[token_idx] |= FLAG_BITS["token_input"]
        f[news_idx] |= FLAG_BITS["news_input"]
        f[readout_idx] |= FLAG_BITS["readout"]
        f[group(graph, "descending")] |= FLAG_BITS["descending"]
        f[group(graph, "motor")] |= FLAG_BITS["motor"]
        for g in ("kc", "mbon", "pam", "ppl1"):
            f[group(graph, g)] |= FLAG_BITS["mushroom_body"]
        f[group(graph, "ascending")] |= FLAG_BITS["ascending"]
        return f

    def neuron_flags(self):
        return self._flags

    def dn_indices(self):
        return self._dn

    def _struct(self):
        S = _Struct()
        S.n, S.crow, S.col, S.row = self.n, self.crow, self.col, self.row
        S.crow_t, S.col_t, S.perm_t, S.chunk = self.crow_t, self.col_t, self.perm_t, self.cfg.edge_chunk
        return S

    def _values(self):
        if self.cfg.mode == "edges":
            return self.edge_sign * torch.exp(self.logw)
        return self.w0

    def init_state(self, batch, device=None):
        device = device or self.w0.device
        x = torch.zeros(self.n, batch, device=device)
        hist = torch.full((batch, self.cfg.delay), PAD, dtype=torch.long, device=device)
        return x, hist

    def _token_input(self, hist):
        e = self.emb(hist)
        parts = [e[:, j] @ self.in_proj[s:t].t() for j, (s, t) in enumerate(self.group_bounds)]
        return torch.cat(parts, dim=1).t()

    def forward(self, tokens, news=None, state=None, record=False, vision=None):
        """tokens [B,T] -> logits [B,T,V]. news [B, news_dim] or None, vision [B, vision_n] is the current
        on the photoreceptors. state = (x [n,B], hist [B,delay]).
        record=True also returns every neuron after every tick (a list of [n,B] tensors)."""
        B, T = tokens.shape
        if state is None:
            state = self.init_state(B, tokens.device)
        x, hist = state
        S = self._struct()
        vals = self._values()
        scale = torch.exp(self.log_rho) * self.gain
        a = torch.sigmoid(self.leak_logit)
        news_u = None
        if self.news_proj is not None and news is not None:
            news_u = self.news_proj(news)
            if self.news_scatter:
                news_u = news_u.index_select(1, self.news_glom_of)
            news_u = news_u.t()
        vis_u = vision.t().to(x.dtype) if vision is not None else None
        logits, frames = [], []
        for t in range(T):
            hist = torch.cat([tokens[:, t:t + 1], hist[:, :-1]], dim=1)
            u = torch.zeros(self.n, B, device=x.device, dtype=x.dtype)
            u = u.index_add(0, self.token_idx, self._token_input(hist))
            if news_u is not None:
                u = u.index_add(0, self.news_idx, news_u)
            if vis_u is not None:
                u = u.index_add(0, self.photo_idx, vis_u)
            u = u + self.bias
            for _ in range(self.cfg.ticks):
                rec = SparseRecurrent.apply(vals, x, S)
                x = (1 - a) * x + a * torch.tanh(scale * rec + u)
                if record:
                    frames.append(x.detach())
            h = x.index_select(0, self.readout_idx).t()
            logits.append(self.head(h))
        return torch.stack(logits, 1), (x, hist), frames

    def describe(self):
        groups = {"token input": len(self.token_idx), "post input": len(self.news_idx),
                  "reserved": self.n_reserved,
                  "photoreceptors": len(self.photo_idx), "readout": len(self.readout_idx), "DN": len(self._dn)}
        if self.cfg.news_glom:
            groups["glomeruli"] = self.cfg.news_glom
        n_params = {name: p.numel() for name, p in self.named_parameters()}
        lines = [f"neurons {self.n:,}, active edges {self.cfg.graph_e:,} (dropped {self.dropped_edges:,}: "
                 f"modulatory/unknown transmitter or the min_syn threshold)",
                 "groups: " + ", ".join(f"{k} {v:,}" for k, v in groups.items()),
                 f"mode {self.cfg.mode}, token input {self.cfg.token_input}, "
                 f"ticks per token {self.cfg.ticks}, delay line {self.cfg.delay}"
                 + (f", vision {self.cfg.vision}" if self.cfg.vision != "off" else ""),
                 "trainable parameters: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in n_params.items()),
                 f"total {sum(n_params.values()) / 1e6:.2f}M"]
        return "\n".join(lines)


def graph_sha(graph):
    """Identity of the wiring itself. Edge counts alone cannot tell two graphs apart."""
    h = hashlib.sha256()
    for key in ("pre", "post", "count", "nt"):
        h.update(np.ascontiguousarray(graph[key]).tobytes())
    return h.hexdigest()


def token_idx_sha(token_idx):
    """Identity of the input layout: which neuron holds which row of `in_proj`, in order."""
    a = token_idx.detach().cpu().numpy() if torch.is_tensor(token_idx) else np.asarray(token_idx)
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def save_checkpoint(path, model, extra):
    """Parameters only: every graph buffer is `persistent=False` and is rebuilt from `--graph`. The
    token layout travels with the weights because it is a numpy permutation, and numpy does not
    promise a stable stream across versions."""
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict(),
                "token_idx": model.token_idx.cpu().numpy(),
                "token_idx_sha": token_idx_sha(model.token_idx), **extra}, path)


def load_checkpoint(path, graph, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlyConfig.from_dict(ck["cfg"])
    model = FlyLM(cfg, graph, token_idx=ck.get("token_idx"))
    for key, label in (("graph_e", "edge count"), ("graph_n", "neuron count"), ("graph_sha", "wiring")):
        want = ck["cfg"].get(key)
        if want and getattr(model.cfg, key) != want:
            raise ValueError(f"the graph does not match the one the model was trained on ({label})")
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


# Everything below the architecture line has to agree for one checkpoint to initialise another;
# `ticks` only changes the dynamics, so it is a warning rather than a refusal.
MUST_MATCH = ["vocab_size", "d_emb", "delay", "mode", "readout", "readout_rank"]


def init_from_checkpoint(model, ck, tokenizer_sha=None):
    """Load the parameters of one model into another — a decoder into an encoder-decoder, typically.

    Refuses anything that would quietly produce a different brain: another graph, another input
    layout, another vocabulary. Returns which tensors were taken, which stayed freshly initialised
    and which were left behind.
    """
    cfg = ck["cfg"]
    for key, label in (("graph_n", "neuron count"), ("graph_e", "edge count"), ("graph_sha", "wiring")):
        want = cfg.get(key)
        if want and getattr(model.cfg, key) != want:
            raise ValueError(f"--init-from: the checkpoint was trained on a different graph ({label})")
    if ck.get("token_idx_sha") and ck["token_idx_sha"] != token_idx_sha(model.token_idx):
        raise ValueError("--init-from: different token population; --graph, --token-input, "
                         "--news-group and the graph seed must match the checkpoint")
    if tokenizer_sha and ck.get("tokenizer_sha") and ck["tokenizer_sha"] != tokenizer_sha:
        raise ValueError("--init-from: different tokenizer; the embedding and the head are bound to "
                         "the vocabulary, pass --tokenizer from the checkpoint's run")
    for key in MUST_MATCH:
        if cfg.get(key) is not None and cfg[key] != getattr(model.cfg, key):
            raise ValueError(f"--init-from: {key} is {cfg[key]} in the checkpoint, "
                             f"{getattr(model.cfg, key)} here")
    if cfg.get("ticks") != model.cfg.ticks:
        warnings.warn(f"--init-from: ticks {cfg.get('ticks')} -> {model.cfg.ticks}; the weights load "
                      "but the dynamics they were trained for are different")

    own, sd = model.state_dict(), ck["state_dict"]
    bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    if bad:
        raise ValueError(f"--init-from: shape mismatch on {bad}")
    take = {k: v for k, v in sd.items() if k in own}
    missing = model.load_state_dict(take, strict=False).missing_keys
    return {"loaded": sorted(take), "fresh": sorted(missing), "ignored": sorted(set(sd) - set(own))}
