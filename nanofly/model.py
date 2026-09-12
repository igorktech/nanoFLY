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
import math
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
    def __init__(self, cfg: FlyConfig, graph: dict):
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n = len(graph["nt"])
        cfg.graph_n = n

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

        sensory = token_population(graph, cfg.token_input)
        news_idx = np.zeros(0, dtype=np.int64)
        if cfg.news_dim > 0:
            news_idx = group(graph, cfg.news_group)
            if len(news_idx) < cfg.news_min_neurons:
                k = max(cfg.news_min_neurons, len(sensory) // 7)
                news_idx = np.sort(rng.choice(sensory, size=min(k, len(sensory) // 2), replace=False))
        token_idx = np.setdiff1d(sensory, news_idx)
        if len(token_idx) < cfg.delay:
            raise ValueError("too few sensory neurons for the token input")
        token_idx = rng.permutation(token_idx)
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


def save_checkpoint(path, model, extra):
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, graph, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlyConfig.from_dict(ck["cfg"])
    model = FlyLM(cfg, graph)
    if model.cfg.graph_e != ck["cfg"].get("graph_e", model.cfg.graph_e):
        raise ValueError("the graph does not match the one the model was trained on (different edge count)")
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck
