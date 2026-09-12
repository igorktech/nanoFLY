"""Replyfly: a fruit-fly connectome as the recurrent layer of a language model.

State update, one line per neuron i, repeated `ticks` times per token:

    x_i <- (1 - a_i) * x_i + a_i * tanh( rho * g_i * sum_j W_ij x_j + u_i + b_i )

`W` is anatomy and is not trained here: signed, row-normalised synapse counts from MaleCNS v1.0.
`u` is the input current — the token embedding fanned into sensory neurons through a delay line,
plus, for the encoder-decoder model, a constant current on the olfactory neurons that carries the
post being answered. The readout is a linear head on a subset of neurons (all, descending, motor…).

This file is the inference/fine-tuning copy that ships with the weights. Training lives in the
project repo (see `config.source_repo`), where the connectome is rebuilt from the release tables.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_replyfly import ReplyflyConfig


class ReplyflyCache:
    """What carries over between generate() steps: neuron state and the delay line of token ids."""

    is_compileable = False

    def __init__(self, state, last_tokens, seq_len=0):
        self.state = state              # [B, N] neuron activations
        self.last_tokens = last_tokens  # [B, delay], most recent token first
        self.seq_len = seq_len

    def get_seq_length(self, layer_idx=0):
        return self.seq_len

    def get_max_cache_shape(self):
        return None

    def reorder_cache(self, beam_idx):
        self.state = self.state.index_select(0, beam_idx.to(self.state.device))
        self.last_tokens = self.last_tokens.index_select(0, beam_idx.to(self.last_tokens.device))
        return self


@dataclass
class ReplyflyOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_params: Optional[ReplyflyCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ReplyflyCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[ReplyflyCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class _SpMM(torch.autograd.Function):
    """y = W @ x with W a fixed sparse CSR matrix; the backward pass reuses a cached transpose."""

    @staticmethod
    def forward(ctx, x, W, WT):
        ctx.WT = WT
        return torch.sparse.mm(W, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.WT, g.contiguous()), None, None


class ReplyflyPreTrainedModel(PreTrainedModel):
    config_class = ReplyflyConfig
    base_model_prefix = "brain"
    supports_gradient_checkpointing = False
    _is_stateful = True
    _no_split_modules = []

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # the model keeps its own ReplyflyCache; generate() must not build a KV cache for it
        return False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=min(0.02, module.in_features ** -0.5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ReplyflyModel):
            # the raw parameters need their own init: a zeroed in_proj means no current ever reaches
            # the neurons, so a model built from the config alone would be deaf. A reservoir wants
            # O(1) drive, not transformer-scale.
            nn.init.normal_(module.in_proj, std=module.config.d_emb ** -0.5)
            nn.init.ones_(module.gain)
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.leak_logit)      # sigmoid(0) = 0.5
            nn.init.zeros_(module.log_rho)         # exp(0) = 1


class ReplyflyModel(ReplyflyPreTrainedModel):
    """The brain: connectome recurrence, token and post input, neuron-subset readout."""

    def __init__(self, config: ReplyflyConfig):
        super().__init__(config)
        N, E, d = config.n_neurons, config.n_edges, config.d_emb
        self.emb = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        self.in_proj = nn.Parameter(torch.empty(config.n_token_input, d))
        self.bias = nn.Parameter(torch.zeros(N, 1))
        self.gain = nn.Parameter(torch.ones(N, 1))
        self.log_rho = nn.Parameter(torch.zeros(()))
        self.leak_logit = nn.Parameter(torch.zeros(N, 1))
        self.news_proj = None
        if config.conditional:
            out = config.news_glom if config.news_mode == "glomeruli" else config.n_news_input
            self.news_proj = nn.Linear(config.news_dim, out)
            if config.news_mode == "glomeruli":
                self.register_buffer("news_glom_of", torch.zeros(config.n_news_input, dtype=torch.long))

        # connectome in CSR by target neuron: crow [N+1], col = source neuron, w = signed strength
        self.register_buffer("crow", torch.zeros(N + 1, dtype=torch.int32))
        self.register_buffer("col", torch.zeros(E, dtype=torch.int32))
        self.register_buffer("w", torch.zeros(E))
        self.register_buffer("token_idx", torch.zeros(config.n_token_input, dtype=torch.long))
        self.register_buffer("news_idx", torch.zeros(config.n_news_input, dtype=torch.long))
        self.register_buffer("readout_idx", torch.zeros(config.readout_size, dtype=torch.long))
        # delay line: slot j covers token_idx[bounds[j]:bounds[j+1]] and sees token t-j
        self.register_buffer("bounds", torch.zeros(config.delay + 1, dtype=torch.long))
        self._wt_key = None
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def connectome(self):
        N = self.config.n_neurons
        W = torch.sparse_csr_tensor(self.crow, self.col, self.w, size=(N, N))
        key = (self.w.data_ptr(), self.w._version, self.w.device)
        if self._wt_key != key:
            self._wt = W.t().to_sparse_csr()
            self._wt_key = key
        return W, self._wt

    def drive_from_tokens(self, prev_ids, inputs_embeds):
        """Delay-line current: input group j receives the embedding of token t-j. -> [T, n_in, B]"""
        k = self.config.delay
        B, T, _ = inputs_embeds.shape
        ext = torch.cat([self.emb(prev_ids.flip(1)), inputs_embeds], dim=1)   # oldest first, then new
        bounds = self.bounds.tolist()
        parts = []
        for j in range(k):
            e_j = ext[:, k - j: k - j + T]                                    # token t-j for t in [0, T)
            parts.append(e_j @ self.in_proj[bounds[j]:bounds[j + 1]].t())
        return torch.cat(parts, dim=-1).permute(1, 2, 0).contiguous()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        news_embeds=None,
        cache_params: Optional[ReplyflyCache] = None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        cfg = self.config
        use_cache = use_cache if use_cache is not None else cfg.use_cache
        return_dict = return_dict if return_dict is not None else getattr(cfg, "return_dict", True)
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)
        B, T, _ = inputs_embeds.shape
        N, k, dev = cfg.n_neurons, cfg.delay, inputs_embeds.device

        if cache_params is None:
            x = inputs_embeds.new_zeros(N, B)
            prev_ids = torch.full((B, k), cfg.pad_token_id, dtype=torch.long, device=dev)
            seq_len = 0
        else:
            x = cache_params.state.t().contiguous()
            prev_ids = cache_params.last_tokens
            seq_len = cache_params.seq_len

        drive = self.drive_from_tokens(prev_ids, inputs_embeds)
        news_u = None
        if self.news_proj is not None and news_embeds is not None:
            news_u = self.news_proj(news_embeds.to(inputs_embeds.dtype))
            if cfg.news_mode == "glomeruli":
                news_u = news_u.index_select(1, self.news_glom_of)
            news_u = news_u.t()                                               # [n_news_input, B]

        W, WT = self.connectome()
        a = torch.sigmoid(self.leak_logit)
        scale = torch.exp(self.log_rho) * self.gain
        mask = attention_mask.to(x.dtype).t() if attention_mask is not None else None
        outs = []
        for t in range(T):
            u = torch.zeros(N, B, device=dev, dtype=x.dtype)
            u = u.index_add(0, self.token_idx, drive[t])
            if news_u is not None:
                u = u.index_add(0, self.news_idx, news_u)
            u = u + self.bias
            for _ in range(cfg.ticks):
                new = (1 - a) * x + a * torch.tanh(scale * _SpMM.apply(x, W, WT) + u)
                if mask is not None:                                          # padding freezes the state
                    m = mask[t][None, :]
                    new = m * new + (1 - m) * x
                x = new
            outs.append(x.index_select(0, self.readout_idx))
        hidden = torch.stack(outs, dim=0).permute(2, 0, 1)                    # [B, T, readout_size]

        cache = None
        if use_cache:
            ids = prev_ids if input_ids is None else torch.cat([input_ids.flip(1), prev_ids], dim=1)[:, :k]
            cache = ReplyflyCache(x.t().contiguous(), ids, seq_len + T)
        if not return_dict:
            return tuple(v for v in [hidden, cache, (hidden,) if output_hidden_states else None] if v is not None)
        return ReplyflyOutput(last_hidden_state=hidden, cache_params=cache,
                              hidden_states=(hidden,) if output_hidden_states else None)


def _build_head(config: ReplyflyConfig):
    if config.head_type == "lowrank":
        return nn.Sequential(
            nn.Linear(config.readout_size, config.readout_rank, bias=False),
            nn.LayerNorm(config.readout_rank),
            nn.Linear(config.readout_rank, config.vocab_size),
        )
    return nn.Sequential(nn.LayerNorm(config.readout_size), nn.Linear(config.readout_size, config.vocab_size))


class ReplyflyForCausalLM(ReplyflyPreTrainedModel, GenerationMixin):
    """Decoder-only: tokens in, tokens out. `news_embeds` is accepted but is None for this arch."""

    def __init__(self, config: ReplyflyConfig):
        super().__init__(config)
        self.brain = ReplyflyModel(config)
        self.head = _build_head(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.brain.emb

    def set_input_embeddings(self, value):
        self.brain.emb = value

    def get_output_embeddings(self):
        return self.head[-1]

    def prepare_inputs_for_generation(self, input_ids, cache_params=None, use_cache=None,
                                      attention_mask=None, news_embeds=None, **kwargs):
        if cache_params is not None:                       # feed only what the state has not seen
            input_ids = input_ids[:, cache_params.seq_len:]
            attention_mask = None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_params": cache_params,
            "use_cache": use_cache if use_cache is not None else self.config.use_cache,
        }
        if news_embeds is not None:
            model_inputs["news_embeds"] = news_embeds
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        news_embeds=None,
        cache_params: Optional[ReplyflyCache] = None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        out = self.brain(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            news_embeds=news_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        h = out.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.head(h[:, sl])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                   labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100)
        if not return_dict:
            return tuple(v for v in [loss, logits, out.cache_params, out.hidden_states] if v is not None)
        return ReplyflyCausalLMOutput(loss=loss, logits=logits, cache_params=out.cache_params,
                                      hidden_states=out.hidden_states)


def _hash_embed(text, dim):
    """The `hash` encoder from the training repo: signed character-trigram hashing, no download."""
    import hashlib
    v = torch.zeros(dim)
    t = f"  {text.lower()}  "
    for k in range(len(t) - 2):
        h = int(hashlib.md5(t[k:k + 3].encode()).hexdigest()[:8], 16)
        v[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    n = torch.linalg.norm(v)
    return v / n if n > 0 else v


class ReplyflyForConditionalGeneration(ReplyflyForCausalLM):
    """Encoder-decoder: a frozen sentence encoder smells the post, the connectome writes the reply.

    The encoder is *not* part of these weights — `config.news_encoder` names the model on the Hub
    that produced the embeddings during training. `encode_posts` loads it on first use.
    """

    def encode_posts(self, texts, device=None, batch_size=64):
        """Post text -> embedding for `news_embeds`, the same way the training data was encoded."""
        if isinstance(texts, str):
            texts = [texts]
        name = self.config.news_encoder
        if not name or name == "none":
            raise ValueError("this checkpoint has no sentence encoder recorded in its config")
        if name == "hash":
            return torch.stack([_hash_embed(t, self.config.news_dim) for t in texts]).to(
                device or self.device, dtype=self.dtype)
        if getattr(self, "_encoder", None) is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:      # the pooling head lives in the sentence-transformers config
                raise ImportError("pip install sentence-transformers to encode posts, or pass "
                                  "news_embeds yourself") from e
            self._encoder = SentenceTransformer(name, device=str(device or self.device))
        vec = self._encoder.encode([self.config.news_prefix + t for t in texts], batch_size=batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True)
        return torch.from_numpy(vec).to(device or self.device, dtype=self.dtype)

    @torch.no_grad()
    def reply(self, post, tokenizer, max_new_tokens=60, temperature=0.9, top_k=40, **kwargs):
        """Convenience: post text -> reply text."""
        news = self.encode_posts(post)
        ids = torch.tensor([[self.config.bos_token_id]], device=self.device)
        out = self.generate(ids, news_embeds=news, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                            temperature=temperature, top_k=top_k, **kwargs)
        return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
