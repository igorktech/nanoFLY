"""Config for the Replyfly connectome language models (published with trust_remote_code)."""
from transformers import PretrainedConfig

CONNECTOME = ("MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, "
              "Google Research. CC BY 4.0.")


class ReplyflyConfig(PretrainedConfig):
    """A fruit-fly connectome as the recurrent layer of a language model.

    Two architectures share this config:

    * ``decoder`` — token in, token out, nothing else. Comparable to ngxson/fly-llm-hf.
    * ``encoder_decoder`` — the post being answered is encoded once by a frozen sentence encoder
      (``news_encoder``) and injected as a constant current into the olfactory neurons for the whole
      generation, so the fly "smells" the post while it writes. Only the projection into those
      neurons is trained here; the sentence encoder is a separate model and is not shipped.

    The recurrent weights are anatomy, not learned: ``w_values`` are signed, row-normalised synapse
    counts between the neurons of the release above. ``mode`` records whether the published values
    are the raw connectome (``gains``) or connectome-masked learned strengths (``edges``).
    """

    model_type = "replyfly"
    keys_to_ignore_at_inference = ["cache_params"]

    def __init__(
        self,
        arch="decoder",
        vocab_size=2048,
        d_emb=256,
        delay=8,
        ticks=2,
        mode="gains",
        token_input="cb_sensory",
        n_neurons=1,
        n_edges=0,
        n_token_input=1,
        n_news_input=0,
        readout="all",
        readout_size=1,
        readout_rank=256,
        head_type="lowrank",
        news_dim=0,
        news_group="orn",
        n_reserved=0,
        news_mode="direct",
        news_glom=0,
        news_encoder="none",
        news_prefix="",
        min_syn=1,
        modulatory_sign=0.0,
        connectome=CONNECTOME,
        source_repo="",
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.arch = arch
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.delay = delay
        self.ticks = ticks
        self.mode = mode
        self.token_input = token_input      # which population the delay line writes into
        self.n_neurons = n_neurons
        self.n_edges = n_edges
        self.n_token_input = n_token_input
        self.n_news_input = n_news_input
        self.readout = readout
        self.readout_size = readout_size
        self.readout_rank = readout_rank
        self.head_type = head_type          # lowrank: Linear->LayerNorm->Linear, linear: LayerNorm->Linear
        self.news_dim = news_dim
        self.news_group = news_group        # population carved out for the post channel
        self.n_reserved = n_reserved        # its size, even when this variant does not use it
        self.news_mode = news_mode
        self.news_glom = news_glom
        self.news_encoder = news_encoder
        self.news_prefix = news_prefix
        self.min_syn = min_syn
        self.modulatory_sign = modulatory_sign
        self.connectome = connectome
        self.source_repo = source_repo
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def hidden_size(self):
        return self.readout_size

    @property
    def conditional(self):
        return self.arch == "encoder_decoder" and self.news_dim > 0
