"""nanoFLY: a fruit-fly connectome as the recurrent layer of a language model.

Importing this package pulls in torch and numpy only. `nanofly.hf` (transformers) and
`nanofly.vision` (PIL) are optional and must be imported explicitly.
"""
from nanofly.encoders import HashNewsEncoder, STNewsEncoder, load_news_encoder
from nanofly.model import (BOS, EOS, FLAG_BITS, NT_SIGN, PAD, SPECIAL_TOKENS, FlyConfig, FlyLM,
                           group, load_checkpoint, load_graph, save_checkpoint, superclass_idx,
                           token_population)

__version__ = "0.1.0"
__all__ = ["BOS", "EOS", "PAD", "SPECIAL_TOKENS", "FLAG_BITS", "NT_SIGN", "FlyConfig", "FlyLM",
           "load_graph", "group", "superclass_idx", "token_population",
           "save_checkpoint", "load_checkpoint",
           "HashNewsEncoder", "STNewsEncoder", "load_news_encoder", "__version__"]
