"""Sentence encoders for the post channel. The encoder is never part of the published weights:
`config.news_encoder` records which Hub model produced the embeddings and it is downloaded on
first use. `hash` is a download-free stand-in for tests."""
import hashlib

import numpy as np


class HashNewsEncoder:
    """Deterministic encoder with nothing to download (hashed character trigrams). For tests."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f"  {t.lower()}  "
            for k in range(len(t) - 2):
                h = int(hashlib.md5(t[k:k + 3].encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            nrm = np.linalg.norm(out[i])
            if nrm > 0:
                out[i] /= nrm
        return out


class STNewsEncoder:
    def __init__(self, name, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.prefix = "query: " if "e5" in name.lower() else ""

    def encode(self, texts):
        return self.model.encode([self.prefix + t for t in texts], batch_size=64,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def load_news_encoder(name, device="cpu"):
    if not name or name == "none":
        return None
    if name == "hash":
        return HashNewsEncoder()
    return STNewsEncoder(name, device)
