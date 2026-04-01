"""Default embedder using sentence-transformers via ONNX for zero-config semantic search.

Requires the ``embedder`` extra: ``pip install amfs-core[embedder]``

Falls back to a simple TF-IDF-like hash embedder when the ONNX model is not available,
so semantic search always works (with degraded quality).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from amfs_core.embedder import EmbedderABC

logger = logging.getLogger(__name__)

_HASH_DIM = 384


class HashEmbedder(EmbedderABC):
    """Deterministic hash-based embedder as a zero-dependency fallback.

    Produces fixed-dimension vectors from text via SHA-256 hashing of
    overlapping character n-grams. Not semantically meaningful but
    provides consistent, deterministic embeddings for basic similarity.
    """

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        text = text.lower().strip()
        vec = [0.0] * self._dim
        for i in range(max(1, len(text) - 2)):
            ngram = text[i : i + 3]
            h = hashlib.sha256(ngram.encode()).digest()
            for j in range(min(self._dim, len(h))):
                idx = (h[j] + j * 7) % self._dim
                vec[idx] += (h[j] / 255.0) - 0.5
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def create_default_embedder() -> EmbedderABC:
    """Create the best available embedder.

    Tries to load the ONNX-based sentence-transformers model first.
    Falls back to HashEmbedder if dependencies are missing.
    """
    try:
        return _create_onnx_embedder()
    except ImportError:
        logger.info(
            "sentence-transformers/onnxruntime not available, "
            "falling back to HashEmbedder. Install amfs-core[embedder] for better quality."
        )
        return HashEmbedder()


def _create_onnx_embedder() -> EmbedderABC:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    class SentenceTransformerEmbedder(EmbedderABC):
        def __init__(self) -> None:
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                backend="onnx",
            )

        def embed(self, text: str) -> list[float]:
            return self._model.encode(text, normalize_embeddings=True).tolist()

        def embed_value(self, value: Any) -> list[float]:
            import json

            if isinstance(value, str):
                return self.embed(value)
            return self.embed(json.dumps(value, default=str))

    return SentenceTransformerEmbedder()
