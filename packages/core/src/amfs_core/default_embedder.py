"""Default embedder using sentence-transformers via ONNX for zero-config semantic search.

Requires the ``embedder`` extra: ``pip install amfs-core[embedder]``

Falls back to a simple TF-IDF-like hash embedder when the ONNX model is not available,
so semantic search always works (with degraded quality).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
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


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, matches the pgvector column


def create_default_embedder() -> EmbedderABC:
    """Create the best available embedder.

    Preference order (all 384-dim, so interchangeable with the pgvector column):
    1. ``fastembed`` ``TextEmbedding`` — ONNX, offline, no torch; fast on CPU.
       Default ``BAAI/bge-small-en-v1.5`` (retrieval-tuned).
    2. ``sentence-transformers`` ``all-MiniLM-L6-v2`` via ONNX.
    3. ``HashEmbedder`` — deterministic but not semantically meaningful.

    Falls back gracefully so semantic search never hard-fails; only quality
    degrades when the optional model dependencies are missing.
    """
    try:
        return _create_fastembed_embedder()
    except Exception as exc:  # noqa: BLE001 - fastembed missing/model unavailable
        logger.info("fastembed unavailable (%s); trying sentence-transformers", exc)

    try:
        return _create_onnx_embedder()
    except ImportError:
        logger.warning(
            "Neither fastembed nor sentence-transformers/onnxruntime available — "
            "falling back to HashEmbedder (poor semantic quality). "
            "Install amfs-core[embedder] for real embeddings."
        )
        return HashEmbedder()


def _create_fastembed_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> EmbedderABC:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    class FastEmbedEmbedder(EmbedderABC):
        def __init__(self) -> None:
            self._model = TextEmbedding(model_name)
            self.model_name = model_name

        def embed(self, text: str) -> list[float]:
            return [float(x) for x in next(iter(self._model.embed([text])))]

        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            # `embed` already wraps its one string in a list because that is the
            # only signature fastembed has; handing it the whole batch instead is
            # the entire optimisation, and it batches internally from there.
            return [
                [float(x) for x in vector]
                for vector in self._model.embed(list(texts))
            ]

    logger.info("Default embedder: fastembed %s", model_name)
    return FastEmbedEmbedder()


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

        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            if not texts:
                return []
            return self._model.encode(
                list(texts), normalize_embeddings=True
            ).tolist()

        def embed_value(self, value: Any) -> list[float]:
            import json

            if isinstance(value, str):
                return self.embed(value)
            return self.embed(json.dumps(value, default=str))

    return SentenceTransformerEmbedder()
