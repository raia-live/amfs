"""Embedder ABC and cosine similarity for semantic search."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Any


class EmbedderABC(ABC):
    """Interface for converting memory values to embedding vectors.

    Bring your own model — OpenAI, Cohere, sentence-transformers, or
    any other embedding provider. AMFS doesn't ship a default model to
    keep the SDK lightweight.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert a text string to a dense embedding vector."""

    def embed_value(self, value: Any) -> list[float]:
        """Convert an arbitrary memory value to an embedding.

        Default: JSON-serialise the value and embed the string.
        Override for custom serialisation (e.g. structured extraction).
        """
        if isinstance(value, str):
            return self.embed(value)
        return self.embed(json.dumps(value, default=str))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
