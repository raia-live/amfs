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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert many texts at once, one vector per text, in input order.

        Concrete rather than abstract so that every embedder already written
        against this interface satisfies it unchanged. The default is a loop,
        which is correct but is the slow path: nearly every real backend
        batches internally, and callers embedding a row group at a time are
        the reason this exists rather than the per-text method alone.

        Overriding it must preserve two things the loop gives for free, both
        of which callers rely on to pair vectors back to their inputs: one
        vector per text, and the order they arrived in.
        """
        return [self.embed(text) for text in texts]

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
