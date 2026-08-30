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

#: How much attention a single model call may allocate, in units of
#: ``rows x tokens^2``.
#:
#: A transformer's peak memory on a batch is dominated by the attention scores,
#: which are shaped ``[rows, heads, tokens, tokens]`` -- quadratic in the length
#: of the longest row in the batch, because every row is padded to it. For a
#: 12-head fp32 model this budget is about ``12e6 * 12 * 4`` bytes, or 570MB of
#: scores, leaving room for the weights and the runtime's arena inside a 4GB
#: container.
#:
#: Why a budget rather than a row count: a fixed batch size has to be chosen for
#: the worst case, and then short rows -- overwhelmingly the common case -- get
#: embedded in batches far smaller than they could be, losing most of the speed
#: this method exists for. Holding ``rows x tokens^2`` roughly constant instead
#: gives hundreds of rows per call on short text and tens on long text, at
#: roughly constant memory.
#:
#: This was not hypothetical. Importing `stanfordnlp/imdb` -- 25k movie reviews,
#: long enough to pad near the model's limit -- at fastembed's default of 256
#: rows per batch allocated over 3GB of scores and was killed by the kernel ten
#: seconds in, on every retry, having written nothing.
_ATTENTION_BUDGET = 12_000_000

#: Never send more rows than this in one call, however short they are. Past a
#: few hundred the attention tensors stop being what dominates and the
#: tokenizer's own output starts to, which this budget does not model. Also
#: fastembed's own default, so it is a well-trodden ceiling.
_MAX_BATCH_ROWS = 256

#: Tokens the longest row of a batch is assumed to occupy, at most. Sequences
#: are truncated to the model's limit, so no row can cost more than this however
#: long its text -- without the clamp, one enormous string would drive the
#: estimate down to a single row per call and crawl.
_MAX_TOKENS = 512

#: Bytes of text per token, deliberately low so the estimate runs high.
#: Over-estimating costs a little throughput; under-estimating costs the
#: container.
_BYTES_PER_TOKEN = 3


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


def _tokens(text: str) -> int:
    """Roughly how many tokens `text` will occupy, never above the model's cap."""
    return min(_MAX_TOKENS, max(1, len(text) // _BYTES_PER_TOKEN + 2))


def _within_budget(texts: list[str]) -> list[list[str]]:
    """`texts` split into consecutive runs each cheap enough for one model call.

    Greedy and order-preserving, which both matter. Order, because `embed_batch`
    promises one vector per text in input order and the caller matches them to
    rows positionally -- sorting by length would embed the right texts and
    attach them to the wrong content, the one failure worse than no embedding.
    Greedy, because a run only has to be small enough, not optimal: the cost is
    driven by the longest member, so packing short rows together and letting a
    long one start a new run is most of what any smarter split would achieve.
    """
    runs: list[list[str]] = []
    run: list[str] = []
    widest = 0
    for text in texts:
        length = _tokens(text)
        widest_with = max(widest, length)
        too_big = (len(run) + 1) * widest_with * widest_with > _ATTENTION_BUDGET
        if run and (too_big or len(run) >= _MAX_BATCH_ROWS):
            runs.append(run)
            run, widest = [text], length
        else:
            run.append(text)
            widest = widest_with
    if run:
        runs.append(run)
    return runs


def _create_fastembed_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> EmbedderABC:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    class FastEmbedEmbedder(EmbedderABC):
        def __init__(self) -> None:
            self._model = TextEmbedding(model_name)
            self.model_name = model_name

        def embed(self, text: str) -> list[float]:
            return [float(x) for x in next(iter(self._model.embed([text])))]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            # Batched, because one session run over many texts instead of one
            # per text is where the order-of-magnitude on bulk imports comes
            # from -- but bounded, because the batch fastembed would choose by
            # default (256 rows, whatever their length) allocates gigabytes of
            # attention on long text and gets the process killed. See
            # `_ATTENTION_BUDGET`. `embed` is a generator, so iterating it is
            # what actually runs the model.
            if not texts:
                return []
            vectors: list[list[float]] = []
            for run in _within_budget(list(texts)):
                vectors.extend(
                    [float(x) for x in vector]
                    for vector in self._model.embed(run, batch_size=len(run))
                )
            return vectors

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

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            # Bounded for the same reason as the fastembed path above:
            # `encode` batches internally at 32 by default, but it accepts the
            # whole list first and the attention cost still scales with the
            # longest row, so a batch of long text is what gets the process
            # killed. See `_ATTENTION_BUDGET`.
            if not texts:
                return []
            vectors: list[list[float]] = []
            for run in _within_budget(list(texts)):
                vectors.extend(
                    self._model.encode(
                        run, batch_size=len(run), normalize_embeddings=True
                    ).tolist()
                )
            return vectors

        def embed_value(self, value: Any) -> list[float]:
            import json

            if isinstance(value, str):
                return self.embed(value)
            return self.embed(json.dumps(value, default=str))

    return SentenceTransformerEmbedder()
