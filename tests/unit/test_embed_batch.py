"""Batch embedding, which is what makes a bulk import affordable.

Embedding one string per call pays the inference runtime's per-call overhead on
every row, so a 25,000-row dataset import costs 25,000 separate inferences. That
is the dominant CPU cost of an import, and the models underneath all accept a
list, so the cost was pure overhead.

The contract these tests pin is narrow but load-bearing: one vector out per text
in, in the order given. Callers zip the result straight back onto their rows, so
a batch that drops or reorders silently attaches the wrong vector to the wrong
row — a corruption no exception announces and no later read can detect.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from amfs_core.default_embedder import HashEmbedder
from amfs_core.embedder import EmbedderABC


class _Counting(EmbedderABC):
    """Only implements `embed`, like every embedder written before batching."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 1.0]


class _Batching(EmbedderABC):
    """Overrides the batch path, like the fastembed and ST embedders do."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError("the batch path must not fall back to embed")

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


class TestTheDefaultCostsNothingToAdopt:
    """An embedder written before this method existed keeps working."""

    def test_it_falls_back_to_one_call_per_text(self):
        embedder = _Counting()

        vectors = embedder.embed_batch(["a", "bb", "ccc"])

        assert embedder.calls == 3
        assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]

    def test_an_empty_batch_asks_the_model_nothing(self):
        embedder = _Counting()

        assert embedder.embed_batch([]) == []
        assert embedder.calls == 0

    def test_the_hash_fallback_batches_too(self):
        # It gains no speed from batching, but callers should not have to know
        # which embedder they were handed.
        vectors = HashEmbedder(dim=8).embed_batch(["alpha", "beta"])

        assert len(vectors) == 2
        assert all(len(v) == 8 for v in vectors)


class TestAnOverrideReplacesTheLoopEntirely:
    def test_one_call_carries_the_whole_batch(self):
        embedder = _Batching()

        embedder.embed_batch(["a", "bb", "ccc"])

        assert embedder.batches == [["a", "bb", "ccc"]], "one call, not three"


class TestTheOrderingContract:
    """The part a caller cannot check for itself.

    A vector list that comes back reordered still has the right length and the
    right dimensions, so nothing downstream can tell it apart from a correct
    one — it just attaches every row to the wrong meaning.
    """

    # Built per test rather than at collection time: _Counting accumulates call
    # counts, so sharing one instance across tests would leak state between them.
    @pytest.mark.parametrize("build", [_Counting, _Batching])
    def test_vectors_come_back_in_input_order(self, build):
        embedder = build()
        texts = ["a", "bbbb", "cc", "ddddddd", "e"]

        vectors = embedder.embed_batch(texts)

        assert [v[0] for v in vectors] == [1.0, 4.0, 2.0, 7.0, 1.0]

    @pytest.mark.parametrize("build", [_Counting, _Batching])
    def test_one_vector_per_text_including_duplicates(self, build):
        embedder = build()
        # Deduplicating would be a reasonable-looking optimisation and would
        # break the zip, so the count has to hold for repeated texts too.
        vectors = embedder.embed_batch(["same", "same", "same"])

        assert len(vectors) == 3

    def test_a_single_text_batch_still_returns_a_list_of_lists(self):
        vectors = _Counting().embed_batch(["only"])

        assert vectors == [[4.0, 1.0]]
