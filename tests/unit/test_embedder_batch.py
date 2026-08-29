"""Tests for `EmbedderABC.embed_batch` — the batch contract every embedder owes.

This file exists because of a bug it would have caught. Callers that embed in
bulk were written against `embed_batch`, no embedder implemented it, and every
test of those callers passed anyway because each one used a fake that did. The
fakes defined the interface the real object never had, so the failure only
appeared in production, as rows stored with no vector and a search that found
nothing.

The lesson shapes what is tested here: not the fakes, but the real classes this
package can hand out, checked against the contract their callers assume.
"""

from __future__ import annotations

import sys
import types

import pytest

from amfs_core import default_embedder
from amfs_core.default_embedder import HashEmbedder, create_default_embedder
from amfs_core.embedder import EmbedderABC

# ── Test doubles for the optional model backends ──────────────────────


class _Array:
    """The one method of numpy's interface the ONNX embedder actually uses."""

    def __init__(self, value: object) -> None:
        self._value = value

    def tolist(self) -> object:
        return self._value


def _stub_fastembed(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake `fastembed` and return the class standing in for its model.

    Defined inside the function so each test gets a class with its own
    `instances`, rather than sharing recorded calls with every other test.

    `embed` is a generator, like the real one: an override that forgets to
    consume it would pass against a list and fail against fastembed.
    """

    class FakeTextEmbedding:
        instances: list[FakeTextEmbedding] = []

        def __init__(self, model_name: str, *args: object, **kwargs: object) -> None:
            self.model_name = model_name
            self.batches: list[list[str]] = []
            FakeTextEmbedding.instances.append(self)

        def embed(self, documents: object, *args: object, **kwargs: object):
            batch = list(documents)  # type: ignore[call-overload]
            self.batches.append(batch)
            for doc in batch:
                yield [float(len(doc)), 0.25]

    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return FakeTextEmbedding


def _stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake `sentence_transformers` and return its model class.

    `encode` returns a 1-D array for a string and a 2-D array for a list, which
    is the distinction the two methods depend on.
    """

    class FakeSentenceTransformer:
        instances: list[FakeSentenceTransformer] = []

        def __init__(self, name: str, **kwargs: object) -> None:
            self.name = name
            self.batches: list[list[str]] = []
            FakeSentenceTransformer.instances.append(self)

        def encode(self, text: object, normalize_embeddings: bool = False) -> _Array:
            if isinstance(text, str):
                self.batches.append([text])
                return _Array([float(len(text)), 0.25])
            batch = list(text)  # type: ignore[call-overload]
            self.batches.append(batch)
            return _Array([[float(len(t)), 0.25] for t in batch])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


# ── The contract ──────────────────────────────────────────────────────

TEXTS = ["the first one", "a second, rather longer one", "third"]


def assert_honours_the_contract(embedder: EmbedderABC) -> None:
    """Assert the three properties every caller of `embed_batch` relies on.

    Agreement with `embed` is asserted rather than merely "some vector of the
    right width", because it is what makes a batch interchangeable with a loop.
    An embedder whose batch path disagrees with its single path would give a
    corpus embedded one way and queries embedded another — a search that
    returns confident nonsense rather than an error.
    """
    vectors = embedder.embed_batch(TEXTS)

    assert len(vectors) == len(TEXTS), "one vector per text"
    for text, vector in zip(TEXTS, vectors):
        singly = embedder.embed(text)
        assert len(vector) == len(singly), "same dimension as embed()"
        assert vector == pytest.approx(singly), "same vector as embed(), in order"

    assert embedder.embed_batch([]) == [], "an empty batch is an empty list"


class TestTheContractHoldsForEveryEmbedderThisPackageBuilds:
    """Each concrete embedder, plus whatever this environment actually picks."""

    def test_the_hash_fallback(self):
        assert_honours_the_contract(HashEmbedder())

    def test_whatever_create_default_embedder_returns(self):
        """The object a deployment really gets, whichever backend is installed.

        Here that is usually the hash fallback, since the model extras are not
        a test dependency. The real fastembed model is covered by the assertion
        baked into Dockerfile.amfs-pro, which runs where it is installed.
        """
        assert_honours_the_contract(create_default_embedder())

    def test_the_fastembed_backend(self, monkeypatch: pytest.MonkeyPatch):
        _stub_fastembed(monkeypatch)
        assert_honours_the_contract(
            default_embedder._create_fastembed_embedder("stub/model")
        )

    def test_the_sentence_transformers_backend(self, monkeypatch: pytest.MonkeyPatch):
        _stub_sentence_transformers(monkeypatch)
        assert_honours_the_contract(default_embedder._create_onnx_embedder())


class TestTheDefaultKeepsExistingEmbeddersWorking:
    """`embed_batch` is concrete, not abstract, and this is why.

    Anyone who implemented this interface before the method existed — every
    third-party embedder in every install — must keep working untouched.
    """

    def test_an_embedder_that_only_implements_embed_still_batches(self):
        seen: list[str] = []

        class OnlyEmbed(EmbedderABC):
            def embed(self, text: str) -> list[float]:
                seen.append(text)
                return [float(len(text))]

        assert OnlyEmbed().embed_batch(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]
        assert seen == ["a", "bb", "ccc"], "in order, once each"

    def test_the_default_is_still_a_loop(self):
        """If this ever stops being true, the two tests above stop meaning much."""
        calls = 0

        class Counting(EmbedderABC):
            def embed(self, text: str) -> list[float]:
                nonlocal calls
                calls += 1
                return [1.0]

        Counting().embed_batch(["a", "b", "c", "d"])
        assert calls == 4


class TestTheOverridesActuallyBatch:
    """The reason for overriding at all: one call to the model, not one per row.

    Without this, reverting either override to the inherited loop is invisible
    — the contract tests above pass either way, since a loop is correct. It is
    only slow, and "slow" on a five-million-row import is the difference
    between an hour and a day.
    """

    def test_fastembed_is_handed_the_whole_batch_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        embedder.embed_batch(["a", "b", "c"])

        assert fake.instances[-1].batches == [["a", "b", "c"]]

    def test_sentence_transformers_is_handed_the_whole_batch_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _stub_sentence_transformers(monkeypatch)
        embedder = default_embedder._create_onnx_embedder()

        embedder.embed_batch(["a", "b", "c"])

        assert fake.instances[-1].batches == [["a", "b", "c"]]

    def test_an_empty_batch_does_not_reach_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """fastembed on an empty list is a model run with nothing to run on."""
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        assert embedder.embed_batch([]) == []

        assert fake.instances[-1].batches == []
