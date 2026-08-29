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

        def encode(
            self,
            text: object,
            normalize_embeddings: bool = False,
            batch_size: int | None = None,
        ) -> _Array:
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


class TestABatchIsBoundedByWhatAttentionCosts:
    """Why a batch is split at all, and the invariant that makes it safe.

    A transformer pads every row of a batch to the longest one and allocates
    attention scores shaped `[rows, heads, tokens, tokens]`. So the cost is
    quadratic in the longest row, and the batch size that is fine for short
    text is fatal for long text. Handing fastembed its default of 256 rows of
    movie review allocated over 3GB and the kernel killed the importer ten
    seconds in, on every retry, having written nothing.

    Splitting is only safe if it is invisible: same vectors, same order, one per
    input. `assert_honours_the_contract` covers that for a small batch; these
    cover it where a split actually happens.
    """

    def test_a_long_batch_is_split_rather_than_sent_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        # Long enough to reach the token clamp, so the budget allows tens of
        # rows per call and 400 of them cannot be one batch.
        long_text = "x" * (default_embedder._MAX_TOKENS * 10)
        embedder.embed_batch([long_text] * 400)

        batches = fake.instances[-1].batches
        assert len(batches) > 1, "400 long rows must not be one model call"
        assert sum(len(b) for b in batches) == 400, "every row embedded once"

    def test_no_batch_exceeds_the_budget(self, monkeypatch: pytest.MonkeyPatch):
        """The invariant, asserted over a deliberately mixed workload.

        Mixed rather than uniform because the split is greedy over the running
        maximum: one long row arriving late must close the run it would have
        blown, not be absorbed into it.
        """
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        texts = [
            "short",
            "x" * 6000,
            *["tiny"] * 300,
            "y" * 40_000,
            "medium " * 50,
        ]
        embedder.embed_batch(texts)

        for batch in fake.instances[-1].batches:
            widest = max(default_embedder._tokens(t) for t in batch)
            assert len(batch) * widest * widest <= default_embedder._ATTENTION_BUDGET
            assert len(batch) <= default_embedder._MAX_BATCH_ROWS

    def test_short_text_still_batches_in_bulk(self, monkeypatch: pytest.MonkeyPatch):
        """The budget must not have quietly become a per-row loop.

        This is the regression that would make the fix worse than the bug: a
        conservative fixed batch size would be safe and would also throw away
        the order of magnitude `embed_batch` exists for. Short rows must still
        go hundreds at a time.
        """
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        embedder.embed_batch(["a short row"] * 1000)

        batches = fake.instances[-1].batches
        assert max(len(b) for b in batches) == default_embedder._MAX_BATCH_ROWS
        assert len(batches) == 4, "1000 short rows in 256-row calls"

    def test_splitting_preserves_order_and_content(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Positional matching is how callers attach vectors to rows.

        The fake encodes each text's length, so a reordered split shows up as
        vectors in the wrong order rather than as an error -- which is exactly
        how it would fail in production: embeddings attached to the wrong
        content, silently.
        """
        _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        texts = ["z" * (i * 200) for i in range(1, 60)]
        vectors = embedder.embed_batch(texts)

        assert len(vectors) == len(texts)
        assert [v[0] for v in vectors] == [float(len(t)) for t in texts]

    def test_one_enormous_row_is_still_embedded(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A single row over the budget has nowhere smaller to go.

        Without the token clamp the estimate for a megabyte of text would drive
        the batch to zero rows and the split would either loop forever or drop
        the row. It must come back as one batch of one.
        """
        fake = _stub_fastembed(monkeypatch)
        embedder = default_embedder._create_fastembed_embedder("stub/model")

        vectors = embedder.embed_batch(["q" * 1_000_000])

        assert len(vectors) == 1
        assert fake.instances[-1].batches == [["q" * 1_000_000]]

    def test_sentence_transformers_is_bounded_too(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The same ceiling on the other backend, which had the same hole."""
        fake = _stub_sentence_transformers(monkeypatch)
        embedder = default_embedder._create_onnx_embedder()

        long_text = "x" * (default_embedder._MAX_TOKENS * 10)
        embedder.embed_batch([long_text] * 400)

        batches = fake.instances[-1].batches
        assert len(batches) > 1
        assert sum(len(b) for b in batches) == 400
