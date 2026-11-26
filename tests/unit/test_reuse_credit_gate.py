"""A retrieve hit is counted as a reuse only when it was about the query.

Every retrieve credited its top hit: a bump to the entry's ``recall_count``,
a row in the reuse events, and the block a client relays to its user as what
memory just did for them. But a ranked list has a first row whatever was
asked. Step 9 of the route never drops the single best result, so a question
the store holds nothing on still came back with one, and that row was booked
as a memory reused and a number of tokens saved.

The test for "about the query" is the cross-encoder's own probability, and
``_reuse_creditable`` records why nothing else in the pipeline can make it
(the measurements are in its docstring). These hold the behaviour:

* an unrelated top hit is returned and not credited, in any of the three
  places the credit lands;
* a hit the cross-encoder endorses is credited as before;
* where no cross-encoder judged the hit, nothing changes;
* the gate can be turned off without a deploy.
"""

from __future__ import annotations

import asyncio
import json
import types
from datetime import UTC, datetime

import amfs_http.server as server
import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_http.models import RetrieveRequest
from fastapi import Response


def _entry(key: str, value: str) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/billing",
        key=key,
        value=value,
        confidence=0.9,
        provenance=Provenance(agent_id="author", session_id="s", written_at=datetime.now(UTC)),
    )


class _Adapter:
    _has_is_artifact_col = True

    def __init__(self, hits):
        self._hits = list(hits)
        self.bumped: list[str] = []
        self.events: list[dict] = []

    async def semantic_search(self, query, embedder, *, branch="main"):
        return list(self._hits)

    async def search(self, query, *, branch="main"):
        return []

    async def increment_recall_count(self, entity_path, key, *, branch="main"):
        self.bumped.append(key)

    async def record_reuse_event(self, **row):
        self.events.append(row)


class _Reranker:
    """Logits by the word a document carries, as the serving model emits them."""

    available = True

    def __init__(self, logits: dict[str, float]):
        self._logits = logits
        self.calls = 0

    def rerank(self, query, docs):
        self.calls += 1
        return [next(v for word, v in self._logits.items() if word in doc) for doc in docs]


def _request():
    return types.SimpleNamespace(
        state=types.SimpleNamespace(visibility_filter=None),
        headers={"x-amfs-agent-id": "reader"},
    )


@pytest.fixture(autouse=True)
def _server(monkeypatch):
    monkeypatch.setattr(server, "_retrieval_reranker", None, raising=False)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None, raising=False)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: object())
    monkeypatch.setenv("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")
    monkeypatch.delenv("AMFS_REUSE_CREDIT_GATE", raising=False)
    persisted: list[dict] = []
    monkeypatch.setattr(
        server, "_persist_reuse_event", lambda credited, **row: persisted.append(row)
    )
    return persisted


def _retrieve(monkeypatch, adapter, *, reranker=None, **req):
    monkeypatch.setattr(server, "_async_adapter", adapter)
    if reranker is not None:
        monkeypatch.setattr(server, "_retrieval_reranker", reranker)
    response = Response()
    rows = asyncio.run(
        server.retrieve_entries(
            _request(), RetrieveRequest(query="what changed recently", limit=5, **req),
            response, _auth=None,
        )
    )
    return [r for r in rows if not r.get("_meta")], response


# Logits as measured on the serving cross-encoder: an unrelated passage sits
# around -9, a passage that answers the question around +6.
UNRELATED, ANSWERS = -8.9, 6.1


def test_an_unrelated_top_hit_is_returned_and_not_counted(monkeypatch, _server):
    """The row still comes back: abstaining is the caller's to ask for. What
    stops is the claim that memory was reused."""
    chatter = _entry("conversation-1", "a transcript about gallery lighting")
    adapter = _Adapter([(chatter, 0.66)])

    rows, response = _retrieve(monkeypatch, adapter, reranker=_Reranker({"gallery": UNRELATED}))

    assert [r["key"] for r in rows] == ["conversation-1"], "step 9 keeps the single best result"
    assert rows[0]["_breakdown"]["rerank_absolute"] < 0.01
    assert adapter.bumped == [], "recall_count is the stored form of the same claim"
    assert server.REUSE_VALUE_HEADER not in response.headers
    assert _server == [], "and so is the reuse event"


def test_a_hit_the_cross_encoder_endorses_is_counted_as_before(monkeypatch, _server):
    decision = _entry("decision-1", "decided: the launch moved to December")
    adapter = _Adapter([(decision, 0.7)])

    rows, response = _retrieve(monkeypatch, adapter, reranker=_Reranker({"decided": ANSWERS}))

    assert adapter.bumped == ["decision-1"]
    block = json.loads(response.headers[server.REUSE_VALUE_HEADER])
    assert block["memories_used"] == 1
    assert len(_server) == 1


def test_the_bar_is_the_one_the_abstain_floor_already_uses(monkeypatch):
    """Better-than-even odds by the model's own reckoning: one threshold for
    "the cross-encoder vouches for this", not two that drift."""
    assert server._reuse_creditable({"rerank_absolute": server.RERANK_ENDORSED})
    assert not server._reuse_creditable({"rerank_absolute": server.RERANK_ENDORSED - 0.001})


def test_it_is_the_absolute_score_and_not_standing_in_the_batch(monkeypatch, _server):
    """The best of a bad batch normalises high. Crediting on that would count
    exactly the case this exists for: nothing in the store is about the query."""
    best_of_bad = _entry("conversation-1", "art chatter")
    worse = _entry("conversation-2", "food chatter")
    adapter = _Adapter([(best_of_bad, 0.66), (worse, 0.64)])

    rows, response = _retrieve(
        monkeypatch, adapter, reranker=_Reranker({"art": -3.0, "food": -9.0})
    )

    top = rows[0]["_breakdown"]
    assert rows[0]["key"] == "conversation-1"
    assert top["rerank_normalised"] > 0.9 and top["rerank_absolute"] < 0.05
    assert adapter.bumped == [] and server.REUSE_VALUE_HEADER not in response.headers


def test_only_the_top_hit_was_ever_a_candidate_for_the_credit(monkeypatch, _server):
    """An endorsed hit in second place is not promoted into the credit when
    the first is refused: the credit is for what led the list, and the agent
    was handed the list in this order."""
    leads_on_trust = _entry("note-1", "trusted and beside the point").model_copy(
        update={"confidence": 1.0}
    )
    answers = _entry("note-2", "untested and answers it").model_copy(update={"confidence": 0.1})
    adapter = _Adapter([(leads_on_trust, 0.5), (answers, 0.5)])

    class _Calibrated(_Reranker):
        """Probabilities, so the absolute score is the number returned."""

        def rerank(self, query, docs):
            return [0.49 if "beside the point" in d else 0.51 for d in docs]

    rows, response = _retrieve(monkeypatch, adapter, reranker=_Calibrated({}))

    assert [r["key"] for r in rows] == ["note-1", "note-2"]
    assert rows[1]["_breakdown"]["rerank_absolute"] >= server.RERANK_ENDORSED
    assert adapter.bumped == []
    assert server.REUSE_VALUE_HEADER not in response.headers


# ── where nothing judged the hit, nothing changes ─────────────────────


def test_without_a_cross_encoder_the_top_hit_is_counted_as_before(monkeypatch, _server):
    """A deployment with no reranker has no opinion to gate on. Its numbers
    are what they were."""
    chatter = _entry("conversation-1", "a transcript about gallery lighting")
    adapter = _Adapter([(chatter, 0.66)])

    _, response = _retrieve(monkeypatch, adapter)

    assert adapter.bumped == ["conversation-1"]
    assert server.REUSE_VALUE_HEADER in response.headers


def test_a_caller_that_declined_the_cross_encoder_is_counted_as_before(monkeypatch, _server):
    chatter = _entry("conversation-1", "a transcript about gallery lighting")
    adapter = _Adapter([(chatter, 0.66)])
    reranker = _Reranker({"gallery": UNRELATED})

    _, response = _retrieve(monkeypatch, adapter, reranker=reranker, rerank=False)

    assert reranker.calls == 0
    assert adapter.bumped == ["conversation-1"]
    assert server.REUSE_VALUE_HEADER in response.headers


def test_a_score_that_cannot_be_read_does_not_withhold_a_credit():
    """Reporting never fails a read, and a malformed breakdown is not evidence
    that the hit was unrelated."""
    assert server._reuse_creditable({})
    assert server._reuse_creditable({"rerank_absolute": None})
    assert server._reuse_creditable({"rerank_absolute": "n/a"})


# ── the switch ────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["off", "0", "false", "no", " OFF "])
def test_the_gate_can_be_turned_off_without_a_deploy(monkeypatch, _server, value):
    monkeypatch.setenv("AMFS_REUSE_CREDIT_GATE", value)
    chatter = _entry("conversation-1", "a transcript about gallery lighting")
    adapter = _Adapter([(chatter, 0.66)])

    _, response = _retrieve(monkeypatch, adapter, reranker=_Reranker({"gallery": UNRELATED}))

    assert adapter.bumped == ["conversation-1"]
    assert server.REUSE_VALUE_HEADER in response.headers


@pytest.mark.parametrize("value", ["", "on", "endorsed", "1"])
def test_it_is_on_unless_turned_off(monkeypatch, value):
    monkeypatch.setenv("AMFS_REUSE_CREDIT_GATE", value)
    assert server._reuse_credit_gated()
    assert not server._reuse_creditable({"rerank_absolute": 0.0001})
