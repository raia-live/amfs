"""An entry written ``shared=False`` is its author's, on every read.

``AgentMemory.read``, ``list`` and ``search`` have always kept a private entry
from every agent but the one that wrote it. ``/api/v1/retrieve`` did not, and
neither did ``semantic_search``: the same entry that ``read`` reported as not
found and ``list`` left out came back from a search by meaning with its whole
value. So the one way to find a memory without knowing its key was also the
one way to read what an agent had marked as its own.

It was also the noisiest thing in the store. An agent that writes privately
tends to write a lot (a simulation, a scratchpad, a per-run log), and none of
it is meant for anyone else, so another agent's broad question came back as a
page of it.

Two things are held here.

**The rule itself**, wherever entries are handed back: the route, the SDK,
and the base adapter that has no SQL to state it in.

**Where it is applied.** A filter over the results is not enough. The route
reads a pool of candidates and ranks them, so private rows discarded
afterwards have already taken their places in the pool, and where one agent
keeps thousands the pool is all theirs. The rule is therefore in the query
handed to the adapter (``visible_to_agent``), and the Postgres statements put
it in their ``WHERE``. ``tests/integration/test_postgres_visible_to_agent.py``
runs those statements; this file needs no database.
"""

from __future__ import annotations

import asyncio
import types
from datetime import UTC, datetime

import amfs_http.server as server
import pytest
from amfs_core.abc import AdapterABC
from amfs_core.models import (
    MemoryEntry,
    Provenance,
    SearchQuery,
    SemanticQuery,
    readable_by,
)
from amfs_http.models import RetrieveRequest


def _entry(key: str, *, by: str, shared: bool = True) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/billing",
        key=key,
        value="deploy order for billing",
        confidence=0.9,
        shared=shared,
        provenance=Provenance(agent_id=by, session_id="s", written_at=datetime.now(UTC)),
    )


def _request(agent: str | None = None):
    headers = {"x-amfs-agent-id": agent} if agent is not None else {}
    return types.SimpleNamespace(
        state=types.SimpleNamespace(visibility_filter=None), headers=headers
    )


class _Adapter:
    """Hands back whatever it holds and records the queries it was given.

    It ignores ``visible_to_agent`` on purpose: that is an adapter that has not
    learnt the field, and the route must not depend on it having done so.
    """

    _has_is_artifact_col = True

    def __init__(self, semantic=(), lexical=()):
        self._semantic = list(semantic)
        self._lexical = list(lexical)
        self.queries: list = []

    async def semantic_search(self, query, embedder, *, branch="main"):
        self.queries.append(query)
        return list(self._semantic)

    async def search(self, query, *, branch="main"):
        self.queries.append(query)
        return list(self._lexical)


@pytest.fixture(autouse=True)
def _server(monkeypatch):
    monkeypatch.setattr(server, "_retrieval_reranker", None, raising=False)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None, raising=False)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: object())
    monkeypatch.setenv("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")


def _retrieve(adapter, monkeypatch, *, agent: str | None, **req) -> list[str]:
    monkeypatch.setattr(server, "_async_adapter", adapter)
    out = asyncio.run(
        server.retrieve_entries(
            _request(agent),
            RetrieveRequest(query="billing deploy order", limit=10, **req),
            _auth=None,
        )
    )
    return [row["key"] for row in out if not row.get("_meta")]


# ── the rule ──────────────────────────────────────────────────────────


def test_shared_or_ones_own_and_nobody_to_hide_from_when_no_agent_is_named():
    private = _entry("scratch", by="simulator", shared=False)
    shared = _entry("runbook", by="simulator")

    assert readable_by(shared, "reader")
    assert not readable_by(private, "reader")
    assert readable_by(private, "simulator")
    # Not an agent's read: an admin page, a migration.
    assert readable_by(private, None) and readable_by(private, "")


# ── the route ─────────────────────────────────────────────────────────


def test_another_agents_private_entry_is_not_retrieved(monkeypatch):
    """Both channels: a private row that matched only lexically is hidden
    exactly as one that matched by meaning."""
    by_meaning = _entry("scratch-1", by="simulator", shared=False)
    by_word = _entry("scratch-2", by="simulator", shared=False)
    runbook = _entry("runbook", by="simulator")
    adapter = _Adapter(semantic=[(by_meaning, 0.9), (runbook, 0.6)], lexical=[by_word, runbook])

    assert _retrieve(adapter, monkeypatch, agent="reader") == ["runbook"]


def test_its_author_still_retrieves_it(monkeypatch):
    private = _entry("scratch", by="simulator", shared=False)
    adapter = _Adapter(semantic=[(private, 0.9)])

    assert _retrieve(adapter, monkeypatch, agent="simulator") == ["scratch"]


def test_a_request_that_names_no_agent_is_answered_as_before(monkeypatch):
    """The account's own tooling reads with a key and no agent: the page that
    lists every topic, a service calling on the account's behalf. Hiding rows
    from "nobody" would empty views that have always shown them."""
    private = _entry("scratch", by="simulator", shared=False)
    adapter = _Adapter(semantic=[(private, 0.9)])

    assert _retrieve(adapter, monkeypatch, agent=None) == ["scratch"]
    assert all(q.visible_to_agent is None for q in adapter.queries)


def test_an_empty_header_names_nobody(monkeypatch):
    private = _entry("scratch", by="simulator", shared=False)
    adapter = _Adapter(semantic=[(private, 0.9)])

    assert _retrieve(adapter, monkeypatch, agent="   ") == ["scratch"]


def test_the_bodys_agent_id_is_not_the_reader(monkeypatch):
    """It is there to spread a fleet over untried actions, and services that
    read on an account's behalf send it. Only the session's own identity, the
    header, says an agent is asking."""
    private = _entry("scratch", by="simulator", shared=False)
    adapter = _Adapter(semantic=[(private, 0.9)])

    assert _retrieve(adapter, monkeypatch, agent=None, agent_id="reader") == ["scratch"]
    assert all(q.visible_to_agent is None for q in adapter.queries)
    assert _retrieve(adapter, monkeypatch, agent="reader", agent_id="simulator") == []


def test_every_read_the_route_makes_states_the_rule(monkeypatch):
    """Stated in the query, so an adapter with a statement to put it in applies
    it before the limit. All of them: the two channels, and the below-gate
    pair that runs when the caller sets a confidence gate on a scope."""
    adapter = _Adapter()
    # An embedder the route can drive: the below-gate semantic read reuses the
    # vector the route computed, and is skipped without one.
    monkeypatch.setattr(server, "_get_server_embedder", lambda: _Embedder())

    _retrieve(adapter, monkeypatch, agent="reader", entity_path="svc/billing", min_confidence=0.6)

    kinds = {type(q) for q in adapter.queries}
    assert kinds == {SemanticQuery, SearchQuery}
    assert len(adapter.queries) == 4, "semantic, lexical, and the below-gate pair"
    assert all(q.visible_to_agent == "reader" for q in adapter.queries)


def test_the_count_of_what_could_have_been_consulted_follows_the_same_rule(monkeypatch):
    """The gap report says how many memories the agent would have been shown
    had it asked. It is checked by asking, so it has to hide what asking hides."""
    private = _entry("scratch", by="simulator", shared=False)
    runbook = _entry("runbook", by="simulator")
    adapter = _Adapter(semantic=[(private, 0.9), (runbook, 0.8)])
    monkeypatch.setattr(server, "_async_adapter", adapter)

    keys = asyncio.run(
        server._memories_matching_task("billing deploy order", request=_request("reader"))
    )

    assert all(q.visible_to_agent == "reader" for q in adapter.queries)
    assert adapter.queries, "it searched"
    # The fake ignores the field, so both come back here; the statement is
    # what removes the private one, and the integration suite runs it.
    assert runbook.entry_key in keys


# ── an adapter with no SQL to state it in ─────────────────────────────


class _Listing(AdapterABC):
    """The base adapter's own ``search`` and ``semantic_search`` over a list."""

    def __init__(self, entries):
        self._entries = list(entries)

    def ensure_agent(self, agent_id, namespace="default"):
        return None

    def list(self, entity_path=None, **_kw):
        return [e for e in self._entries if entity_path is None or e.entity_path == entity_path]


# Everything abstract the two methods under test never touch.
_Listing.__abstractmethods__ = frozenset()


class _Embedder:
    def embed(self, text):
        return [1.0, 0.0]


def test_the_base_search_hides_them_before_its_limit():
    """Three private rows ahead of the shared one on confidence, and a page of
    two: filtered afterwards the page would hold nothing the reader may see."""
    private = [
        _entry(f"scratch-{i}", by="simulator", shared=False).model_copy(update={"confidence": 0.99})
        for i in range(3)
    ]
    runbook = _entry("runbook", by="simulator").model_copy(update={"confidence": 0.5})
    adapter = _Listing([*private, runbook])

    seen = adapter.search(
        SearchQuery(entity_path="svc/billing", limit=2, visible_to_agent="reader")
    )
    assert [e.key for e in seen] == ["runbook"]

    everything = adapter.search(SearchQuery(entity_path="svc/billing", limit=2))
    assert [e.key for e in everything] == ["scratch-0", "scratch-1"]


def test_the_base_semantic_search_hides_them_before_its_limit():
    near = [
        _entry(f"scratch-{i}", by="simulator", shared=False).model_copy(
            update={"embedding": [1.0, 0.0]}
        )
        for i in range(3)
    ]
    runbook = _entry("runbook", by="simulator").model_copy(update={"embedding": [0.6, 0.8]})
    adapter = _Listing([*near, runbook])

    query = SemanticQuery(text="q", entity_path="svc/billing", limit=2, visible_to_agent="reader")
    assert [e.key for e, _ in adapter.semantic_search(query, _Embedder())] == ["runbook"]

    own = SemanticQuery(
        text="q", entity_path="svc/billing", limit=2, visible_to_agent="simulator"
    )
    mine = adapter.semantic_search(own, _Embedder())
    assert [e.key for e, _ in mine] == ["scratch-0", "scratch-1"]


# ── the SDK ───────────────────────────────────────────────────────────


def _memory(adapter, agent_id: str):
    from amfs.memory import AgentMemory

    return AgentMemory(agent_id, adapter=adapter, embedder=_Embedder())


def test_semantic_search_asks_for_what_this_agent_may_read():
    private = _entry("scratch", by="simulator", shared=False)
    runbook = _entry("runbook", by="simulator")
    asked: list[SemanticQuery] = []

    class _Old(_Listing):
        """An adapter that predates the field and returns everything."""

        def semantic_search(self, query, embedder):
            asked.append(query)
            return [(private, 0.9), (runbook, 0.8)]

    found = _memory(_Old([]), "reader").semantic_search("billing deploy order")
    assert [e.key for e, _ in found] == ["runbook"]
    assert asked[0].visible_to_agent == "reader"

    own = _memory(_Old([]), "simulator").semantic_search("q")
    assert [e.key for e, _ in own] == ["scratch", "runbook"]


def test_retrieve_drops_what_a_server_without_the_rule_hands_back(monkeypatch):
    """A newer client against an older server: the private entry is still not
    this agent's to be handed."""
    private = _entry("scratch", by="simulator", shared=False)
    runbook = _entry("runbook", by="simulator")

    class _OldServer(_Listing):
        def retrieve(self, **_kw):
            return [(private, 0.9, {}), (runbook, 0.8, {})]

    mem = _memory(_OldServer([]), "reader")
    monkeypatch.setattr(type(mem), "environment", lambda self: {})
    monkeypatch.setattr(type(mem), "_record_retrieval_reuse", lambda self, *a, **k: None)

    assert [s.entry.key for s in mem.retrieve("billing deploy order")] == ["runbook"]
