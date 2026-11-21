"""What a caller that reads on every turn can ask of /api/v1/retrieve.

A retrieve is a handful of stages: embed the query, search by meaning, search
by words, read the discredited rules, score, rank the head with the
cross-encoder. Called once at the start of a session, nobody counts them.
Called before every reply of a conversation, the caller waits for all of them
each time, and the cross-encoder is most of the wait.

Three things are added for that caller, and each is something it has to ask
for. The tests here are mostly about the other side of that: a request that
asks for none of them is answered exactly as it was before.

  - ``rerank: false`` skips the cross-encoder. The answer is then the one a
    server with no cross-encoder installed gives, and the ``X-AMFS-Reranked``
    header says which it was.
  - ``own_account_only: true`` reaches every read of the request as a filter
    naming the calling account, with "this path and below" stated as a range
    an index can seek. It narrows; it is never applied unasked, because the
    rows another account shares with the caller are not the caller's own, and
    because the range is read by a different plan than the form it stands in
    for, which changes what a nearest-neighbour search finds.
  - A ``Server-Timing`` header says where the time went. Durations only.
"""

from __future__ import annotations

import asyncio
import re
import types
from datetime import UTC, datetime, timedelta

import amfs_http.server as server
import pytest
from amfs_core.models import MemoryEntry, Provenance, SearchQuery, SemanticQuery
from amfs_http.models import RetrieveRequest
from amfs_postgres.adapter import PostgresAdapter
from amfs_postgres.async_adapter import AsyncPostgresAdapter
from fastapi import Response


def _entry(entity_path: str, key: str, value: str, *, confidence: float = 0.9) -> MemoryEntry:
    written = datetime.now(UTC) - timedelta(days=1)
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        confidence=confidence,
        provenance=Provenance(agent_id="a", session_id="s", written_at=written),
    )


class _Adapter:
    """Preset hits, and a record of every query it was handed."""

    _has_is_artifact_col = True

    def __init__(self, semantic_hits=None, lexical_hits=None):
        self._semantic = semantic_hits or []
        self._lexical = lexical_hits or []
        self.semantic_queries: list[SemanticQuery] = []
        self.lexical_queries: list[SearchQuery] = []

    async def semantic_search(self, query, embedder, *, branch="main"):
        self.semantic_queries.append(query)
        return [] if query.max_confidence is not None else list(self._semantic)

    async def search(self, query, *, branch="main"):
        self.lexical_queries.append(query)
        return [] if query.max_confidence is not None else list(self._lexical)

    async def increment_recall_count(self, entity_path, key, *, branch="main"):
        return None


class _Reranker:
    """Prefers the document that says "clipper", and counts its calls."""

    available = True

    def __init__(self):
        self.calls = 0

    def rerank(self, query, docs):
        self.calls += 1
        return [0.99 if "clipper" in d else 0.01 for d in docs]


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(visibility_filter=None))


def _run(req: RetrieveRequest, response: Response | None = None):
    return asyncio.run(server.retrieve_entries(_request(), req, response=response, _auth=None))


def _keys(out):
    return [row["entity_path"] + "/" + row["key"] for row in out]


def _scores(out):
    return [row["_score"] for row in out]


@pytest.fixture(autouse=True)
def _server_globals(monkeypatch):
    monkeypatch.setattr(server, "_retrieval_reranker", None, raising=False)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None, raising=False)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: object())
    monkeypatch.setenv("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")
    yield


def _two_hits():
    target = _entry("amfs/browser-extension", "target", "browser extension clipper")
    other = _entry("app/ui", "other", "browser extension unrelated", confidence=0.99)
    return target, other


class TestDecliningTheCrossEncoder:
    def test_by_default_it_ranks_and_the_response_says_so(self, monkeypatch):
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))
        reranker = _Reranker()
        monkeypatch.setattr(server, "_retrieval_reranker", reranker)
        response = Response()

        out = _run(RetrieveRequest(query="browser extension", limit=10), response)

        assert reranker.calls == 1
        assert _keys(out)[0] == target.entry_key
        assert "rerank" in out[0]["_breakdown"]
        assert response.headers[server.RERANKED_HEADER] == "1"

    def test_declined_it_is_never_called(self, monkeypatch):
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))
        reranker = _Reranker()
        monkeypatch.setattr(server, "_retrieval_reranker", reranker)
        response = Response()

        out = _run(RetrieveRequest(query="browser extension", limit=10, rerank=False), response)

        assert reranker.calls == 0
        assert all("rerank" not in row["_breakdown"] for row in out)
        assert response.headers[server.RERANKED_HEADER] == "0"

    def test_declined_the_answer_is_the_one_a_server_without_one_gives(self, monkeypatch):
        """Not a third ranking: the same rows, order and scores as where no
        cross-encoder is installed, which is a path that already exists."""
        target, other = _two_hits()
        third = _entry("app/ui", "third", "something about a browser", confidence=0.6)
        hits = [(target, 0.4), (other, 0.4), (third, 0.3)]

        monkeypatch.setattr(server, "_async_adapter", _Adapter(hits, [other]))
        without = _run(RetrieveRequest(query="browser extension", limit=10))

        monkeypatch.setattr(server, "_async_adapter", _Adapter(hits, [other]))
        monkeypatch.setattr(server, "_retrieval_reranker", _Reranker())
        declined = _run(RetrieveRequest(query="browser extension", limit=10, rerank=False))

        assert _keys(declined) == _keys(without)
        assert _scores(declined) == _scores(without)
        assert [r["_breakdown"] for r in declined] == [r["_breakdown"] for r in without]

    def test_a_request_that_does_not_mention_it_is_ranked_as_before(self, monkeypatch):
        """The field's default is the behaviour every caller had."""
        target, other = _two_hits()
        hits = [(target, 0.4), (other, 0.4)]
        monkeypatch.setattr(server, "_retrieval_reranker", _Reranker())

        monkeypatch.setattr(server, "_async_adapter", _Adapter(hits))
        unmentioned = _run(RetrieveRequest.model_validate({"query": "browser extension"}))
        monkeypatch.setattr(server, "_async_adapter", _Adapter(hits))
        asked = _run(RetrieveRequest(query="browser extension", rerank=True))

        assert _keys(unmentioned) == _keys(asked)
        assert _scores(unmentioned) == _scores(asked)
        assert _keys(unmentioned)[0] == target.entry_key

    @pytest.mark.parametrize("reranker", [None, types.SimpleNamespace(available=False)])
    def test_the_header_says_no_where_there_is_none_to_run(self, monkeypatch, reranker):
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))
        monkeypatch.setattr(server, "_retrieval_reranker", reranker)
        response = Response()

        _run(RetrieveRequest(query="browser extension", limit=10), response)

        assert response.headers[server.RERANKED_HEADER] == "0"

    def test_the_header_says_no_when_it_failed(self, monkeypatch):
        """Best-effort ranking that raised did not rank."""
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))

        class _Broken:
            available = True

            def rerank(self, query, docs):
                raise RuntimeError("model unavailable")

        monkeypatch.setattr(server, "_retrieval_reranker", _Broken())
        response = Response()

        out = _run(RetrieveRequest(query="browser extension", limit=10), response)

        assert out
        assert response.headers[server.RERANKED_HEADER] == "0"

    def test_a_caller_with_no_response_to_decorate_still_gets_its_answer(self, monkeypatch):
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))

        out = _run(RetrieveRequest(query="browser extension", limit=10, rerank=False), None)

        assert len(out) == 2


class TestSearchingOnesOwnRowsOnly:
    @pytest.fixture(autouse=True)
    def _an_embedder_that_embeds(self, monkeypatch):
        """The below-gate semantic read reuses the query vector, so it only
        happens where the server could compute one."""
        embedder = types.SimpleNamespace(embed=lambda text: [0.1, 0.2, 0.3, 0.4])
        monkeypatch.setattr(server, "_get_server_embedder", lambda: embedder)

    def test_no_read_is_narrowed_unless_the_request_asks(self, monkeypatch):
        adapter = _Adapter()
        monkeypatch.setattr(server, "_async_adapter", adapter)

        _run(RetrieveRequest(query="billing", entity_path="svc", include_descendants=True,
                             min_confidence=0.2))

        queries = adapter.semantic_queries + adapter.lexical_queries
        assert len(queries) == 4  # the ranked reads and the below-gate reads
        assert not any(q.own_account_only for q in queries)

    def test_asked_it_reaches_every_read_of_the_request(self, monkeypatch):
        """The ranked reads and the below-gate reads alike: one of them left
        wide would still read every account's rows at the path."""
        adapter = _Adapter()
        monkeypatch.setattr(server, "_async_adapter", adapter)

        _run(RetrieveRequest(query="billing", entity_path="svc", include_descendants=True,
                             min_confidence=0.2, own_account_only=True))

        queries = adapter.semantic_queries + adapter.lexical_queries
        assert len(queries) == 4
        assert all(q.own_account_only for q in queries)


class _Scoped:
    """The one thing ``_own_account`` reads from its adapter."""

    def __init__(self, account_id):
        self._account_id = account_id

    def _get_current_account_id(self):
        return self._account_id


ACCOUNT = "11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize("adapter_cls", [PostgresAdapter, AsyncPostgresAdapter])
class TestTheAdaptersNameTheAccountOnlyWhenAsked:
    """``_own_account`` is the single switch: it returns the account to name,
    and only then does the adapter also state the path as a range. ``None``
    leaves the statement exactly as it was (the integration tests run the
    statements themselves against Postgres)."""

    def test_asked_with_an_account_on_the_request(self, adapter_cls):
        for query in (
            SearchQuery(entity_path="a", own_account_only=True),
            SemanticQuery(text="t", entity_path="a", own_account_only=True),
        ):
            assert adapter_cls._own_account(_Scoped(ACCOUNT), query) == ACCOUNT

    def test_not_asked_there_is_no_account_to_name(self, adapter_cls):
        for query in (SearchQuery(entity_path="a"), SemanticQuery(text="t", entity_path="a")):
            assert adapter_cls._own_account(_Scoped(ACCOUNT), query) is None

    def test_asked_with_no_account_there_is_nothing_to_name(self, adapter_cls):
        """A single-tenant deployment: every row is the one tenant's."""
        query = SearchQuery(entity_path="a", own_account_only=True)
        assert adapter_cls._own_account(_Scoped(None), query) is None
        assert adapter_cls._own_account(_Scoped(""), query) is None

    def test_the_account_is_handed_back_as_text_for_a_parameter(self, adapter_cls):
        import uuid

        query = SearchQuery(entity_path="a", own_account_only=True)
        assert adapter_cls._own_account(_Scoped(uuid.UUID(ACCOUNT)), query) == ACCOUNT


class TestServerTiming:
    def test_the_stages_of_a_retrieve_are_reported(self, monkeypatch):
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))
        monkeypatch.setattr(server, "_retrieval_reranker", _Reranker())
        response = Response()

        _run(RetrieveRequest(query="browser extension", limit=10), response)

        header = response.headers["Server-Timing"]
        names = [part.split(";")[0] for part in header.split(", ")]
        assert names == [
            "embed", "semantic", "lexical", "below_gate", "score", "rerank", "finish", "total",
        ]

    def test_it_carries_durations_and_nothing_else(self, monkeypatch):
        """No query text, no path, no count: a header is logged by proxies."""
        target, other = _two_hits()
        monkeypatch.setattr(server, "_async_adapter", _Adapter([(target, 0.4), (other, 0.4)]))
        response = Response()

        _run(RetrieveRequest(query="browser extension secret-word", entity_path="amfs",
                             include_descendants=True, limit=10), response)

        header = response.headers["Server-Timing"]
        assert all(re.fullmatch(r"[a-z_]+;dur=\d+\.\d", part) for part in header.split(", "))
        assert "secret-word" not in header and "amfs" not in header

    def test_a_stage_marked_twice_is_one_entry(self):
        """One per query variant, in a loop: the header names the stage once."""
        stages = server._Stages()
        stages.mark("semantic")
        stages.mark("lexical")
        stages.mark("semantic")

        names = [part.split(";")[0] for part in stages.header().split(", ")]
        assert names == ["semantic", "lexical", "total"]

    def test_the_total_is_the_sum_of_the_stages(self):
        stages = server._Stages()
        for name in ("a", "b", "c"):
            stages.mark(name)

        durations = {
            part.split(";dur=")[0]: float(part.split(";dur=")[1])
            for part in stages.header().split(", ")
        }
        assert durations["total"] == pytest.approx(
            durations["a"] + durations["b"] + durations["c"], abs=0.2
        )
