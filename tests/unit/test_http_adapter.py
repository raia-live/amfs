"""Unit tests for HttpAdapter — verifies correct HTTP calls are made.

Uses httpx mock transport so no real server is needed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from amfs_core.models import (
    GraphNeighborQuery,
    SearchQuery,
)


def _make_adapter(responses: dict[str, Any] | None = None):
    """Create an HttpAdapter with a mock transport that returns canned responses.

    A canned value is normally the JSON body to return with a 200. Pass a
    ``(status, body)`` tuple instead when the status is the thing under test —
    a 404 for a commit that does not exist, or for an endpoint an older server
    has never heard of.
    """
    from amfs_adapter_http.adapter import HttpAdapter

    canned = responses or {}
    call_log: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        key = f"{method} {path}"

        body = None
        if request.content:
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        call_log.append({
            "method": method,
            "path": path,
            "params": dict(request.url.params),
            "body": body,
        })

        resp_data = canned.get(key, canned.get(path, {}))
        if isinstance(resp_data, tuple):
            status, resp_data = resp_data
            return httpx.Response(status, json=resp_data)
        return httpx.Response(200, json=resp_data)

    adapter = HttpAdapter.__new__(HttpAdapter)
    adapter._base = "http://test"
    adapter._api_key = "test-key"
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "test-key"},
        transport=httpx.MockTransport(_handler),
    )
    return adapter, call_log


class TestSearchForwardsDepth:
    def test_depth_included_in_body(self) -> None:
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        sq = SearchQuery(entity_path="svc", depth=2, limit=10)
        adapter.search(sq)

        assert len(calls) == 1
        assert calls[0]["body"]["depth"] == 2

    def test_depth_defaults_to_3(self) -> None:
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        sq = SearchQuery(entity_path="svc")
        adapter.search(sq)

        assert calls[0]["body"]["depth"] == 3

    def test_branch_forwarded_via_kwargs(self) -> None:
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        sq = SearchQuery(entity_path="svc")
        adapter.search(sq, branch="feature-x")

        assert calls[0]["body"]["branch"] == "feature-x"

    def test_include_artifacts_false_forwarded(self) -> None:
        # Dropping this flag made every SaaS caller's include_artifacts=False a
        # no-op, so a search could come back carrying whole source files.
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        adapter.search(SearchQuery(entity_path="svc", include_artifacts=False))

        assert calls[0]["body"]["include_artifacts"] is False

    def test_include_artifacts_omitted_when_default(self) -> None:
        # Left out when True so older servers keep their own default.
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        adapter.search(SearchQuery(entity_path="svc"))

        assert "include_artifacts" not in calls[0]["body"]


class TestGraphNeighbors:
    def test_proxies_to_server(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/pro/graph/neighbors": {
                "entity": "checkout",
                "edges": [
                    {
                        "source_entity": "checkout",
                        "source_type": "service",
                        "relation": "calls",
                        "target_entity": "payment",
                        "target_type": "service",
                        "confidence": 0.9,
                    }
                ],
                "count": 1,
            },
        })
        query = GraphNeighborQuery(
            entity="checkout",
            relation="calls",
            direction="outgoing",
            depth=2,
            limit=20,
        )
        edges = adapter.graph_neighbors(query)

        assert len(edges) == 1
        assert edges[0].target_entity == "payment"
        assert calls[0]["params"]["entity"] == "checkout"
        assert calls[0]["params"]["relation"] == "calls"
        assert calls[0]["params"]["direction"] == "outgoing"
        assert calls[0]["params"]["depth"] == "2"
        assert calls[0]["params"]["limit"] == "20"


class TestListForwardsSuperseded:
    def test_include_superseded_sent(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/entries": {"entries": []},
        })
        adapter.list("svc", include_superseded=True)

        assert calls[0]["params"]["include_superseded"] == "true"

    def test_superseded_not_sent_by_default(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/entries": {"entries": []},
        })
        adapter.list("svc")

        assert "include_superseded" not in calls[0]["params"]


class TestWriteCarriesTheCallersSession:
    """The server builds provenance from what this body says.

    Anything omitted here is filled in from the server's own tagger, which is
    one process serving every agent — so a dropped session_id silently
    relabels the write with a session that has nothing to do with the caller,
    and the decision trace (keyed on the caller's session) can no longer be
    joined to the entries it produced.
    """

    def _write(self):
        from datetime import datetime, timezone

        from amfs_core.models import MemoryEntry, Provenance

        adapter, calls = _make_adapter({
            "POST /api/v1/entries": {
                "entity_path": "svc",
                "key": "k",
                "value": "v",
                "provenance": {
                    "agent_id": "agent-1",
                    "session_id": "sess-caller",
                    "written_at": "2026-01-01T00:00:00Z",
                },
            },
        })
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="v",
            provenance=Provenance(
                agent_id="agent-1",
                session_id="sess-caller",
                written_at=datetime.now(timezone.utc),
            ),
        )
        adapter.write(entry)
        return calls

    def test_the_session_id_is_sent(self) -> None:
        assert self._write()[0]["body"]["session_id"] == "sess-caller"

    def test_the_agent_id_is_still_sent(self) -> None:
        assert self._write()[0]["body"]["agent_id"] == "agent-1"


class TestBriefingProxy:
    def test_proxies_to_server_endpoint(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/briefing": {
                "digests": [
                    {
                        "digest_type": "entity",
                        "scope": "checkout-service",
                        "summary": {"key_facts": ["uses Stripe"]},
                        "entry_count": 5,
                        "compiled_at": "2026-04-12T00:00:00Z",
                    }
                ],
                "total": 1,
            },
        })
        digests = adapter.briefing(entity_path="checkout-service", agent_id="test-agent")

        assert len(digests) == 1
        assert digests[0].scope == "checkout-service"
        assert calls[0]["params"]["entity_path"] == "checkout-service"
        assert calls[0]["params"]["agent_id"] == "test-agent"


class TestEnsureAgent:
    def test_returns_stub_without_http_call(self) -> None:
        adapter, calls = _make_adapter()
        agent = adapter.ensure_agent("my-agent", "default")

        assert agent.agent_id == "my-agent"
        assert len(calls) == 0


def _make_error_adapter(status_code: int, body: Any):
    """Create an HttpAdapter whose transport always returns an error response."""
    from amfs_adapter_http.adapter import HttpAdapter

    def _handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, (dict, list)):
            return httpx.Response(status_code, json=body)
        return httpx.Response(status_code, text=body)

    adapter = HttpAdapter.__new__(HttpAdapter)
    adapter._base = "http://test"
    adapter._api_key = "test-key"
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "test-key"},
        transport=httpx.MockTransport(_handler),
    )
    return adapter


class TestErrorDetailSurfaced:
    """4xx/5xx responses must expose the server's `detail` in the exception
    message so MCP agents can read the guidance and self-correct (e.g. the
    409 agent-identity collision tells them to call amfs_set_identity)."""

    def test_409_detail_in_message(self) -> None:
        detail = (
            "Agent identity 'agent/dai' is already registered to a different "
            "user in this account. Set a unique agent identity and retry."
        )
        adapter = _make_error_adapter(409, {"detail": detail})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert detail in str(exc_info.value)
        assert exc_info.value.response.status_code == 409

    def test_update_agent_profile_surfaces_detail(self) -> None:
        detail = "Agent identity 'agent/dai' is already registered to a different user"
        adapter = _make_error_adapter(409, {"detail": detail})

        profile = MagicMock()
        profile.model_dump.return_value = {"display_name": "Dai"}

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter.update_agent_profile("agent/dai", profile)

        assert detail in str(exc_info.value)

    def test_error_key_used_as_fallback(self) -> None:
        adapter = _make_error_adapter(403, {"error": "restricted member"})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert "restricted member" in str(exc_info.value)

    def test_non_json_body_falls_back_to_plain_raise(self) -> None:
        adapter = _make_error_adapter(500, "internal server error")

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert exc_info.value.response.status_code == 500

    def test_404_still_catchable_by_status_code(self) -> None:
        # entity_summaries() and friends catch HTTPStatusError and check
        # response.status_code == 404 — the enriched error must preserve that.
        adapter = _make_error_adapter(404, {"detail": "not found"})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/missing")

        assert exc_info.value.response.status_code == 404


def _commit(commit_id: str, *parents: str, branch: str = "main") -> dict[str, Any]:
    """The JSON the server returns for one commit."""
    return {
        "id": commit_id,
        "message": f"commit {commit_id}",
        "author_agent_id": "some-agent",
        "parent_ids": list(parents),
        "branch": branch,
        "namespace": "default",
    }


class TestCommitsOverHttp:
    """The commit methods, which until now were inherited placeholders.

    ``get_commit``, ``list_commits`` and ``common_ancestor`` all fell through to
    ``AdapterABC``, whose implementations return ``None``, ``[]`` and — via the
    walk that calls ``get_commit`` — ``None`` again. Every one of those is a
    plausible answer about an account with no history, so over HTTP the commit
    log was empty for everybody and no two commits ever had a common ancestor,
    and nothing anywhere reported a problem.
    """

    def test_the_placeholders_are_no_longer_what_runs(self) -> None:
        """Guards the actual regression: inheriting these silently is the bug.

        A future refactor that removes an override brings back a wrong answer
        that looks like a right one, so pin where each method comes from.
        """
        from amfs_adapter_http.adapter import HttpAdapter

        def defined_in(name: str) -> str:
            return next(k.__name__ for k in HttpAdapter.__mro__ if name in k.__dict__)

        assert defined_in("get_commit") == "HttpAdapter"
        assert defined_in("list_commits") == "HttpAdapter"
        assert defined_in("common_ancestor") == "HttpAdapter"
        # Still deliberately inherited: commits are minted by the server inside
        # its own transaction, so there is nothing for a client to save.
        assert defined_in("save_commit") == "AdapterABC"

    def test_a_commit_is_fetched_and_parsed(self) -> None:
        adapter, calls = _make_adapter({
            "GET /api/v1/commits/c-2": _commit("c-2", "c-1"),
        })

        commit = adapter.get_commit("c-2")

        assert commit is not None
        assert commit.id == "c-2"
        assert commit.parent_ids == ["c-1"]
        assert calls[0]["path"] == "/api/v1/commits/c-2"

    def test_a_missing_commit_is_none_rather_than_an_exception(self) -> None:
        """404 is the ordinary answer for an id that does not exist.

        The signature promises ``Commit | None`` and the DAG walk depends on it:
        raising here would abort a traversal that should simply stop following
        that branch.
        """
        adapter, _calls = _make_adapter({
            "GET /api/v1/commits/nope": (404, {"detail": "Commit not found"}),
        })

        assert adapter.get_commit("nope") is None

    def test_a_real_error_still_raises(self) -> None:
        """Only 404 means "no such commit". A 500 must not read as absence."""
        adapter, _calls = _make_adapter({
            "GET /api/v1/commits/c-1": (500, {"detail": "database is down"}),
        })

        with pytest.raises(httpx.HTTPStatusError):
            adapter.get_commit("c-1")

    def test_the_commit_log_comes_back_populated(self) -> None:
        adapter, calls = _make_adapter({
            "GET /api/v1/commits": {
                "commits": [_commit("c-2", "c-1"), _commit("c-1")],
                "count": 2,
            },
        })

        commits = adapter.list_commits(limit=10)

        assert [c.id for c in commits] == ["c-2", "c-1"]
        assert calls[0]["params"]["limit"] == "10"

    def test_the_ancestor_is_one_request_not_a_walk(self) -> None:
        """The reason for the override.

        The inherited walk asks for every commit it visits, so over HTTP the
        cost of the answer grows with the history the two commits share — and
        each of those round trips is a billable operation. The server holds the
        whole graph and answers this exact question, so it is asked once.
        """
        adapter, calls = _make_adapter({
            "POST /api/v1/merge-base": {
                "ancestor_commit_id": "c-1",
                "commit_a": "c-a",
                "commit_b": "c-b",
            },
        })

        assert adapter.common_ancestor("c-a", "c-b") == "c-1"
        assert len(calls) == 1
        assert calls[0]["path"] == "/api/v1/merge-base"
        assert calls[0]["body"] == {"commit_a": "c-a", "commit_b": "c-b"}

    def test_an_older_server_still_gets_an_answer(self) -> None:
        """A 404 on the endpoint means the server predates it, not that there
        is no ancestor. Fall back to the walk, which only needs ``get_commit``.
        """
        adapter, calls = _make_adapter({
            "POST /api/v1/merge-base": (404, {"detail": "Not Found"}),
            "GET /api/v1/commits/c-a": _commit("c-a", "c-shared"),
            "GET /api/v1/commits/c-b": _commit("c-b", "c-shared"),
            "GET /api/v1/commits/c-shared": _commit("c-shared"),
        })

        assert adapter.common_ancestor("c-a", "c-b") == "c-shared"
        # Which is the expensive shape the override exists to avoid.
        assert len(calls) > 1


class TestTheBranchFilterReachesTheServer:
    """Filtering after the limit is applied is not filtering.

    The first version of ``list_commits`` sent only ``limit`` and filtered the
    page it got back. That takes the newest N commits account-wide and discards
    the ones the caller did not ask for, so a page full of another branch's
    commits comes back empty while the requested branch has plenty.

    What makes it more than a paging nicety is ``TransactionBuffer.flush``,
    which asks for exactly one commit to use as the new commit's parent. One
    commit on the wrong branch means no parent — so every commit written over
    HTTP is a root, no two share an ancestor, and ``common_ancestor`` answers
    "none" forever. That answer is indistinguishable from an honest one about
    unrelated history, which is how it would have shipped unnoticed.
    """

    def test_branch_and_namespace_are_sent_as_query_parameters(self) -> None:
        adapter, calls = _make_adapter({"GET /api/v1/commits": {"commits": []}})

        adapter.list_commits(branch="feature-x", limit=10, namespace="ns1")

        params = calls[0]["params"]
        assert params["branch"] == "feature-x"
        assert params["namespace"] == "ns1"
        assert params["limit"] == "10"

    def test_the_parent_lookup_asks_the_server_for_its_branch(self) -> None:
        """flush's limit=1 call, which is where this actually bit."""
        adapter, calls = _make_adapter({
            "GET /api/v1/commits": {"commits": [_commit("c-tip", branch="feature-x")]},
        })

        found = adapter.list_commits(branch="feature-x", limit=1)

        assert calls[0]["params"]["branch"] == "feature-x"
        assert [c.id for c in found] == ["c-tip"], (
            "the branch tip was not returned, so a commit written now would "
            "have no parent"
        )

    def test_another_branch_is_still_dropped_if_a_server_ignores_the_filter(
        self,
    ) -> None:
        """The local filter stays, as a guard rather than as the mechanism.

        A server old enough not to know these parameters returns an unfiltered
        page. Handing the caller another branch's commits because the server
        did not understand the question is worse than handing it too few: the
        result would be wrong rather than short.
        """
        adapter, _ = _make_adapter({
            "GET /api/v1/commits": {"commits": [
                _commit("c-main", branch="main"),
                _commit("c-feature", branch="feature-x"),
            ]},
        })

        found = adapter.list_commits(branch="feature-x", limit=10)

        assert [c.id for c in found] == ["c-feature"]

    def test_no_filter_means_no_parameter_rather_than_an_empty_one(self) -> None:
        """Empty means "every branch" to the guard, so it cannot go on the wire.

        Sent as branch="", it would reach a server that reads it as a branch
        named nothing and matches no commit at all — turning "show me
        everything" into "show me nothing".
        """
        adapter, calls = _make_adapter({
            "GET /api/v1/commits": {"commits": [
                _commit("c-main", branch="main"),
                _commit("c-exp", branch="experiment"),
            ]},
        })

        commits = adapter.list_commits(branch="", limit=5, namespace="")

        assert "branch" not in calls[0]["params"]
        assert "namespace" not in calls[0]["params"]
        assert [c.id for c in commits] == ["c-main", "c-exp"]


class TestCommitBatchRunsServerSide:
    """Writing a group of entries as one commit, without assembling it here.

    The client used to build the Commit itself and hand it to ``save_commit``,
    which this adapter has never implemented — there was no endpoint that
    accepted a commit somebody else had minted. So the caller got an id back
    for an object that was dropped, and n separate write requests that could
    half-succeed were described to them as atomic.

    ``commit_batch`` posts the writes instead and lets the server transact,
    mint and persist. What these pin is that the metadata survives the trip,
    because the reason this was not done sooner is that the endpoint used to
    forward only path, key and value — moving a batch across without fixing
    that would have written the right entries with the wrong metadata, which
    is worse than not writing them.
    """

    def test_the_writes_are_posted_as_one_request(self) -> None:
        adapter, calls = _make_adapter({
            "POST /api/v1/commits": {"commit_id": "c1", "commit": _commit("c1")},
        })

        adapter.commit_batch(
            [
                {"entity_path": "app/a", "key": "k1", "value": "v1"},
                {"entity_path": "app/b", "key": "k2", "value": "v2"},
            ],
            "two entries",
        )

        assert len(calls) == 1, "a batch must not become one request per entry"
        assert calls[0]["path"] == "/api/v1/commits"
        assert calls[0]["method"] == "POST"
        assert len(calls[0]["body"]["writes"]) == 2
        assert calls[0]["body"]["message"] == "two entries"

    def test_per_write_metadata_survives_the_trip(self) -> None:
        """The regression that kept this on the client for so long."""
        adapter, calls = _make_adapter({
            "POST /api/v1/commits": {"commit": _commit("c1")},
        })

        adapter.commit_batch([{
            "entity_path": "app/a",
            "key": "k1",
            "value": "v1",
            "confidence": 0.4,
            "memory_type": "belief",
            "pattern_refs": ["risk-x"],
            "shared": True,
        }])

        sent = calls[0]["body"]["writes"][0]
        assert sent["confidence"] == 0.4
        assert sent["memory_type"] == "belief"
        assert sent["pattern_refs"] == ["risk-x"]
        assert sent["shared"] is True

    def test_the_minted_commit_comes_back(self) -> None:
        adapter, _ = _make_adapter({
            "POST /api/v1/commits": {"commit": _commit("c-abc", "c-parent")},
        })

        commit = adapter.commit_batch([{"entity_path": "a/b", "key": "k"}])

        assert commit is not None
        assert commit.id == "c-abc"
        assert commit.parent_ids == ["c-parent"]

    def test_a_server_that_returns_no_commit_is_not_a_failure(self) -> None:
        """An older server writes the entries and says nothing about a commit.

        The transaction has already landed by the time the response is read, so
        raising here would have a caller retry writes that are already there.
        ``None`` means "committed, details unavailable".
        """
        adapter, _ = _make_adapter({
            "POST /api/v1/commits": {"commit_id": "c1", "entries_written": 1},
        })

        assert adapter.commit_batch([{"entity_path": "a/b", "key": "k"}]) is None

    def test_save_commit_stays_a_no_op_rather_than_raising(self) -> None:
        """TransactionBuffer.flush still calls it on paths that have not moved.

        Making it raise would break working code to make a point about an
        arrangement this adapter has now stopped relying on.
        """
        from amfs_core.models import Commit

        adapter, calls = _make_adapter()

        adapter.save_commit(Commit(id="c1", author_agent_id="a"))

        assert calls == [], "save_commit must not reach the network"


class TestAnEmptyBatchIsNotACommit:
    """"Nothing happened" and "landed, silently" must not look the same.

    ``commit_batch`` returns None for a server too old to send a commit back,
    and reads that as "committed, details unavailable" — which is right, since
    by then the transaction has landed and raising would have the caller retry
    writes that already exist.

    An empty writes list broke that reading. It never reached flush, so the
    server answered 200 with no commit: the same response, and the caller
    concluded a commit it never made had succeeded.
    """

    def test_an_empty_batch_is_refused_before_it_is_sent(self) -> None:
        adapter, calls = _make_adapter({})

        with pytest.raises(ValueError, match="nothing to commit"):
            adapter.commit_batch([], "empty")

        assert calls == [], "a request was sent for a batch with no writes"

    def test_none_still_means_an_old_server_rather_than_an_error(self) -> None:
        """The one remaining meaning of None, which is why the above matters."""
        adapter, _ = _make_adapter({
            "POST /api/v1/commits": {"commit_id": "c1", "entries_written": 1},
        })

        assert adapter.commit_batch(
            [{"entity_path": "app/a", "key": "k", "value": "v"}], "one"
        ) is None


class TestABatchSaysWhoWroteIt:
    """The server transacts, so it has to be told whose writes these are.

    Moving the transaction server-side took the caller's name off the work:
    everything went in as the server process's own agent. The entries were
    credited to the server and so was the commit, and since the commit log
    hides commits whose author the caller cannot see, the caller could not see
    the commit they had just made — amfs_commit_log answered zero for an
    account that had commits in it.
    """

    def test_the_agent_and_session_are_sent(self) -> None:
        adapter, calls = _make_adapter({
            "POST /api/v1/commits": {"commit_id": "c1", "entries_written": 1},
        })

        adapter.commit_batch(
            [{"entity_path": "app/a", "key": "k", "value": "v"}],
            "one",
            agent_id="builder-agent",
            session_id="sess-9",
        )

        body = calls[0]["body"]
        assert body["agent_id"] == "builder-agent"
        assert body["session_id"] == "sess-9"

    def test_they_are_left_out_when_not_given(self) -> None:
        """An absent field means "use the default", not an agent called None."""
        adapter, calls = _make_adapter({
            "POST /api/v1/commits": {"commit_id": "c1", "entries_written": 1},
        })

        adapter.commit_batch([{"entity_path": "app/a", "key": "k", "value": "v"}])

        body = calls[0]["body"]
        assert "agent_id" not in body
        assert "session_id" not in body


class TestAKeyWithASlashInIt:
    """The path route cannot express one, so read must not use it.

    /api/v1/entries/{entity_path:path}/{key} is greedy: everything up to the
    final segment becomes the entity path. Reading sweep/abc + the key
    active-negotiation/xyz asks for sweep/abc/active-negotiation + xyz, which
    nothing was written at, and the answer is an ordinary "not found" rather
    than anything that looks like a bug. 314 of 4,726 entries in production
    have such a key, and none could be read back.
    """

    def test_it_goes_by_query_rather_than_by_path(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/entry": {"status": "not_found"},
        })

        adapter.read("sweep/abc", "active-negotiation/xyz")

        assert calls[0]["path"] == "/api/v1/entry"
        assert calls[0]["params"]["entity_path"] == "sweep/abc"
        assert calls[0]["params"]["key"] == "active-negotiation/xyz"

    def test_the_entry_comes_back_whole(self) -> None:
        adapter, _calls = _make_adapter({
            "/api/v1/entry": {
                "entity_path": "sweep/abc",
                "key": "active-negotiation/xyz",
                "version": 1,
                "value": "a proposal",
                "provenance": {
                    "agent_id": "a",
                    "session_id": "s",
                    "written_at": "2026-01-01T00:00:00Z",
                },
                "confidence": 0.9,
            },
        })

        entry = adapter.read("sweep/abc", "active-negotiation/xyz")

        assert entry is not None
        assert entry.key == "active-negotiation/xyz"

    def test_the_branch_still_travels(self) -> None:
        adapter, calls = _make_adapter({"/api/v1/entry": {"status": "not_found"}})

        adapter.read("sweep/abc", "active-negotiation/xyz", branch="feature")

        assert calls[0]["params"]["branch"] == "feature"


class TestAnOrdinaryKeyIsLeftAlone:
    """Most keys have no slash, and the path route answers those correctly.

    Moving them too would change every read against every deployed server to
    gain nothing.
    """

    def test_it_still_goes_by_path(self) -> None:
        adapter, calls = _make_adapter({
            "/api/v1/entries/app/svc/plain-key": {"status": "not_found"},
        })

        adapter.read("app/svc", "plain-key")

        assert calls[0]["path"] == "/api/v1/entries/app/svc/plain-key"
        assert "key" not in calls[0]["params"]
