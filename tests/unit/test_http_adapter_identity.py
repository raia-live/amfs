"""Who is calling, on every request the HTTP adapter makes.

A read carries no agent or session in its query or body, so a hosted server
answering ``GET /api/v1/entries/...`` learns who asked only from headers. Two
things depend on it: attributing the read at all, and putting the whole session
on one arm of a live repair canary — which needs a key that is the same for
every request of the session and different between sessions. ``X-AMFS-Agent-Id``
and ``X-AMFS-Session`` are those headers, and ``AgentMemory`` binds them when it
is built.

The identity lives on the bound handle, not on the shared ``httpx.Client``:
``as_agent`` gives two handles one adapter, and a header set on the client
would stamp one handle's identity on the other's requests.

``X-AMFS-Situation`` travels the same way and says what kind of task the
session is on. A canary on a fix for one situation routes only the sessions
that declare it, so the header has to be there before the first read:
``AgentMemory`` sends it from the moment the ``situation`` attribute is set
(``Run.begin(situation=...)``) until the bag is cleared at commit.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from amfs_adapter_http import AGENT_ID_HEADER, SESSION_HEADER, SITUATION_HEADER, HttpAdapter


def _adapter(seen: list[httpx.Request]) -> HttpAdapter:
    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST" and request.url.path == "/api/v1/outcomes":
            return httpx.Response(200, json={"entries": []})
        if request.method == "POST" and request.url.path == "/api/v1/traces":
            # The server echoes the sealed trace; the one sent is good enough.
            return httpx.Response(200, content=request.content,
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, json={"status": "not_found"})

    adapter = HttpAdapter("http://test", "key")
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "key"},
        transport=httpx.MockTransport(_handler),
    )
    return adapter


class TestBind:
    def test_an_unbound_adapter_sends_no_identity(self) -> None:
        seen: list[httpx.Request] = []
        _adapter(seen).read("acme/billing", "k")
        assert AGENT_ID_HEADER not in seen[0].headers
        assert SESSION_HEADER not in seen[0].headers

    def test_a_bound_adapter_sends_both_on_every_request(self) -> None:
        seen: list[httpx.Request] = []
        bound = _adapter(seen).bind("sre-agent", "sess-1234")
        bound.read("acme/billing", "k")
        bound.list("acme/billing")
        assert len(seen) == 2
        for req in seen:
            assert req.headers[AGENT_ID_HEADER] == "sre-agent"
            assert req.headers[SESSION_HEADER] == "sess-1234"
        assert bound.identity_headers == {AGENT_ID_HEADER: "sre-agent", SESSION_HEADER: "sess-1234"}

    def test_the_client_is_shared_and_the_identity_is_not(self) -> None:
        """Two handles, one pool, two identities — ``as_agent``'s shape."""
        seen: list[httpx.Request] = []
        base = _adapter(seen)
        one = base.bind("agent-one", "sess-1")
        two = base.bind("agent-two", "sess-1")
        assert one._client is two._client is base._client

        one.read("p", "k")
        two.read("p", "k")
        base.read("p", "k")
        assert seen[0].headers[AGENT_ID_HEADER] == "agent-one"
        assert seen[1].headers[AGENT_ID_HEADER] == "agent-two"
        assert AGENT_ID_HEADER not in seen[2].headers
        assert base.identity_headers == {}

    def test_a_missing_id_means_no_header(self) -> None:
        seen: list[httpx.Request] = []
        _adapter(seen).bind(None, "sess-1").read("p", "k")
        assert AGENT_ID_HEADER not in seen[0].headers
        assert seen[0].headers[SESSION_HEADER] == "sess-1"

    def test_unsafe_characters_are_dropped_rather_than_refused(self) -> None:
        """An agent id is caller-chosen text; a header value is ASCII on one line."""
        seen: list[httpx.Request] = []
        _adapter(seen).bind("équipe\r\nX-Evil: 1", "s").read("p", "k")
        assert seen[0].headers[AGENT_ID_HEADER] == "quipeX-Evil: 1"
        assert "x-evil" not in {k.lower() for k in seen[0].headers}

    def test_a_per_call_header_wins_over_the_identity(self) -> None:
        seen: list[httpx.Request] = []
        bound = _adapter(seen).bind("sre-agent", "s")
        bound._request("GET", "/api/v1/entries/p/k", headers={AGENT_ID_HEADER: "override"})
        assert seen[0].headers[AGENT_ID_HEADER] == "override"
        assert seen[0].headers[SESSION_HEADER] == "s"

    def test_a_handle_built_without_init_is_simply_unbound(self) -> None:
        """Subclasses and test doubles skip ``__init__``; they must still work."""
        bare = HttpAdapter.__new__(HttpAdapter)
        assert bare.identity_headers == {}


class TestDeclareSituation:
    def test_the_declared_situation_rides_every_request_until_withdrawn(self) -> None:
        seen: list[httpx.Request] = []
        bound = _adapter(seen).bind("sre-agent", "sess-1")
        bound.read("p", "k")
        bound.declare_situation("spend summary")
        bound.read("p", "k")
        bound.list("p")
        bound.declare_situation(None)
        bound.read("p", "k")
        assert SITUATION_HEADER not in seen[0].headers
        assert seen[1].headers[SITUATION_HEADER] == "spend summary"
        assert seen[2].headers[SITUATION_HEADER] == "spend summary"
        assert SITUATION_HEADER not in seen[3].headers
        # The identity stayed put throughout.
        assert all(r.headers[SESSION_HEADER] == "sess-1" for r in seen)

    def test_a_blank_declaration_withdraws_and_a_long_one_is_cut(self) -> None:
        seen: list[httpx.Request] = []
        bound = _adapter(seen).bind("a", "s")
        bound.declare_situation("x" * 300)
        bound.read("p", "k")
        assert seen[0].headers[SITUATION_HEADER] == "x" * 200
        bound.declare_situation("   ")
        bound.read("p", "k")
        assert SITUATION_HEADER not in seen[1].headers

    def test_the_situation_is_the_handles_not_the_connections(self) -> None:
        seen: list[httpx.Request] = []
        base = _adapter(seen)
        one = base.bind("agent-one", "s")
        two = base.bind("agent-two", "s")
        one.declare_situation("spend summary")
        one.read("p", "k")
        two.read("p", "k")
        base.read("p", "k")
        assert seen[0].headers[SITUATION_HEADER] == "spend summary"
        assert SITUATION_HEADER not in seen[1].headers
        assert SITUATION_HEADER not in seen[2].headers

    def test_a_bare_handle_can_declare(self) -> None:
        bare = HttpAdapter.__new__(HttpAdapter)
        bare.declare_situation("s")
        assert bare.identity_headers == {SITUATION_HEADER: "s"}
        assert HttpAdapter._identity_headers == {}  # the class default is untouched


class TestRetrieveNamesNoBranchForMain:
    """The one read that still sent ``branch: main`` in its body. A named main
    is an explicit choice the server honours over a routed canary branch, so
    a canary session's retrieves — its primary recall path — never followed
    the route."""

    def test_main_is_omitted_and_a_branch_is_sent(self) -> None:
        from tests.unit.test_http_adapter import _make_adapter

        adapter, calls = _make_adapter({"POST /api/v1/retrieve": {"entries": []}})
        adapter.retrieve("how do I rotate the key")
        adapter.retrieve("how do I rotate the key", branch="main")
        adapter.retrieve("how do I rotate the key", branch="repair/fix-1")
        assert "branch" not in calls[0]["body"]
        assert "branch" not in calls[1]["body"]
        assert calls[2]["body"]["branch"] == "repair/fix-1"


class TestAgentMemoryBinds:
    @pytest.fixture
    def seen(self) -> list[httpx.Request]:
        return []

    @pytest.fixture
    def memory(self, seen):
        from amfs import AgentMemory

        return AgentMemory("sre-agent", adapter=_adapter(seen), session_id="sess-42")

    def test_reads_carry_the_memorys_agent_and_session(self, memory, seen) -> None:
        memory.read("acme/billing", "k")
        assert seen[0].headers[AGENT_ID_HEADER] == "sre-agent"
        assert seen[0].headers[SESSION_HEADER] == "sess-42"

    def test_as_agent_carries_its_own_agent_in_the_same_session(self, memory, seen) -> None:
        memory.as_agent("billing-agent").read("acme/billing", "k")
        memory.read("acme/billing", "k")
        assert seen[0].headers[AGENT_ID_HEADER] == "billing-agent"
        assert seen[0].headers[SESSION_HEADER] == "sess-42"
        assert seen[1].headers[AGENT_ID_HEADER] == "sre-agent"

    def test_the_situation_attribute_is_declared_before_the_first_read(self, memory, seen) -> None:
        """What ``Run.begin(situation=...)`` does: the bag gets the attribute,
        the adapter gets the header, every read that follows carries it."""
        memory.read("acme/billing", "k")
        memory.set_session_attributes({"situation": "spend summary", "customer": "acme"})
        memory.read("acme/billing", "k")
        memory.retrieve("how much did we spend")
        assert SITUATION_HEADER not in seen[0].headers
        assert seen[1].headers[SITUATION_HEADER] == "spend summary"
        assert seen[2].headers[SITUATION_HEADER] == "spend summary"
        # Other attributes do not touch the header.
        memory.set_session_attributes({"region": "eu"})
        memory.read("acme/billing", "k")
        assert seen[3].headers[SITUATION_HEADER] == "spend summary"

    def test_the_header_is_dropped_with_the_bag(self, memory, seen) -> None:
        memory.set_session_attributes({"situation": "spend summary"})
        memory.clear_session_attributes()
        memory.read("acme/billing", "k")
        assert SITUATION_HEADER not in seen[-1].headers
        memory.set_session_attributes({"situation": "renewal quote"})
        memory.read("acme/billing", "k")
        assert seen[-1].headers[SITUATION_HEADER] == "renewal quote"

    def test_the_header_is_dropped_when_the_trace_is_committed(self, memory, seen) -> None:
        memory.set_session_attributes({"situation": "spend summary"})
        memory.read("acme/billing", "k")
        assert seen[-1].headers[SITUATION_HEADER] == "spend summary"
        from amfs_core.models import OutcomeType

        memory.commit_outcome("t-1", OutcomeType.SUCCESS)
        assert memory.session_attributes == {}
        memory.read("acme/billing", "k")
        assert SITUATION_HEADER not in seen[-1].headers

    def test_as_agent_starts_without_a_situation(self, memory, seen) -> None:
        memory.set_session_attributes({"situation": "spend summary"})
        memory.as_agent("billing-agent").read("acme/billing", "k")
        memory.read("acme/billing", "k")
        assert SITUATION_HEADER not in seen[0].headers
        assert seen[1].headers[SITUATION_HEADER] == "spend summary"

    def test_run_begin_declares_its_situation_on_the_retrieve(self, memory, seen) -> None:
        from amfs import Run

        Run(memory).begin("how much did acme spend", entity_path="acme/billing",
                          situation="spend summary")
        retrieves = [r for r in seen if r.url.path.endswith("/retrieve")]
        assert retrieves and retrieves[0].headers[SITUATION_HEADER] == "spend summary"

    def test_the_mcp_retrieve_declares_the_situation_it_was_given(
        self, memory, seen, monkeypatch
    ) -> None:
        """An agent on the OSS MCP server has no ``Run.begin``; the situation it
        passes to ``amfs_retrieve`` is what it knows. It goes on the bag, and
        so on the header, before the retrieve is sent."""
        from amfs_mcp import server as mcp_server

        monkeypatch.setattr(mcp_server, "_get_memory", lambda: memory)
        mcp_server.amfs_retrieve("how much did acme spend", situation="spend summary")
        retrieves = [r for r in seen if r.url.path.endswith("/retrieve")]
        assert retrieves and retrieves[0].headers[SITUATION_HEADER] == "spend summary"
        assert memory.session_attributes["situation"] == "spend summary"

    def test_adapters_without_bind_are_left_alone(self, tmp_path) -> None:
        from amfs import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        fs = FilesystemAdapter(tmp_path / "amfs")
        mem = AgentMemory("sre-agent", adapter=fs)
        assert mem.adapter is fs
        assert mem.as_agent("other").adapter is fs

    def test_a_bind_that_raises_does_not_break_construction(self, seen) -> None:
        from amfs import AgentMemory

        class _Broken(HttpAdapter):
            def bind(self, agent_id: Any, session_id: Any) -> HttpAdapter:
                raise RuntimeError("no")

        adapter = _Broken.__new__(_Broken)
        mem = AgentMemory("sre-agent", adapter=adapter)
        assert mem.adapter is adapter

    def test_a_bind_that_returns_nothing_leaves_the_adapter_in_place(self, tmp_path) -> None:
        """A test double whose ``__getattr__`` answers every name with a
        no-op, or a wrapper that forgot to return: the handle keeps its
        adapter rather than ending up with ``None``."""
        from amfs import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        class _Forgetful(FilesystemAdapter):
            def bind(self, agent_id: Any, session_id: Any) -> None:
                return None

        fs = _Forgetful(tmp_path / "amfs")
        mem = AgentMemory("sre-agent", adapter=fs)
        assert mem.adapter is fs
        assert mem.as_agent("other").adapter is fs
