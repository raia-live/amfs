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
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from amfs_adapter_http import AGENT_ID_HEADER, SESSION_HEADER, HttpAdapter


def _adapter(seen: list[httpx.Request]) -> HttpAdapter:
    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
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
