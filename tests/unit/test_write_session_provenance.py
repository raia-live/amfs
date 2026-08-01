"""A write is stamped with the caller's session, not the server's.

The hosted MCP runs client-side with an ``HttpAdapter`` pointed at this API, so
the agent's session lives in the client process and the write is served by a
long-lived server process shared by every agent on the box. The server builds
provenance from the request body, which means anything the body leaves out is
silently filled in from the server's own tagger.

For ``session_id`` that is not a harmless default. It relabels the entry with a
session that belongs to the server process — one id shared by every write it
ever serves — while the DecisionTrace for the same work is keyed on the
client's session. The two can then never be joined, so "which trace produced
this entry" has no answer, and the session shown against an entry means nothing.

The agent id already worked this way. These tests hold the session to the same
rule, and pin the fallback so older clients that send no session still write.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402
from amfs_core.models import MemoryEntry, Provenance  # noqa: E402

SERVER_SESSION = "sess-server"
SERVER_AGENT = "server-agent"


@pytest.fixture()
def tagger() -> SimpleNamespace:
    return SimpleNamespace(agent_id=SERVER_AGENT, session_id=SERVER_SESSION)


@pytest.fixture()
def observed() -> dict:
    """What the tagger held at the moment the write actually happened."""
    return {}


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    tagger: SimpleNamespace,
    observed: dict,
) -> TestClient:
    mem = MagicMock()
    mem.namespace = "test-ns"
    mem._tagger = tagger

    def _capture(entity_path, key, value, **kwargs):
        observed["agent_id"] = tagger.agent_id
        observed["session_id"] = tagger.session_id
        return MemoryEntry(
            entity_path=entity_path,
            key=key,
            value=value,
            provenance=Provenance(
                agent_id=tagger.agent_id,
                session_id=tagger.session_id,
                written_at=datetime.now(timezone.utc),
            ),
        )

    mem.write.side_effect = _capture
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    return TestClient(server.app)


def _post(client: TestClient, **extra) -> None:
    body = {"entity_path": "svc", "key": "k", "value": "v"}
    body.update(extra)
    resp = client.post("/api/v1/entries", json=body)
    assert resp.status_code == 200, resp.text


class TestTheCallersSessionIsUsed:
    def test_it_reaches_the_write(self, client, observed) -> None:
        _post(client, agent_id="agent-1", session_id="sess-caller")
        assert observed["session_id"] == "sess-caller"

    def test_the_agent_id_still_reaches_the_write(self, client, observed) -> None:
        _post(client, agent_id="agent-1", session_id="sess-caller")
        assert observed["agent_id"] == "agent-1"


class TestTheServerTaggerIsLeftAsItWasFound:
    """The tagger is process-global, so a write that does not put it back
    leaks one caller's session onto the next caller's write."""

    def test_the_session_is_restored(self, client, tagger) -> None:
        _post(client, agent_id="agent-1", session_id="sess-caller")
        assert tagger.session_id == SERVER_SESSION

    def test_the_agent_is_restored(self, client, tagger) -> None:
        _post(client, agent_id="agent-1", session_id="sess-caller")
        assert tagger.agent_id == SERVER_AGENT


class TestOlderClientsStillWrite:
    """amfs-adapter-http before this change sends no session_id at all."""

    def test_the_server_session_is_the_fallback(self, client, observed) -> None:
        _post(client, agent_id="agent-1")
        assert observed["session_id"] == SERVER_SESSION

    def test_the_write_is_not_rejected(self, client, observed) -> None:
        _post(client, agent_id="agent-1")
        assert observed["agent_id"] == "agent-1"
