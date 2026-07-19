"""Server-side guard against duplicate WRITE timeline events.

The SaaS MCP runs client-side via ``uvx amfs-mcp-server-pro`` with an
``HttpAdapter`` pointed at the hosted HTTP API. Both the server's write handler
(``POST /api/v1/entries``) and the client SDK's background log path emitted a
``WRITE`` event, producing two identical timeline events per write.

The write handler is the single source of truth for WRITE events, so the
generic timeline endpoint must drop client-originated WRITE events. This is
version-agnostic — it fixes every client regardless of the SDK version pulled
from PyPI, and takes effect the moment the hosted server is redeployed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


@pytest.fixture()
def spy_memory(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a mock AgentMemory so we can spy on adapter.log_event."""
    mem = MagicMock()
    mem.namespace = "test-ns"

    def _echo_log_event(event):
        return event

    mem._adapter.log_event.side_effect = _echo_log_event
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    return mem


@pytest.fixture()
def client(spy_memory: MagicMock) -> TestClient:
    return TestClient(server.app)


def test_write_event_is_not_persisted(client: TestClient, spy_memory: MagicMock) -> None:
    resp = client.post(
        "/api/v1/timeline/events",
        json={
            "agent_id": "agent-1",
            "event_type": "write",
            "summary": "wrote bruno/senselab/foo",
            "details": {"entity_path": "bruno/senselab", "key": "foo", "version": 1},
        },
    )
    assert resp.status_code == 200
    # The write handler owns WRITE events — the client's redundant one is dropped.
    spy_memory._adapter.log_event.assert_not_called()
    # A well-formed event is still echoed back so clients don't error.
    body = resp.json()
    assert body["event_type"] == "write"
    assert body["agent_id"] == "agent-1"


def test_non_write_event_is_persisted(client: TestClient, spy_memory: MagicMock) -> None:
    resp = client.post(
        "/api/v1/timeline/events",
        json={
            "agent_id": "agent-1",
            "event_type": "read",
            "summary": "read bruno/senselab/foo",
            "details": {"entity_path": "bruno/senselab", "key": "foo"},
        },
    )
    assert resp.status_code == 200
    spy_memory._adapter.log_event.assert_called_once()
    logged_event = spy_memory._adapter.log_event.call_args.args[0]
    assert logged_event.event_type.value == "read"


def test_outcome_event_is_persisted(client: TestClient, spy_memory: MagicMock) -> None:
    resp = client.post(
        "/api/v1/timeline/events",
        json={"agent_id": "agent-1", "event_type": "outcome", "summary": "ok"},
    )
    assert resp.status_code == 200
    spy_memory._adapter.log_event.assert_called_once()
