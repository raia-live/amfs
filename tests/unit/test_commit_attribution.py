"""A batch commit is credited to the caller, not to the server that runs it.

The batch used to be assembled client-side: each entry went over as its own
write carrying the caller's agent, and the caller stitched a commit together
afterwards. Moving the transaction onto the server made the group genuinely one
round trip, and took the caller's name off it in the process — everything now
ran through the server's own long-lived ``AgentMemory``, so the entries were
credited to the server process and so was the commit.

The commit part is the one that shows. ``GET /api/v1/commits`` hides commits
whose author the caller is not allowed to see, and the server's own agent is
never one of those, so a caller could not see the commit they had just made:
``amfs_commit_log`` answered zero for an account with commits sitting in it.
The entries were quieter about it and no less wrong, since which agent knows
what is the thing this system is for.

These mirror ``test_write_session_provenance``, which pinned the same rule for
``POST /api/v1/entries``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402

SERVER_SESSION = "sess-server"
SERVER_AGENT = "amfs-server"

WRITES = [{"entity_path": "svc", "key": "k", "value": "v"}]


@pytest.fixture()
def tagger() -> SimpleNamespace:
    return SimpleNamespace(agent_id=SERVER_AGENT, session_id=SERVER_SESSION)


@pytest.fixture()
def observed() -> dict:
    """What the tagger held at the moment the transaction actually ran."""
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

    class _Tx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, entity_path, key, value, **options):
            # Read at write time, not afterwards: the point of the fix is that
            # the tagger holds the caller's name for the duration, and a check
            # made after the block would pass even if it were restored early.
            observed["agent_id"] = tagger.agent_id
            observed["session_id"] = tagger.session_id

        commit = SimpleNamespace(
            id="c-1", model_dump=lambda mode="json": {"id": "c-1"}
        )
        entries = ["one"]

    mem.transaction.return_value = _Tx()
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    return TestClient(server.app)


def _post(client: TestClient, headers=None, **extra):
    body = {"writes": WRITES, "message": "batch"}
    body.update(extra)
    resp = client.post("/api/v1/commits", json=body, headers=headers or {})
    assert resp.status_code == 200, resp.text
    return resp


class TestTheCallerIsCredited:
    def test_the_agent_reaches_the_transaction(self, client, observed) -> None:
        _post(client, agent_id="builder-agent", session_id="sess-caller")
        assert observed["agent_id"] == "builder-agent"

    def test_the_session_reaches_it_too(self, client, observed) -> None:
        _post(client, agent_id="builder-agent", session_id="sess-caller")
        assert observed["session_id"] == "sess-caller"


class TestTheHeaderIsTheFallback:
    """The gateway has always sent X-AMFS-Agent-Id on every request.

    Reading it means a gateway already in production is attributed correctly
    from the moment this deploys, rather than after a client release.
    """

    def test_the_header_is_used_when_the_body_is_silent(
        self, client, observed
    ) -> None:
        _post(client, headers={"X-AMFS-Agent-Id": "header-agent"})
        assert observed["agent_id"] == "header-agent"

    def test_the_body_wins_when_both_are_present(self, client, observed) -> None:
        _post(
            client,
            headers={"X-AMFS-Agent-Id": "header-agent"},
            agent_id="body-agent",
        )
        assert observed["agent_id"] == "body-agent"


class TestTheServerTaggerIsLeftAsItWasFound:
    """This memory is a process singleton, so a tagger left holding one
    caller's name signs the next caller's writes with it."""

    def test_the_agent_is_restored(self, client, tagger) -> None:
        _post(client, agent_id="builder-agent", session_id="sess-caller")
        assert tagger.agent_id == SERVER_AGENT

    def test_the_session_is_restored(self, client, tagger) -> None:
        _post(client, agent_id="builder-agent", session_id="sess-caller")
        assert tagger.session_id == SERVER_SESSION

    def test_it_is_restored_even_when_the_batch_is_rejected(
        self, client, tagger
    ) -> None:
        resp = client.post(
            "/api/v1/commits",
            json={
                "writes": [
                    {
                        "entity_path": "svc",
                        "key": "k",
                        "value": "v",
                        "memory_type": "nonsense",
                    }
                ],
                "agent_id": "builder-agent",
            },
        )

        assert resp.status_code == 422
        assert tagger.agent_id == SERVER_AGENT


class TestOlderClientsStillCommit:
    """A caller that names nobody gets the behaviour it had."""

    def test_the_server_agent_is_the_fallback(self, client, observed) -> None:
        _post(client)
        assert observed["agent_id"] == SERVER_AGENT
