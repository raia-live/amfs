"""An entry whose key contains a slash can be read back.

``/api/v1/entries/{entity_path:path}/{key}`` cannot express one. The path
converter is greedy, so everything up to the final segment becomes the entity
path: a read of ``sweep/abc`` + ``active-negotiation/xyz`` arrives as
``sweep/abc/active-negotiation`` + ``xyz``. Nothing was written there, so the
server answers "not found" — a confident, wrong answer that looks exactly like
an entry that never existed.

314 of the 4,726 entries in production have such a key. The negotiation engine
writes them, and so does anything else that namespaces its keys. None of them
could be read back.

The OpenAI ``fetch`` tool is the sharpest edge: it exists to resolve an id that
``search`` has just handed out, so search would offer an entry and fetch would
deny it existed — a contradiction between the only two tools the connector
specification actually requires.
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

ENTITY = "sweep/abc"
SLASHED_KEY = "active-negotiation/xyz"


@pytest.fixture()
def asked() -> list[tuple[str, str]]:
    """The coordinates the read actually reached storage with."""
    return []


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, asked: list) -> TestClient:
    stored = {
        (ENTITY, SLASHED_KEY): MemoryEntry(
            entity_path=ENTITY,
            key=SLASHED_KEY,
            value="a proposal nobody could read",
            provenance=Provenance(
                agent_id="builder-agent",
                session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
    }

    def _read(entity_path, key, branch="main", **_kw):
        asked.append((entity_path, key))
        return stored.get((entity_path, key))

    mem = MagicMock()
    mem.namespace = "test-ns"
    mem._tagger = SimpleNamespace(agent_id="srv", session_id="sess")
    mem.read.side_effect = _read

    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda _r: None)
    return TestClient(server.app)


class TestTheQueryRouteFindsIt:
    def test_the_entry_comes_back(self, client) -> None:
        resp = client.get(
            "/api/v1/entry", params={"entity_path": ENTITY, "key": SLASHED_KEY}
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["key"] == SLASHED_KEY

    def test_storage_was_asked_for_the_right_coordinates(
        self, client, asked
    ) -> None:
        """The failure this fixes is a wrong question, not a missing answer."""
        client.get(
            "/api/v1/entry", params={"entity_path": ENTITY, "key": SLASHED_KEY}
        )

        assert asked == [(ENTITY, SLASHED_KEY)]

    def test_a_key_that_is_genuinely_absent_is_still_not_found(
        self, client
    ) -> None:
        resp = client.get(
            "/api/v1/entry",
            params={"entity_path": ENTITY, "key": "nothing/here"},
        )

        assert resp.json()["status"] == "not_found"


class TestThePathRouteStillCannot:
    """Kept as a record of why the other route exists.

    This is not a bug left in place: the path form has no way to say where the
    entity ends and the key begins, and both are user-supplied. The fix is to
    ask a different question, not to guess better.
    """

    def test_it_splits_at_the_last_segment(self, client, asked) -> None:
        client.get(f"/api/v1/entries/{ENTITY}/{SLASHED_KEY}")

        assert asked == [("sweep/abc/active-negotiation", "xyz")]


class TestOrdinaryKeysAreUnaffected:
    def test_the_path_route_still_answers(self, client, asked) -> None:
        client.get(f"/api/v1/entries/{ENTITY}/plain-key")

        assert asked == [(ENTITY, "plain-key")]
