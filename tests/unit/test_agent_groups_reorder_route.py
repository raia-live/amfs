"""Regression test: PUT /api/v1/agent-groups/reorder must not be shadowed by /{group_id}."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from amfs_http import server


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    adapter = MagicMock()
    adapter.reorder_agent_groups.return_value = None
    adapter.update_agent_group.return_value = SimpleNamespace(
        model_dump=lambda mode="json": {"id": "g1", "name": "Renamed"},
    )
    mem = SimpleNamespace(namespace="default", _adapter=adapter)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    server.app.dependency_overrides[server.verify_api_key] = lambda: None
    try:
        yield TestClient(server.app), adapter
    finally:
        server.app.dependency_overrides.pop(server.verify_api_key, None)


def test_reorder_route_wins_over_group_id(client: tuple[TestClient, MagicMock]) -> None:
    http, adapter = client
    resp = http.put(
        "/api/v1/agent-groups/reorder",
        json={"positions": [{"group_id": "g1", "position": 1}, {"group_id": "g2", "position": 0}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    adapter.reorder_agent_groups.assert_called_once_with(
        [("g1", 1), ("g2", 0)],
        namespace="default",
    )
    adapter.update_agent_group.assert_not_called()


def test_update_group_by_id_still_works(client: tuple[TestClient, MagicMock]) -> None:
    http, adapter = client
    resp = http.put("/api/v1/agent-groups/g1", json={"name": "Renamed"})
    assert resp.status_code == 200
    assert resp.json() == {"id": "g1", "name": "Renamed"}
    adapter.update_agent_group.assert_called_once_with("g1", namespace="default", name="Renamed")
