"""Regression tests for invited-user tenant isolation on the OSS HTTP server.

A non-admin (invited) member of a multi-user account must only see data
authored by their own agents or shared via rooms. These tests pin the shared
visibility helpers and a representative set of gated endpoints:

- ``_active_visibility_filter`` / ``_visible_agent_ids`` /
  ``_require_agent_visible`` / ``_require_account_admin`` semantics.
- ``/api/v1/agents/{agent_id}/timeline`` 404s for hidden agents.
- ``/api/v1/admin/teams`` (and members) return empty for restricted members.
"""

from __future__ import annotations

import types

import pytest
from unittest.mock import MagicMock

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


class _RestrictedVis:
    """UserVisibilityFilter stand-in for a non-admin invited member."""

    def __init__(self, agents=("my-agent",)):
        self._agents = set(agents)

    def should_filter(self):
        return True

    def get_visible_agent_ids(self):
        return set(self._agents)

    def get_user_agents(self):
        return set(self._agents)

    def is_agent_visible(self, agent_id):
        return agent_id in self._agents

    def get_room_map(self):
        return {}

    def filter_entries(self, entries):
        out = []
        for e in entries:
            agent = getattr(getattr(e, "provenance", None), "agent_id", None)
            if agent in self._agents:
                out.append(e)
        return out


class _AdminVis:
    def should_filter(self):
        return False


def _request(vis=None):
    return types.SimpleNamespace(state=types.SimpleNamespace(visibility_filter=vis))


# ---------------------------------------------------------------------------
# Helper semantics
# ---------------------------------------------------------------------------


class TestVisibilityHelpers:
    def test_active_filter_none_without_context(self):
        assert server._active_visibility_filter(_request(None)) is None

    def test_active_filter_none_for_admin(self):
        assert server._active_visibility_filter(_request(_AdminVis())) is None

    def test_active_filter_set_for_restricted(self):
        vis = _RestrictedVis()
        assert server._active_visibility_filter(_request(vis)) is vis

    def test_visible_agent_ids_unrestricted_is_none(self):
        assert server._visible_agent_ids(_request(None)) is None

    def test_visible_agent_ids_restricted(self):
        assert server._visible_agent_ids(_request(_RestrictedVis())) == {"my-agent"}

    def test_visible_agent_ids_denies_on_error(self):
        class _Broken(_RestrictedVis):
            def get_visible_agent_ids(self):
                raise RuntimeError("db down")

        assert server._visible_agent_ids(_request(_Broken())) == set()

    def test_require_agent_visible_404_for_foreign(self):
        with pytest.raises(HTTPException) as exc:
            server._require_agent_visible(_request(_RestrictedVis()), "other-agent")
        assert exc.value.status_code == 404

    def test_require_agent_visible_passes_for_own(self):
        server._require_agent_visible(_request(_RestrictedVis()), "my-agent")

    def test_require_account_admin_403_for_restricted(self):
        with pytest.raises(HTTPException) as exc:
            server._require_account_admin(_request(_RestrictedVis()))
        assert exc.value.status_code == 403

    def test_require_account_admin_passes_for_admin_and_no_context(self):
        server._require_account_admin(_request(_AdminVis()))
        server._require_account_admin(_request(None))


# ---------------------------------------------------------------------------
# Endpoint behavior through the app
# ---------------------------------------------------------------------------


@pytest.fixture()
def restricted_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    adapter = MagicMock()
    adapter.list_events.return_value = []

    mem = MagicMock()
    mem.namespace = "test-ns"
    mem._adapter = adapter
    mem.list.return_value = []

    vis = _RestrictedVis(agents=("my-agent",))
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
    monkeypatch.setattr(
        server, "_active_visibility_filter",
        lambda request: vis if vis.should_filter() else None,
    )
    return TestClient(server.app)


class TestTimelineIsolation:
    def test_foreign_agent_timeline_404(self, restricted_client: TestClient):
        res = restricted_client.get("/api/v1/agents/other-agent/timeline")
        assert res.status_code == 404

    def test_own_agent_timeline_ok(self, restricted_client: TestClient):
        res = restricted_client.get("/api/v1/agents/my-agent/timeline")
        assert res.status_code == 200
        assert res.json()["agentId"] == "my-agent"


class TestAdminEndpointsRestricted:
    def test_list_teams_empty(self, restricted_client: TestClient):
        res = restricted_client.get("/api/v1/admin/teams")
        assert res.status_code == 200
        assert res.json() == {"teams": []}

    def test_list_team_members_empty(self, restricted_client: TestClient):
        res = restricted_client.get(
            "/api/v1/admin/teams/00000000-0000-0000-0000-000000000000/members"
        )
        assert res.status_code == 200
        assert res.json() == {"members": []}
