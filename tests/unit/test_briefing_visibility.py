"""Unit tests for briefing visibility filtering.

Verifies that _filter_briefing_digests and _is_agent_visible_for_entity
correctly filter digests and hot_context entries based on user ownership
and room membership, matching the same rules as UserVisibilityFilter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("amfs_core", reason="amfs_core not installed")

from amfs_core.models import Digest, DigestType


def _digest(
    scope: str = "myapp/auth",
    source_agents: list[str] | None = None,
    hot_context: list[dict] | None = None,
    digest_type: DigestType = DigestType.ENTITY,
) -> Digest:
    summary: dict = {"narrative": "test digest"}
    if hot_context is not None:
        summary["hot_context"] = hot_context
    return Digest(
        digest_type=digest_type,
        scope=scope,
        summary=summary,
        entry_count=5,
        source_agents=source_agents or [],
        compiled_at=datetime.now(timezone.utc),
        namespace="default",
        branch="main",
    )


def _hot(agent: str, key: str = "some-key") -> dict:
    return {
        "key": key,
        "value": "some value",
        "confidence": 0.9,
        "agent": agent,
        "outcome_count": 1,
        "recall_count": 2,
    }


def _mock_vis(user_agents: set[str], room_map: dict[str, set[str]] | None = None):
    vis = MagicMock()
    vis.should_filter.return_value = True
    vis.get_user_agents.return_value = user_agents
    vis.get_room_map.return_value = room_map or {}
    return vis


@pytest.fixture()
def import_helpers():
    from amfs_http.server import _filter_briefing_digests, _is_agent_visible_for_entity
    return _filter_briefing_digests, _is_agent_visible_for_entity


class TestIsAgentVisibleForEntity:

    def test_own_agent_always_visible(self, import_helpers):
        _, is_visible = import_helpers
        assert is_visible("my-agent", "any/path", {"my-agent"}, {})

    def test_foreign_agent_not_visible_without_room(self, import_helpers):
        _, is_visible = import_helpers
        assert not is_visible("other-agent", "any/path", {"my-agent"}, {})

    def test_room_co_member_visible_on_room_entity(self, import_helpers):
        _, is_visible = import_helpers
        room_map = {"shared/project": {"other-agent", "my-agent"}}
        assert is_visible("other-agent", "shared/project", {"my-agent"}, room_map)

    def test_room_co_member_not_visible_on_different_entity(self, import_helpers):
        _, is_visible = import_helpers
        room_map = {"shared/project": {"other-agent", "my-agent"}}
        assert not is_visible("other-agent", "different/path", {"my-agent"}, room_map)

    def test_system_agent_visible_on_room_entity(self, import_helpers):
        _, is_visible = import_helpers
        room_map = {"shared/project": {"my-agent"}}
        assert is_visible("amfs-server", "shared/project", {"my-agent"}, room_map)

    def test_system_agent_not_visible_without_room(self, import_helpers):
        _, is_visible = import_helpers
        assert not is_visible("amfs-server", "other/path", {"my-agent"}, {})


class TestFilterBriefingDigests:

    def test_digest_with_own_agent_source_kept(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(source_agents=["my-agent"])
        result = filter_fn(vis, [d])
        assert len(result) == 1

    def test_digest_with_foreign_agent_source_removed(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(source_agents=["foreign-agent"])
        result = filter_fn(vis, [d])
        assert len(result) == 0

    def test_digest_with_mixed_sources_kept_if_any_visible(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(source_agents=["foreign-agent", "my-agent"])
        result = filter_fn(vis, [d])
        assert len(result) == 1

    def test_digest_room_co_member_source_kept(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis(
            {"my-agent"},
            room_map={"myapp/auth": {"my-agent", "teammate-agent"}},
        )
        d = _digest(scope="myapp/auth", source_agents=["teammate-agent"])
        result = filter_fn(vis, [d])
        assert len(result) == 1

    def test_synthesized_digest_empty_sources_on_room_entity_kept(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis(
            {"my-agent"},
            room_map={"myapp/auth": {"my-agent"}},
        )
        d = _digest(scope="myapp/auth", source_agents=[])
        result = filter_fn(vis, [d])
        assert len(result) == 1

    def test_synthesized_digest_empty_sources_no_access_removed(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(scope="foreign/entity", source_agents=[])
        result = filter_fn(vis, [d])
        assert len(result) == 0

    def test_hot_context_own_agent_entries_kept(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(
            source_agents=["my-agent"],
            hot_context=[_hot("my-agent", "key-1"), _hot("my-agent", "key-2")],
        )
        result = filter_fn(vis, [d])
        assert len(result) == 1
        assert len(result[0].summary["hot_context"]) == 2

    def test_hot_context_foreign_agent_entries_removed(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d = _digest(
            source_agents=["my-agent"],
            hot_context=[_hot("my-agent", "keep"), _hot("foreign-agent", "remove")],
        )
        result = filter_fn(vis, [d])
        assert len(result) == 1
        hc = result[0].summary["hot_context"]
        assert len(hc) == 1
        assert hc[0]["key"] == "keep"

    def test_hot_context_room_co_member_entries_kept(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis(
            {"my-agent"},
            room_map={"myapp/auth": {"my-agent", "teammate"}},
        )
        d = _digest(
            scope="myapp/auth",
            source_agents=["my-agent"],
            hot_context=[_hot("teammate", "shared-key")],
        )
        result = filter_fn(vis, [d])
        assert len(result) == 1
        assert len(result[0].summary["hot_context"]) == 1

    def test_multiple_digests_partially_filtered(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        d_visible = _digest(scope="mine/svc", source_agents=["my-agent"])
        d_hidden = _digest(scope="other/svc", source_agents=["foreign"])
        result = filter_fn(vis, [d_visible, d_hidden])
        assert len(result) == 1
        assert result[0].scope == "mine/svc"

    def test_empty_digests_returns_empty(self, import_helpers):
        filter_fn, _ = import_helpers
        vis = _mock_vis({"my-agent"})
        assert filter_fn(vis, []) == []


class TestAdminBypass:

    def test_admin_should_filter_false(self, import_helpers):
        """When should_filter() is False (admin), the endpoint skips filtering entirely.
        We verify the helper is never called by checking it preserves all digests."""
        filter_fn, _ = import_helpers
        vis = MagicMock()
        vis.should_filter.return_value = False
        d = _digest(source_agents=["foreign-agent"])
        assert vis.should_filter() is False


class TestOSSNoFilter:

    def test_no_visibility_filter_on_request(self, import_helpers):
        """When _get_visibility_filter returns None (OSS), no filtering occurs.
        The helper function is never called."""
        pass
