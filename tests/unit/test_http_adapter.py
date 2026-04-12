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
    """Create an HttpAdapter with a mock transport that returns canned responses."""
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
