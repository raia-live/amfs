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

    def test_include_artifacts_false_forwarded(self) -> None:
        # Dropping this flag made every SaaS caller's include_artifacts=False a
        # no-op, so a search could come back carrying whole source files.
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        adapter.search(SearchQuery(entity_path="svc", include_artifacts=False))

        assert calls[0]["body"]["include_artifacts"] is False

    def test_include_artifacts_omitted_when_default(self) -> None:
        # Left out when True so older servers keep their own default.
        adapter, calls = _make_adapter({
            "POST /api/v1/search": {"entries": []},
        })
        adapter.search(SearchQuery(entity_path="svc"))

        assert "include_artifacts" not in calls[0]["body"]


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


class TestWriteCarriesTheCallersSession:
    """The server builds provenance from what this body says.

    Anything omitted here is filled in from the server's own tagger, which is
    one process serving every agent — so a dropped session_id silently
    relabels the write with a session that has nothing to do with the caller,
    and the decision trace (keyed on the caller's session) can no longer be
    joined to the entries it produced.
    """

    def _write(self):
        from datetime import datetime, timezone

        from amfs_core.models import MemoryEntry, Provenance

        adapter, calls = _make_adapter({
            "POST /api/v1/entries": {
                "entity_path": "svc",
                "key": "k",
                "value": "v",
                "provenance": {
                    "agent_id": "agent-1",
                    "session_id": "sess-caller",
                    "written_at": "2026-01-01T00:00:00Z",
                },
            },
        })
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="v",
            provenance=Provenance(
                agent_id="agent-1",
                session_id="sess-caller",
                written_at=datetime.now(timezone.utc),
            ),
        )
        adapter.write(entry)
        return calls

    def test_the_session_id_is_sent(self) -> None:
        assert self._write()[0]["body"]["session_id"] == "sess-caller"

    def test_the_agent_id_is_still_sent(self) -> None:
        assert self._write()[0]["body"]["agent_id"] == "agent-1"


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


def _make_error_adapter(status_code: int, body: Any):
    """Create an HttpAdapter whose transport always returns an error response."""
    from amfs_adapter_http.adapter import HttpAdapter

    def _handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, (dict, list)):
            return httpx.Response(status_code, json=body)
        return httpx.Response(status_code, text=body)

    adapter = HttpAdapter.__new__(HttpAdapter)
    adapter._base = "http://test"
    adapter._api_key = "test-key"
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "test-key"},
        transport=httpx.MockTransport(_handler),
    )
    return adapter


class TestErrorDetailSurfaced:
    """4xx/5xx responses must expose the server's `detail` in the exception
    message so MCP agents can read the guidance and self-correct (e.g. the
    409 agent-identity collision tells them to call amfs_set_identity)."""

    def test_409_detail_in_message(self) -> None:
        detail = (
            "Agent identity 'agent/dai' is already registered to a different "
            "user in this account. Set a unique agent identity and retry."
        )
        adapter = _make_error_adapter(409, {"detail": detail})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert detail in str(exc_info.value)
        assert exc_info.value.response.status_code == 409

    def test_update_agent_profile_surfaces_detail(self) -> None:
        detail = "Agent identity 'agent/dai' is already registered to a different user"
        adapter = _make_error_adapter(409, {"detail": detail})

        profile = MagicMock()
        profile.model_dump.return_value = {"display_name": "Dai"}

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter.update_agent_profile("agent/dai", profile)

        assert detail in str(exc_info.value)

    def test_error_key_used_as_fallback(self) -> None:
        adapter = _make_error_adapter(403, {"error": "restricted member"})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert "restricted member" in str(exc_info.value)

    def test_non_json_body_falls_back_to_plain_raise(self) -> None:
        adapter = _make_error_adapter(500, "internal server error")

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/entries")

        assert exc_info.value.response.status_code == 500

    def test_404_still_catchable_by_status_code(self) -> None:
        # entity_summaries() and friends catch HTTPStatusError and check
        # response.status_code == 404 — the enriched error must preserve that.
        adapter = _make_error_adapter(404, {"detail": "not found"})

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter._request("GET", "/api/v1/missing")

        assert exc_info.value.response.status_code == 404
