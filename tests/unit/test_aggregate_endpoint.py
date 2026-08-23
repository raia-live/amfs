"""Endpoint tests for POST /api/v1/aggregate.

Pins the two properties that keep it correct and safe:
- entity_path is required and drives scoping (unscoped would drop room data);
- the visibility filter runs BEFORE reducing, so a restricted user can never
  aggregate over entries they cannot read.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


def _entry(value, key="k", agent="my-agent"):
    return types.SimpleNamespace(
        value=value,
        key=key,
        provenance=types.SimpleNamespace(agent_id=agent),
    )


class _AdminVis:
    def should_filter(self):
        return False

    def filter_entries(self, entries):
        return list(entries)


class _RestrictedVis:
    def __init__(self, agents=("my-agent",)):
        self._agents = set(agents)

    def should_filter(self):
        return True

    def filter_entries(self, entries):
        return [e for e in entries if e.provenance.agent_id in self._agents]


def _client(monkeypatch, entries, vis):
    mem = types.SimpleNamespace(
        namespace="test-ns",
        list=lambda entity_path=None, branch="main": list(entries),
    )
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
    return TestClient(server.app)


def test_count_over_row_path(monkeypatch):
    entries = [
        _entry({"listings": [{"price": 100}, {"price": 300}]}, key="b1"),
        _entry({"listings": [{"price": 200}]}, key="b2"),
    ]
    client = _client(monkeypatch, entries, _AdminVis())
    res = client.post("/api/v1/aggregate", json={
        "entity_path": "@room/listings", "op": "count", "row_path": "listings",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 3
    assert body["entity_path"] == "@room/listings"


def test_mean_grouped(monkeypatch):
    entries = [_entry({"listings": [
        {"price": 100, "city": "A"},
        {"price": 300, "city": "B"},
        {"price": 200, "city": "A"},
    ]}, key="b1")]
    client = _client(monkeypatch, entries, _AdminVis())
    res = client.post("/api/v1/aggregate", json={
        "entity_path": "@room/listings", "op": "mean", "field": "price",
        "group_by": "city", "row_path": "listings",
    })
    assert res.status_code == 200
    groups = {g["key"]: g for g in res.json()["groups"]}
    assert groups["A"]["mean"] == 150


def test_visibility_filter_applied_before_reduce(monkeypatch):
    # Two entries, only one authored by the visible agent. A restricted user's
    # count must not include the foreign entry's rows.
    entries = [
        _entry({"listings": [{"p": 1}, {"p": 2}]}, key="mine", agent="my-agent"),
        _entry({"listings": [{"p": 3}, {"p": 4}, {"p": 5}]}, key="theirs", agent="other-agent"),
    ]
    client = _client(monkeypatch, entries, _RestrictedVis(agents=("my-agent",)))
    res = client.post("/api/v1/aggregate", json={
        "entity_path": "@room/listings", "op": "count", "row_path": "listings",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2  # only the visible entry's two rows
    assert body["entries_scanned"] == 1


def test_unknown_op_is_400(monkeypatch):
    client = _client(monkeypatch, [_entry({"p": 1})], _AdminVis())
    res = client.post("/api/v1/aggregate", json={"entity_path": "@room/x", "op": "median", "field": "p"})
    assert res.status_code == 400


def test_numeric_op_without_field_is_400(monkeypatch):
    client = _client(monkeypatch, [_entry({"p": 1})], _AdminVis())
    res = client.post("/api/v1/aggregate", json={"entity_path": "@room/x", "op": "sum"})
    assert res.status_code == 400


def test_entity_path_is_required(monkeypatch):
    client = _client(monkeypatch, [_entry({"p": 1})], _AdminVis())
    res = client.post("/api/v1/aggregate", json={"op": "count"})
    assert res.status_code == 422  # pydantic: missing required field
