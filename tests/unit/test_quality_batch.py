"""One request scores a whole topic's entries.

The topic page asked ``/quality/{path}/{key}`` once per entry, and each call
listed the topic again to find the sibling keys: a 365-entry topic was 365
round trips of about a second, serialised three at a time, before anything
rendered — 82 s from the first call to the last in production. The batch
route lists each topic once and scores every requested key from that list.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_http import server
from fastapi.testclient import TestClient

PATH = "acme/checkout"
LONG = "The checkout service retries a failed payment capture three times with backoff."


@pytest.fixture
def mem(monkeypatch, tmp_path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="amfs-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_active_visibility_filter", lambda request: None)
    return handle


@pytest.fixture
def client(mem) -> TestClient:
    return TestClient(server.app)


def _single(client: TestClient, key: str) -> float:
    return client.get(f"/api/v1/quality/{PATH}/{key}").json()["quality"]["score"]


def test_the_batch_agrees_with_the_single_route_and_keeps_order(client, mem):
    mem.write(PATH, "retries", LONG, confidence=0.9)
    mem.write(PATH, "timeout", "30s", confidence=0.5)
    mem.write("acme/other", "note", LONG, confidence=0.8)

    body = client.post(
        "/api/v1/quality/batch",
        json={"items": [
            {"entity_path": PATH, "key": "timeout"},
            {"entity_path": "acme/other", "key": "note"},
            {"entity_path": PATH, "key": "retries"},
        ]},
    ).json()

    assert [(r["entity_path"], r["key"]) for r in body["results"]] == [
        (PATH, "timeout"), ("acme/other", "note"), (PATH, "retries"),
    ]
    assert body["results"][0]["score"] == _single(client, "timeout")
    assert body["results"][2]["score"] == _single(client, "retries")
    # The sibling-aware part of the score is the same: an entry with
    # siblings and no pattern_refs scores lower than one standing alone.
    alone = client.get("/api/v1/quality/acme/other/note").json()["quality"]["score"]
    assert body["results"][1]["score"] == alone
    assert body["results"][2]["score"] < alone


def test_a_missing_or_hidden_entry_is_a_null_score_not_an_error(client, mem, monkeypatch):
    mem.write(PATH, "retries", LONG, confidence=0.9)
    mem.write(PATH, "secret", LONG, confidence=0.9)

    class _Vis:
        def filter_entries(self, rows):
            return [r for r in rows if r.key != "secret"]

    monkeypatch.setattr(server, "_active_visibility_filter", lambda request: _Vis())
    body = client.post(
        "/api/v1/quality/batch",
        json={"items": [
            {"entity_path": PATH, "key": "retries"},
            {"entity_path": PATH, "key": "secret"},
            {"entity_path": PATH, "key": "nope"},
            {"entity_path": "acme/empty", "key": "nope"},
        ]},
    ).json()
    scores = [r["score"] for r in body["results"]]
    assert scores[0] is not None and scores[1:] == [None, None, None]


def test_another_agents_private_entry_is_scored_like_the_single_route(client, mem):
    """``mem.list`` on the server's handle keeps only shared entries and the
    server's own; the single route reads through the adapter and scores a
    private entry regardless. The batch must agree, or a topic page that
    lists the entry shows no score for it."""
    mem.as_agent("other-agent").write(PATH, "private", LONG, confidence=0.9, shared=False)
    assert [e.key for e in mem.list(PATH)] == [], "precondition: hidden from the handle"

    single = client.get(f"/api/v1/quality/{PATH}/private").json()["quality"]["score"]
    batch = client.post(
        "/api/v1/quality/batch", json={"items": [{"entity_path": PATH, "key": "private"}]}
    ).json()["results"][0]["score"]
    assert batch == single and batch is not None


def test_each_topic_is_listed_once(client, mem, monkeypatch):
    for i in range(6):
        mem.write(PATH, f"k{i}", LONG, confidence=0.9)
    calls: list[str] = []
    adapter = mem._adapter
    real = adapter.list

    def counting(entity_path=None, **kw):
        calls.append(entity_path)
        return real(entity_path, **kw)

    monkeypatch.setattr(adapter, "list", counting)
    client.post(
        "/api/v1/quality/batch",
        json={"items": [{"entity_path": PATH, "key": f"k{i}"} for i in range(6)]},
    )
    assert calls == [PATH]


def test_the_batch_is_capped(client):
    res = client.post(
        "/api/v1/quality/batch",
        json={"items": [{"entity_path": PATH, "key": f"k{i}"} for i in range(2001)]},
    )
    assert res.status_code == 422
