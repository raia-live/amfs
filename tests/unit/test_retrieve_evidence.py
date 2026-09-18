"""Evidence-aware retrieval: outcomes change what comes back, not just a number.

Runs the server's ``/api/v1/retrieve`` over the filesystem adapter (lexical
fallback, no embedder), which exercises the discredited exclusion, the
evidence term, the avoid list and adaptive k without needing Postgres.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import OutcomeType, RecallConfig
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(monkeypatch, tmp_path) -> AgentMemory:
    from amfs_http import server

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="http-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: None)
    monkeypatch.setattr(server, "_retrieval_reranker", None)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
    # Three fixes for the same symptom, same author confidence.
    handle.write(
        "acme/support", "fix-restart", "queue stuck: restart the ingest worker", confidence=0.8
    )
    handle.write(
        "acme/support", "fix-rotate", "queue stuck: rotate the ingest API key", confidence=0.8
    )
    handle.write(
        "acme/support", "fix-scale", "queue stuck: scale the ingest consumers", confidence=0.8
    )
    handle._read_tracker.clear()
    return handle


@pytest.fixture
def client(mem):
    from amfs_http import server
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _outcome(
    mem: AgentMemory, key: str, outcome: OutcomeType, ref: str, entity_path: str = "acme/support"
) -> None:
    mem.read(entity_path, key)
    mem.commit_outcome(ref, outcome)


def _keys(resp) -> list[str]:
    return [e["key"] for e in resp.json() if not e.get("_avoid")]


def test_discredited_entry_drops_out_and_validated_one_leads(client, mem: AgentMemory) -> None:
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, "t2")
    resp = client.post("/api/v1/retrieve", json={"query": "queue stuck", "limit": 10})
    assert resp.status_code == 200, resp.text
    keys = _keys(resp)
    assert "fix-restart" not in keys
    assert keys[0] == "fix-rotate"
    top = resp.json()[0]
    assert top["evidence_status"] == "validated"
    assert top["_breakdown"]["evidence"] > 0
    untested = next(e for e in resp.json() if e["key"] == "fix-scale")
    assert untested["evidence_status"] == "untested"
    assert untested["_breakdown"]["evidence"] == 0


def test_include_discredited_ranks_it_last(client, mem: AgentMemory) -> None:
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    resp = client.post(
        "/api/v1/retrieve",
        json={
            "query": "queue stuck",
            "limit": 10,
            "include_discredited": True,
        },
    )
    keys = _keys(resp)
    assert keys[-1] == "fix-restart"
    assert resp.json()[-1]["_breakdown"]["evidence"] == -1.0


def test_avoid_list_is_flagged_and_appended(client, mem: AgentMemory) -> None:
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    resp = client.post(
        "/api/v1/retrieve",
        json={
            "query": "queue stuck",
            "limit": 10,
            "include_avoid": True,
        },
    )
    rows = resp.json()
    avoid = [e for e in rows if e.get("_avoid")]
    assert [e["key"] for e in avoid] == ["fix-restart"]
    assert avoid[0]["_score"] == 0.0
    assert avoid[0]["_breakdown"]["evidence_status"] == "discredited"
    assert avoid[0]["_breakdown"]["failure_count"] == 1
    # Appended after the real hits, never interleaved.
    assert rows.index(avoid[0]) == len(rows) - 1


def test_compact_avoid_rows_are_previews(client, mem: AgentMemory) -> None:
    """In compact mode an avoided entry is carried as a one-liner, like a
    tail hit: its job is to name what stopped working, not to be read in
    full. It keeps the fields the agent and lineage need."""
    long_value = "queue stuck: restart the worker. " + "then check the consumer lag; " * 30
    mem.write("acme/support", "fix-restart", long_value, confidence=0.9)
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    rows = client.post(
        "/api/v1/retrieve",
        json={"query": "queue stuck", "limit": 10, "include_avoid": True, "compact": True},
    ).json()
    avoid = [e for e in rows if e.get("_avoid")]
    assert [e["key"] for e in avoid] == ["fix-restart"]
    assert avoid[0]["value_truncated"] is True and len(avoid[0]["value"]) < 200
    assert avoid[0]["_breakdown"]["evidence_status"] == "discredited"
    assert avoid[0]["evidence_status"] == "discredited" and "version" in avoid[0]


def test_adaptive_k_shrinks_behind_a_validated_leader(client, mem: AgentMemory) -> None:
    for i in range(3):
        _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, f"s{i}")
    full = client.post("/api/v1/retrieve", json={"query": "queue stuck", "limit": 10})
    assert len(_keys(full)) == 3
    shrunk = client.post(
        "/api/v1/retrieve",
        json={
            "query": "queue stuck",
            "limit": 10,
            "adaptive_k": True,
        },
    )
    keys = _keys(shrunk)
    assert keys[0] == "fix-rotate"
    assert len(keys) < 3


def test_adaptive_k_leaves_untested_leader_alone(client) -> None:
    resp = client.post(
        "/api/v1/retrieve",
        json={
            "query": "queue stuck",
            "limit": 10,
            "adaptive_k": True,
        },
    )
    assert len(_keys(resp)) == 3


def test_sdk_local_scoring_honours_evidence(mem: AgentMemory) -> None:
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, "t2")
    scored = mem.search(query="queue", entity_path="acme/support", recall_config=RecallConfig())
    keys = [s.entry.key for s in scored]
    assert "fix-restart" not in keys
    assert keys[0] == "fix-rotate"
    assert scored[0].breakdown["evidence_status"] == "validated"
    with_disc = mem.search(
        query="queue",
        entity_path="acme/support",
        recall_config=RecallConfig(include_discredited=True),
    )
    assert [s.entry.key for s in with_disc][-1] == "fix-restart"


def test_sdk_over_http_gets_the_avoid_list_and_does_not_book_it(mem: AgentMemory) -> None:
    """The whole rail: RecallConfig.include_avoid -> HttpAdapter -> server ->
    ``_avoid`` rows -> ScoredEntry.breakdown -> ``is_avoid``. And the avoid row
    must not enter the causal chain, or the next outcome would touch the very
    entry the agent was warned off."""
    import httpx
    from amfs.memory import is_avoid
    from amfs_adapter_http.adapter import HttpAdapter
    from amfs_http import server

    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, "t2")

    from fastapi.testclient import TestClient

    inner = TestClient(server.app)

    def _handler(request: httpx.Request) -> httpx.Response:
        resp = inner.request(
            request.method, request.url.path, content=request.content,
            headers={"content-type": "application/json"},
        )
        return httpx.Response(resp.status_code, json=resp.json())

    adapter = HttpAdapter.__new__(HttpAdapter)
    adapter._base = "http://test"
    adapter._api_key = "k"
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "k"},
        transport=httpx.MockTransport(_handler),
    )
    client = AgentMemory(agent_id="ops-agent", adapter=adapter)

    results = client.retrieve(
        "queue stuck", limit=10, recall_config=RecallConfig(include_avoid=True)
    )
    hits = [r for r in results if not is_avoid(r)]
    avoided = [r for r in results if is_avoid(r)]
    assert [r.entry.key for r in avoided] == ["fix-restart"]
    assert avoided[0].entry.evidence_status == "discredited"
    assert hits[0].entry.key == "fix-rotate"
    # Only the top real hit is booked; the avoid row never is.
    booked = client._read_tracker.causal_keys
    assert "acme/support/fix-rotate" in booked
    assert "acme/support/fix-restart" not in booked


def test_mcp_retrieve_reports_avoid_separately(mem: AgentMemory, monkeypatch) -> None:
    import json

    from amfs_mcp import server as mcp_server

    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "t1")
    _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, "t2")
    # Point the MCP tool at a memory whose adapter answers retrieve like the
    # server does: reuse the server-side handle through a stub adapter.retrieve.
    from amfs_http import server as http_server

    def _retrieve(query, **kwargs):
        from fastapi.testclient import TestClient

        body = {"query": query, "limit": kwargs.get("limit", 10),
                "include_avoid": kwargs.get("include_avoid", False)}
        rows = TestClient(http_server.app).post("/api/v1/retrieve", json=body).json()
        from amfs_adapter_http.adapter import _parse_entry

        out = []
        for e in rows:
            breakdown = dict(e.get("_breakdown") or {})
            if e.get("_avoid"):
                breakdown["_avoid"] = True
            out.append((_parse_entry(e), float(e.get("_score", 0.0)), breakdown))
        return out

    monkeypatch.setattr(mem._adapter, "retrieve", _retrieve, raising=False)
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: mem)
    payload = json.loads(mcp_server.amfs_retrieve("queue stuck", limit=10))
    assert [e["key"] for e in payload["entries"]] and all(
        e["key"] != "fix-restart" for e in payload["entries"]
    )
    assert [a["key"] for a in payload["avoid"]] == ["fix-restart"]
    assert payload["avoid"][0]["failure_count"] == 1
    assert "avoid_note" in payload
    assert payload["entries"][0]["evidence_status"] == "validated"

