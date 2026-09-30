"""Retrieve on a real repair branch leads with the branch's correction (step 9b).

The Postgres adapter's branches, diff and overlay reads, through the server's
``/api/v1/retrieve`` on the lexical channel: a correction written on the branch
at 60% of the stale rule's confidence — under the caller's gate — is the first
hit on the branch and absent on main. Requires a running Postgres instance;
set AMFS_TEST_PG_DSN to enable.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import Branch

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(PG_DSN is None, reason="AMFS_TEST_PG_DSN not set")

SCOPE = "clbench/ci-fix-itest"
BRANCH = "repair/itest"


@pytest.fixture
def mem(monkeypatch):
    from amfs_http import server
    from amfs_postgres.adapter import PostgresAdapter

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    ns = f"bt-{uuid.uuid4().hex[:8]}"
    adapter = PostgresAdapter(dsn=PG_DSN, namespace=ns, auto_schema=True)
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
    handle.write(SCOPE, "pip-audit-add-exception",
                 "pip audit pinned dependency: add a scoped audit exception", confidence=0.95)
    handle.write(SCOPE, "pip-audit-exception-again",
                 "pip audit pinned dependency vulnerable: add an audit exception", confidence=0.9)
    handle.write(SCOPE, "jest-snapshot-ui",
                 "jest snapshot diff on a ui pr: update the snapshot", confidence=0.9)
    adapter.create_branch(Branch(
        namespace=ns, name=BRANCH, parent_branch="main",
        branched_at=datetime.now(UTC), created_by="test",
    ))
    handle.write(SCOPE, "pip-audit-corrected",
                 "pip audit pinned dependency: bump the pin to the fixed version unless the PR "
                 "touches legacy sdk files, then add a scoped exception",
                 confidence=0.45, branch=BRANCH)
    handle._read_tracker.clear()
    yield handle
    adapter.close()


@pytest.fixture
def client(mem):
    from amfs_http import server
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _hits(resp) -> list[dict]:
    return [e for e in resp.json() if not e.get("_avoid") and not e.get("_meta")]


def test_the_branch_diff_names_the_correction(mem) -> None:
    diff = mem._adapter.diff_branch(BRANCH, namespace=mem._adapter._namespace)
    assert [(d.entity_path, d.key, d.diff_type) for d in diff] == [
        (SCOPE, "pip-audit-corrected", "added"),
    ]


def test_main_read_does_not_see_the_branch(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits and "pip-audit-corrected" not in [h["key"] for h in hits]
    assert not any(h.get("_pinned") for h in hits)


def test_branch_read_leads_with_the_correction_under_the_gate(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "pip-audit-corrected", [h["key"] for h in hits]
    assert hits[0]["_pinned"] is True
    assert hits[0]["confidence"] < 0.5  # under the gate, and still the treatment
    assert "pip-audit-add-exception" in [h["key"] for h in hits[1:]]


def test_an_unrelated_query_on_the_branch_is_not_pinned(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "jest snapshot", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "jest-snapshot-ui"
    assert not any(h.get("_pinned") for h in hits)
