"""Retrieve on a repair branch leads with what the branch changed (step 9b).

A read routed to a repair branch is the canary arm of a fix. The repair loop
writes its correction at 60% of the confidence of the rule it corrects, onto
a scope where outcome learning has written validated near-duplicates of the
stale rule; ranked on confidence the correction never reaches the head, and
the canary compares two untreated arms (clbench, 2026-09-30: 18 canaries,
zero reads of the fix's own entry). Over the filesystem adapter (lexical
fallback, no vectors) with the branch diff stubbed: what the branch changed
and the query is about is pinned first and flagged; main reads and unrelated
queries are untouched.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import DiffEntry
from amfs_filesystem.adapter import FilesystemAdapter

SCOPE = "clbench/ci-fix"
BRANCH = "repair/fix1"


class _BranchingFilesystem(FilesystemAdapter):
    """The filesystem store with one branch that changed one entry. Reads
    ignore the branch (the store has no overlay); the diff is what step 9b
    consults, and the pinned entry is read back from the same store."""

    changed: list[tuple[str, str]] = []

    def diff_branch(self, name: str, namespace: str = "default") -> list[DiffEntry]:
        if name != BRANCH:
            return []
        return [
            DiffEntry(entity_path=p, key=k, diff_type="modified", branch_value="x")
            for p, k in self.changed
        ]


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(monkeypatch, tmp_path) -> AgentMemory:
    from amfs_http import server

    adapter = _BranchingFilesystem(root=tmp_path / ".amfs", namespace="test")
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
    # The stale rule and two validated-looking near-duplicates, then the
    # correction the repair loop would write: 60% of the rule's confidence,
    # under the 0.5 gate the benchmark reads with.
    handle.write(SCOPE, "pip-audit-add-exception",
                 "pip audit pinned dependency: add a scoped audit exception", confidence=0.95)
    handle.write(SCOPE, "pip-audit-exception-again",
                 "pip audit pinned dependency vulnerable: add an audit exception", confidence=0.9)
    handle.write(SCOPE, "pip-audit-exception-legacy",
                 "pip audit pinned dependency legacy sdk: audit exception", confidence=0.9)
    handle.write(SCOPE, "pip-audit-corrected",
                 "pip audit pinned dependency: bump the pin to the fixed version unless the PR "
                 "touches legacy sdk files, then add a scoped exception", confidence=0.45)
    handle.write(SCOPE, "jest-snapshot-ui",
                 "jest snapshot diff on a ui pr: update the snapshot", confidence=0.9)
    adapter.changed = [(SCOPE, "pip-audit-corrected")]
    handle._read_tracker.clear()
    return handle


@pytest.fixture
def client(mem):
    from amfs_http import server
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _hits(resp) -> list[dict]:
    return [e for e in resp.json() if not e.get("_avoid") and not e.get("_meta")]


def test_main_read_never_sees_the_correction_under_the_gate(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5,
    })
    assert resp.status_code == 200, resp.text
    keys = [h["key"] for h in _hits(resp)]
    assert "pip-audit-corrected" not in keys
    assert not any(h.get("_pinned") for h in _hits(resp))


def test_branch_read_leads_with_what_the_branch_changed(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "pip-audit-corrected"
    assert hits[0]["_pinned"] is True
    assert hits[0]["_breakdown"]["pinned"] == BRANCH
    # The rest of the ranking is the ranking: the stale rule is still there,
    # below the treatment, unflagged.
    rest = [h["key"] for h in hits[1:]]
    assert "pip-audit-add-exception" in rest
    assert not any(h.get("_pinned") for h in hits[1:])


def test_a_query_the_change_is_not_about_is_left_alone(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "jest snapshot diff", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "jest-snapshot-ui"
    assert "pip-audit-corrected" not in [h["key"] for h in hits]


def test_a_branch_that_changed_nothing_here_pins_nothing(client, mem) -> None:
    mem._adapter.changed = [("other/scope", "pip-audit-corrected")]
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH,
    })
    assert resp.status_code == 200, resp.text
    assert not any(h.get("_pinned") for h in _hits(resp))


def test_compact_rows_carry_the_flag(client) -> None:
    resp = client.post("/api/v1/retrieve", json={
        "query": "pip audit pinned dependency", "entity_path": SCOPE,
        "min_confidence": 0.5, "limit": 5, "branch": BRANCH, "compact": True,
    })
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "pip-audit-corrected" and hits[0]["_pinned"] is True


def test_a_pinned_procedure_still_answers_to_the_environment(client, mem) -> None:
    """The branch's change is a procedure that wants python3.12. Pinning is
    not a way around environment scoping: for a python3.9 run it is dropped
    and named in ``_meta.not_applicable`` like any other procedure; for a
    python3.12 run it leads the list, flagged."""
    from amfs_core.models import MemoryType

    mem.write(SCOPE, "pip-audit-procedure", {
        "goal": "pip audit pinned dependency: bump the pin to the fixed version",
        "preconditions": {"runtime": "python3.12"},
        "steps": ["bump the pin", "run pip audit", "open the PR"],
    }, confidence=0.45, memory_type=MemoryType.PROCEDURE)
    mem._adapter.changed = [(SCOPE, "pip-audit-procedure")]
    mem._read_tracker.clear()
    body = {"query": "pip audit pinned dependency", "entity_path": SCOPE,
            "min_confidence": 0.5, "limit": 5, "branch": BRANCH}

    resp = client.post("/api/v1/retrieve", json={**body, "environment": {"runtime": "python3.9"}})
    assert resp.status_code == 200, resp.text
    assert "pip-audit-procedure" not in [h["key"] for h in _hits(resp)]
    assert not any(h.get("_pinned") for h in _hits(resp))
    meta = next(e for e in resp.json() if e.get("_meta"))
    assert [r["key"] for r in meta["not_applicable"]] == ["pip-audit-procedure"]
    assert meta["not_applicable"][0]["why"] == ["runtime: wants python3.12, run has python3.9"]

    # Above the gate the procedure is a candidate as well as pinned, and both
    # filters drop it: it is named once.
    resp = client.post("/api/v1/retrieve", json={
        **body, "min_confidence": 0.0, "environment": {"runtime": "python3.9"},
    })
    assert resp.status_code == 200, resp.text
    assert "pip-audit-procedure" not in [h["key"] for h in _hits(resp)]
    meta = next(e for e in resp.json() if e.get("_meta"))
    assert [r["key"] for r in meta["not_applicable"]] == ["pip-audit-procedure"]

    resp = client.post("/api/v1/retrieve", json={**body, "environment": {"runtime": "python3.12"}})
    assert resp.status_code == 200, resp.text
    hits = _hits(resp)
    assert hits[0]["key"] == "pip-audit-procedure" and hits[0]["_pinned"] is True
