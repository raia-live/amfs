"""Which session the shared memory handle's tracker describes.

``_get_memory`` returns one ``AgentMemory`` for the whole process, so a single
``ReadTracker`` sits behind every request. It accumulates what the reads returned —
the value, the confidence, who wrote it — along with the contexts, queries and
writes of the session, and it is emptied only inside ``commit_outcome``. So what
one request left there was still in place for the next, and the two things that
read it back described a session that was not the caller's:

* ``explain`` serves its causal entries from those snapshots and returns the
  accumulated contexts.
* ``commit_outcome`` falls back to the same snapshots for its causal entries when
  the caller names none, and takes the contexts, query events and state diff from
  them with no way to pass its own.

Clients that name their causal entries were never affected, and since the
trace-integrity fixes an SDK client declares ``trace_follows`` and no server-side
trace is built at all.

The fix scopes the tracker's state to the request rather than replacing the
tracker, because the instance is captured at construction — ``CoWEngine`` holds one
— so changing what a caller sees needs no cooperation from anything holding a
reference. These tests go through the ASGI stack rather than calling handlers,
because the scope is installed by middleware and its registration order is part of
what has to be right.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_core.models import MemoryType
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_http import server
from fastapi.testclient import TestClient

EARLIER_PATH = "acme/billing"
EARLIER_KEY = "key-rotation-runbook"
EARLIER_VALUE = "rotate from the dashboard, then redeploy the workers"
EARLIER_CONTEXT = "three SEV-1s in the last 24h"


@pytest.fixture
def mem(monkeypatch, tmp_path) -> AgentMemory:
    """The server's one shared handle, as ``_get_memory`` returns it."""
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="http-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
    return handle


@pytest.fixture
def client(mem) -> TestClient:
    return TestClient(server.app)


def _an_earlier_request_reads_and_records(client: TestClient, mem: AgentMemory) -> None:
    """Traffic from a request that is over: an entry read back, a context recorded.

    The write goes in directly, being only setup. The read goes through a request so
    that the snapshot lands in the tracker the way a real read leaves it.
    """
    mem.write(EARLIER_PATH, EARLIER_KEY, EARLIER_VALUE, memory_type=MemoryType.FACT)

    resp = client.get(f"/api/v1/entries/{EARLIER_PATH}/{EARLIER_KEY}")
    assert resp.status_code == 200, resp.text
    assert EARLIER_VALUE in resp.text, "the read returned nothing; the setup is wrong"

    resp = client.post(
        "/api/v1/context",
        json={"label": "upstream", "summary": EARLIER_CONTEXT, "source": "PagerDuty"},
    )
    assert resp.status_code == 200, resp.text


def test_explain_reports_only_this_requests_reads(client, mem) -> None:
    """No commit is needed to reach this one, which is why it is first."""
    _an_earlier_request_reads_and_records(client, mem)

    body = client.get("/api/v1/explain").json()

    stale = [e for e in body["causal_entries"] if e["entity_path"] == EARLIER_PATH]
    assert not stale, (
        f"{len(stale)} entries from a finished request, with their values: "
        f"{[e.get('value') for e in stale]}"
    )
    assert body["causal_chain_length"] == 0


def test_explain_reports_only_this_requests_contexts(client, mem) -> None:
    """Contexts arrive by a different route than reads.

    Asserted separately because a fix reaching only the read snapshots would leave
    these behind, and they are free text from whatever tooling recorded them.
    """
    _an_earlier_request_reads_and_records(client, mem)

    body = client.get("/api/v1/explain").json()

    assert not [c for c in body["external_contexts"] if EARLIER_CONTEXT in c["summary"]]
    assert body["external_contexts"] == []


def test_a_committed_trace_carries_only_this_requests_reads(client, mem) -> None:
    """The write path: a caller that names no causal entries got the tracker's."""
    saved: list = []
    original = mem._adapter.save_trace

    def _capture(trace):
        saved.append(trace)
        return original(trace)

    mem._adapter.save_trace = _capture  # type: ignore[method-assign]

    _an_earlier_request_reads_and_records(client, mem)

    resp = client.post(
        "/api/v1/outcomes",
        json={
            "outcome_ref": "deploy-142",
            "outcome_type": "success",
            "agent_id": "sre-agent",
            "task_input": "ship the checkout fix",
        },
    )
    assert resp.status_code == 200, resp.text

    assert saved, "no trace was written; this test would pass vacuously"
    trace = saved[-1]
    assert not [e for e in trace.causal_entries if e.entity_path == EARLIER_PATH], (
        "an entry from a finished request was written into this trace as a cause, "
        "and would be sealed into its immutable copy"
    )
    assert not [c for c in trace.external_contexts if EARLIER_CONTEXT in c["summary"]]


def test_a_request_still_sees_its_own_reads(client, mem) -> None:
    """The scope must not be so tight that it breaks what the tracker is for.

    Reads and the commit that follows them inside one request belong together, and a
    fix that emptied the tracker per call rather than per request would silently
    stop causal linking working at all.
    """
    from amfs_core.engine import read_tracker_scope

    mem.write(EARLIER_PATH, EARLIER_KEY, EARLIER_VALUE, memory_type=MemoryType.FACT)

    with read_tracker_scope():
        mem.read(EARLIER_PATH, EARLIER_KEY)
        mem.record_context("own-ctx", "something this request learned")

        assert mem._read_tracker.read_count == 1
        explained = mem.explain()
        assert explained["causal_chain_length"] == 1
        assert explained["causal_entries"][0]["value"] == EARLIER_VALUE
        assert len(explained["external_contexts"]) == 1


def test_the_next_request_starts_clean(client, mem) -> None:
    """Two scopes in sequence, which is what two requests are."""
    from amfs_core.engine import read_tracker_scope

    mem.write(EARLIER_PATH, EARLIER_KEY, EARLIER_VALUE, memory_type=MemoryType.FACT)

    with read_tracker_scope():
        mem.read(EARLIER_PATH, EARLIER_KEY)
        assert mem._read_tracker.read_count == 1

    with read_tracker_scope():
        assert mem._read_tracker.read_count == 0
        assert mem.explain()["causal_entries"] == []


def test_a_single_agent_process_is_unchanged(tmp_path) -> None:
    """No scope is installed outside the server, and none may be needed.

    Every local SDK and MCP user relies on the tracker accumulating across calls —
    that is the auto-causal linking the library is built on. This pins the change to
    the process that shares a handle.
    """
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="local-agent", adapter=adapter)

    handle.write("app/mod", "k", "v", memory_type=MemoryType.FACT)
    handle.read("app/mod", "k")
    handle.record_context("ctx", "an external input")

    explained = handle.explain()
    assert explained["causal_chain_length"] == 1
    assert len(explained["external_contexts"]) == 1


def test_the_session_window_belongs_to_the_request(tmp_path) -> None:
    """``session_started_at`` is session state too, and duration comes from it.

    Left on the tracker it reported the lifetime of the process, which is the wrong
    clock the sealed traces were measured against.
    """
    from datetime import datetime, timezone

    from amfs_core.engine import ReadTracker, read_tracker_scope

    tracker = ReadTracker()
    process_start = tracker.session_started_at
    assert process_start.tzinfo is not None

    with read_tracker_scope():
        assert tracker.session_started_at >= process_start
        # Writable, because ``clear()`` and the Pro span recorder both backdate it.
        marker = datetime(2026, 1, 1, tzinfo=timezone.utc)
        tracker._session_started_at = marker
        assert tracker.session_started_at == marker

    assert tracker.session_started_at == process_start, (
        "the scope's window escaped into the tracker's own state"
    )
