"""Per-attempt credit: a task that fails on one memory and succeeds on another.

Before attempt boundaries existed, such a task committed a single ``success``
that cited every entry read — including the one whose advice failed — so the
stale entry was *reinforced* by the agent's recovery from it. These tests pin
the new behaviour end to end: SDK read log → ``record_attempt`` →
``commit_outcome`` → adapter → evidence on each entry → trace metadata →
contrast lesson, and the same through the HTTP server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from amfs import AgentMemory
from amfs.memory import FINAL_ACTION_INDEX_ATTRIBUTE, SESSION_ATTEMPTS_KEY
from amfs_core import evidence as ev
from amfs_core.models import MemoryType, OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(tmp_amfs_root: Path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
    m = AgentMemory(agent_id="support-agent", adapter=adapter)
    m.write("acme/support", "fix-restart", "restart the worker", confidence=0.9)
    m.write("acme/support", "fix-rotate-key", "rotate the API key", confidence=0.7)
    m._read_tracker.clear()
    return m


def _fail_then_succeed(mem: AgentMemory) -> list:
    mem.read("acme/support", "fix-restart")
    mem.record_action("restart_worker", {"service": "ingest"}, result="queue still stuck", success=False)
    attempt = mem.record_attempt(summary="restart per runbook did nothing")
    assert attempt.attempt == 1
    assert attempt.causal_entry_keys == ["acme/support/fix-restart"]
    assert attempt.action_indices == [0]

    mem.read("acme/support", "fix-rotate-key")
    mem.record_action("rotate_key", {"service": "ingest"}, result="queue drained")
    return mem.commit_outcome("ticket-1", OutcomeType.SUCCESS, task_input="ingest queue stuck")


class TestSdkAttemptBoundaries:
    def test_failed_attempt_entry_is_discredited_and_final_entry_validated(self, mem: AgentMemory) -> None:
        updated = {e.key: e for e in _fail_then_succeed(mem)}
        assert set(updated) == {"fix-restart", "fix-rotate-key"}
        stale = updated["fix-restart"]
        good = updated["fix-rotate-key"]
        assert stale.failure_count == 1 and stale.success_count == 0
        assert stale.confidence < ev.DISCREDIT_THRESHOLD
        assert stale.evidence_status == "discredited"
        assert good.success_count == 1 and good.failure_count == 0
        assert good.confidence > 0.7
        assert good.evidence_status == "validated"

    def test_trace_carries_attempts_and_final_action_index(self, mem: AgentMemory) -> None:
        _fail_then_succeed(mem)
        trace = mem._last_trace
        assert trace is not None
        meta = trace.session_metadata.model_dump()
        attempts = meta[SESSION_ATTEMPTS_KEY]
        assert len(attempts) == 1
        assert attempts[0]["causal_entry_keys"] == ["acme/support/fix-restart"]
        assert attempts[0]["action_indices"] == [0]
        assert meta["attributes"][FINAL_ACTION_INDEX_ATTRIBUTE] == 1
        assert [tc.tool_name for tc in trace.tool_calls] == ["restart_worker", "rotate_key"]
        # The terminal outcome cites only what the resolving attempt read.
        assert [c.key for c in trace.causal_entries] == ["fix-rotate-key"]

    def test_contrast_lesson_is_written_as_synthetic_experience(self, mem: AgentMemory) -> None:
        _fail_then_succeed(mem)
        lesson = mem.read("acme/support", ev.contrast_lesson_key("ticket-1"))
        assert lesson is not None
        assert ev.is_synthetic_key(lesson.key)
        assert lesson.memory_type == MemoryType.EXPERIENCE
        assert lesson.value["avoid"] == ["acme/support/fix-restart"]
        assert lesson.value["resolved_with"] == ["acme/support/fix-rotate-key"]
        assert lesson.provenance.agent_id == "support-agent"
        assert set(lesson.provenance.pattern_refs) == {"fix-restart", "fix-rotate-key"}

    def test_no_boundary_means_old_behaviour(self, mem: AgentMemory) -> None:
        mem.read("acme/support", "fix-restart")
        mem.record_action("restart_worker", {})
        updated = mem.commit_outcome("ticket-2", OutcomeType.SUCCESS)
        assert [e.key for e in updated] == ["fix-restart"]
        meta = mem._last_trace.session_metadata
        assert meta is None or SESSION_ATTEMPTS_KEY not in meta.model_dump()
        assert mem.read("acme/support", ev.contrast_lesson_key("ticket-2")) is None

    def test_failure_after_attempts_still_records_them(self, mem: AgentMemory) -> None:
        mem.read("acme/support", "fix-restart")
        mem.record_attempt()
        mem.read("acme/support", "fix-rotate-key")
        updated = {e.key: e for e in mem.commit_outcome("ticket-3", OutcomeType.FAILURE)}
        assert updated["fix-restart"].failure_count == 1
        assert updated["fix-rotate-key"].failure_count == 1
        assert mem.read("acme/support", ev.contrast_lesson_key("ticket-3")) is None

    def test_clear_resets_boundaries(self, mem: AgentMemory) -> None:
        mem.read("acme/support", "fix-restart")
        mem.record_attempt()
        mem.commit_outcome("ticket-4", OutcomeType.FAILURE)
        assert mem._read_tracker.attempts == []
        assert mem._read_tracker.final_action_index is None


class TestHttpServerAttempts:
    @pytest.fixture
    def server_mem(self, monkeypatch, tmp_path) -> AgentMemory:
        from amfs_http import server

        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        handle = AgentMemory(agent_id="http-server", adapter=adapter)
        monkeypatch.setattr(server, "_memory", handle)
        monkeypatch.setattr(server, "_get_memory", lambda: handle)
        monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
        monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
        monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
        monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
        handle.write("acme/support", "fix-restart", "restart the worker", confidence=0.9)
        handle.write("acme/support", "fix-rotate-key", "rotate the API key", confidence=0.7)
        handle._read_tracker.clear()
        return handle

    @pytest.fixture
    def client(self, server_mem):
        from amfs_http import server
        from fastapi.testclient import TestClient

        return TestClient(server.app)

    def test_outcome_with_attempts_credits_each_side(self, client, server_mem: AgentMemory) -> None:
        body = {
            "outcome_ref": "ticket-9",
            "outcome_type": "success",
            "agent_id": "remote-agent",
            "causal_entry_keys": ["acme/support/fix-rotate-key"],
            "tool_calls": [
                {"tool_name": "restart_worker", "arguments": {}, "success": False},
                {"tool_name": "rotate_key", "arguments": {}},
            ],
            "attempts": [
                {"attempt": 1, "outcome_type": "failure",
                 "causal_entry_keys": ["acme/support/fix-restart"], "action_indices": [0]}
            ],
            "final_action_index": 1,
            "trace_follows": True,
        }
        resp = client.post("/api/v1/outcomes", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        by_key = {e["key"]: e for e in data["entries"]}
        assert by_key["fix-restart"]["failure_count"] == 1
        assert by_key["fix-restart"]["discredited_at"] is not None
        assert by_key["fix-rotate-key"]["success_count"] == 1
        # The server wrote the lesson under the caller's identity.
        lesson = server_mem.read("acme/support", ev.contrast_lesson_key("ticket-9"))
        assert lesson is not None
        assert lesson.provenance.agent_id == "remote-agent"

    def test_malformed_attempts_are_a_422(self, client) -> None:
        body = {
            "outcome_ref": "ticket-10",
            "outcome_type": "success",
            "causal_entry_keys": [],
            "attempts": [{"attempt": "one"}],
        }
        assert client.post("/api/v1/outcomes", json=body).status_code == 422

    def test_final_action_index_must_index_tool_calls(self, client) -> None:
        body = {
            "outcome_ref": "ticket-11",
            "outcome_type": "success",
            "causal_entry_keys": [],
            "tool_calls": [{"tool_name": "x", "arguments": {}}],
            "final_action_index": 3,
        }
        assert client.post("/api/v1/outcomes", json=body).status_code == 422
