"""Tests for ReadTracker hygiene in read()/recall() and our_last_read in write(on_conflict)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from amfs import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter


def test_reading_other_agents_private_entry_does_not_pollute_causal_chain(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem_a = AgentMemory(agent_id="agent-a", adapter=adapter)
    mem_a.write("svc/auth", "secret-note", {"secret": "private-val"}, shared=False)

    mem_b = mem_a.as_agent("agent-b")
    assert mem_b.read("svc/auth", "secret-note") is None

    assert mem_b.read_log == []
    assert mem_b.explain()["causal_entries"] == []
    assert adapter.read("svc/auth", "secret-note").recall_count == 0


def test_reading_entry_below_decayed_min_confidence_does_not_record_read(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="agent-a", adapter=adapter, decay_half_life_days=1.0)
    mem.write("svc/auth", "old-note", "stale", confidence=0.5)

    # Backdate written_at by 10 days so decayed confidence falls well below 0.4
    current_file = adapter._layout.current_version_file("svc/auth", "old-note")
    entry = adapter._read_entry(current_file)
    entry.provenance.written_at = datetime.now(timezone.utc) - timedelta(days=10)
    current_file.write_text(entry.model_dump_json(indent=2), encoding="utf-8")

    assert mem.read("svc/auth", "old-note", min_confidence=0.4) is None
    assert mem.read_log == []
    assert adapter.read("svc/auth", "old-note").recall_count == 0


def test_recall_records_own_historical_version_not_other_agents_latest_version(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem_a = AgentMemory(agent_id="agent-a", adapter=adapter)
    mem_a.write("svc/api", "timeout", 30)

    mem_b = mem_a.as_agent("agent-b")
    mem_b.write("svc/api", "timeout", 60)

    recalled = mem_a.recall("svc/api", "timeout")
    assert recalled is not None
    assert recalled.version == 1
    assert recalled.value == 30

    causal = mem_a.explain()["causal_entries"]
    assert len(causal) == 1
    assert causal[0]["version"] == 1
    assert causal[0]["value"] == 30
    assert causal[0]["written_by"] == "agent-a"
    assert adapter.read("svc/api", "timeout").recall_count == 0


def test_recall_miss_does_not_record_other_agents_entry(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem_b = AgentMemory(agent_id="agent-b", adapter=adapter)
    mem_b.write("svc/api", "timeout", 60)

    mem_a = mem_b.as_agent("agent-a")
    assert mem_a.recall("svc/api", "timeout") is None
    assert mem_a.read_log == []
    assert mem_a.explain()["causal_entries"] == []


def test_on_conflict_receives_our_last_read_and_current_version(tmp_path) -> None:
    seen: dict[str, object] = {}

    def resolver(last_read, current, new_val):
        seen["last_read_version"] = last_read.version
        seen["last_read_value"] = last_read.value
        seen["current_version"] = current.version
        seen["current_value"] = current.value
        return new_val

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem_a = AgentMemory(agent_id="agent-a", adapter=adapter, on_conflict=resolver)
    mem_a.write("svc/db", "pool_size", 10)
    mem_a.read("svc/db", "pool_size")  # reads v1 (value=10)

    mem_b = mem_a.as_agent("agent-b")
    mem_b.write("svc/db", "pool_size", 20)  # writes v2 (value=20)

    written = mem_a.write("svc/db", "pool_size", 30)
    assert written.version == 3
    assert seen["last_read_version"] == 1
    assert seen["last_read_value"] == 10
    assert seen["current_version"] == 2
    assert seen["current_value"] == 20
