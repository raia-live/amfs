"""Tests that outcome back-propagation and TTL sweep preserve integrity_chain."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from amfs import AgentMemory, OutcomeType
from amfs_core.lifecycle import LifecycleManager
from amfs_filesystem.adapter import FilesystemAdapter


def test_verify_after_commit_outcome_has_no_chain_break(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="agent-1", adapter=adapter)
    mem.write("svc/checkout", "rule", {"step": "drain first"})

    mem.read("svc/checkout", "rule")
    mem.commit_outcome("deploy-1", OutcomeType.SUCCESS)

    mem.read("svc/checkout", "rule")
    mem.commit_outcome("deploy-2", OutcomeType.MINOR_FAILURE)

    report = mem.verify("svc/checkout")
    assert report["total_checked"] == 3
    assert report["valid"] == 3
    assert report["corrupted"] == []
    assert report["chain_breaks"] == []


def test_verify_after_ttl_sweep_has_no_chain_break(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="agent-1", adapter=adapter)
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    mem.write("svc/checkout", "temp", {"step": "temp"}, ttl_at=past)

    archived = LifecycleManager(mem._adapter).sweep()
    assert len(archived) == 1
    assert archived[0].version == 2

    report = mem.verify("svc/checkout")
    assert report["total_checked"] == 2
    assert report["valid"] == 2
    assert report["corrupted"] == []
    assert report["chain_breaks"] == []
