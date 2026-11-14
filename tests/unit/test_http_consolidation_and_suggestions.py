"""Tests for consolidation proposals diff_branch handling and cluster suggestion visibility scoping."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import amfs_http.server as server_mod
from amfs import AgentMemory
from amfs_core.models import Branch, BranchStatus, DiffEntry, Digest, DigestType
from amfs_filesystem.adapter import FilesystemAdapter
from fastapi.testclient import TestClient


def test_list_consolidation_proposals_reads_diff_branch_list(tmp_path, monkeypatch) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="server", adapter=adapter)

    now = datetime.now(timezone.utc)
    adapter.list_branches = MagicMock(
        return_value=[
            Branch(
                name="cortex/consolidation/svc/auth/20261114-120000",
                parent_branch="main",
                status=BranchStatus.ACTIVE,
                description="Convergent knowledge: 3 agents agreed",
                branched_at=now,
                created_by="cortex-consolidator",
            )
        ]
    )
    adapter.diff_branch = MagicMock(
        return_value=[
            DiffEntry(
                entity_path="svc/auth",
                key="timeout",
                diff_type="modified",
                branch_value={"timeout_ms": 5000},
                branch_confidence=0.92,
            )
        ]
    )

    monkeypatch.setattr(server_mod, "_memory", mem)
    monkeypatch.setattr(server_mod, "_get_memory", lambda: mem)
    monkeypatch.delenv("AMFS_API_KEYS", raising=False)
    monkeypatch.setattr(server_mod, "_visible_entity_paths", lambda _req: None)

    client = TestClient(server_mod.app)
    resp = client.get("/api/v1/cortex/consolidation/proposals")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    proposal = body["proposals"][0]
    assert proposal["entity_path"] == "svc/auth"
    assert proposal["source_entry_keys"] == ["svc/auth/timeout"]
    assert proposal["proposed_value"] == {"timeout_ms": 5000}
    assert proposal["proposed_confidence"] == 0.92


def test_agent_group_suggestions_filters_agents_field_when_scoped(tmp_path, monkeypatch) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="server", adapter=adapter)

    digest = Digest(
        digest_type=DigestType.AGENT_CLUSTERS,
        scope=f"account:{mem.namespace}",
        summary={
            "clusters": [
                {
                    "cluster_id": "cluster-auth",
                    "suggested_name": "Auth Team",
                    "agents": ["agent-a", "agent-b", "agent-c"],
                    "rationale": "Shared entity svc/auth",
                },
                {
                    "cluster_id": "cluster-solo",
                    "suggested_name": "Solo",
                    "agents": ["agent-a", "agent-d"],
                    "rationale": "Only one visible member",
                },
            ]
        },
        entry_count=3,
        compiled_at=datetime.now(timezone.utc),
    )
    adapter.get_digest = MagicMock(return_value=digest)
    adapter.list_dismissed_cluster_ids = MagicMock(return_value=[])

    monkeypatch.setattr(server_mod, "_memory", mem)
    monkeypatch.setattr(server_mod, "_get_memory", lambda: mem)
    monkeypatch.delenv("AMFS_API_KEYS", raising=False)
    monkeypatch.setattr(
        server_mod, "_visible_agent_ids", lambda _req: {"agent-a", "agent-b"}
    )

    client = TestClient(server_mod.app)
    resp = client.get("/api/v1/agent-groups/suggestions")
    assert resp.status_code == 200
    suggestions = resp.json()["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["cluster_id"] == "cluster-auth"
    assert suggestions[0]["agents"] == ["agent-a", "agent-b"]
