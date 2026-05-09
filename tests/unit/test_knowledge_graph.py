"""Tests for Gap B: knowledge graph models, materializers, and graph_neighbors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from amfs_core.models import GraphEdge, GraphNeighborQuery, OutcomeType
from amfs.memory import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter


class TestGraphModels:
    def test_graph_edge_defaults(self):
        edge = GraphEdge(
            source_entity="svc/a",
            source_type="entry",
            relation="references",
            target_entity="svc/b",
            target_type="entry",
        )
        assert edge.confidence == 1.0
        assert edge.evidence_count == 1

    def test_graph_neighbor_query_defaults(self):
        q = GraphNeighborQuery(entity="svc/a")
        assert q.direction == "both"
        assert q.depth == 1
        assert q.limit == 50

    def test_graph_edge_serialization(self):
        edge = GraphEdge(
            source_entity="x",
            source_type="entry",
            relation="informed",
            target_entity="y",
            target_type="outcome",
            provenance={"agent_id": "test"},
        )
        data = edge.model_dump(mode="json")
        restored = GraphEdge(**data)
        assert restored.source_entity == "x"
        assert restored.provenance == {"agent_id": "test"}


def _make_memory(tmp_path: Path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs")
    return AgentMemory(agent_id="test-agent", adapter=adapter)


class TestMaterializerPatternRefs:
    def test_write_with_pattern_refs_calls_upsert(self, tmp_path: Path):
        import time
        mem = _make_memory(tmp_path)
        mock_upsert = MagicMock(return_value=GraphEdge(
            source_entity="", source_type="", relation="",
            target_entity="", target_type="",
        ))
        mem._adapter.upsert_graph_edge = mock_upsert

        mem.write("svc/auth", "pattern-retry", "retry logic", pattern_refs=["pattern-circuit-breaker"])

        time.sleep(0.2)
        assert mock_upsert.call_count >= 1
        edge_arg = mock_upsert.call_args_list[0][0][0]
        assert edge_arg.relation == "references"
        assert edge_arg.target_entity == "pattern-circuit-breaker"

    def test_write_without_pattern_refs_no_graph_call(self, tmp_path: Path):
        mem = _make_memory(tmp_path)
        mock_upsert = MagicMock()
        mem._adapter.upsert_graph_edge = mock_upsert

        mem.write("svc/auth", "task-login", "do login")

        mock_upsert.assert_not_called()


class TestMaterializerCausalEdges:
    def test_commit_outcome_materializes_edges(self, tmp_path: Path):
        mem = _make_memory(tmp_path)
        mock_upsert = MagicMock(return_value=GraphEdge(
            source_entity="", source_type="", relation="",
            target_entity="", target_type="",
        ))
        mem._adapter.upsert_graph_edge = mock_upsert

        mem.write("svc/auth", "task-login", "login impl")
        mem.read("svc/auth", "task-login")

        mem.commit_outcome("deploy-v1", OutcomeType.SUCCESS)

        time.sleep(0.3)

        informed_calls = [
            c for c in mock_upsert.call_args_list
            if c[0][0].relation == "informed"
        ]
        assert len(informed_calls) >= 1


class TestMaterializerCrossAgent:
    def test_read_from_materializes_learned_from(self, tmp_path: Path):
        adapter = FilesystemAdapter(root=tmp_path / ".amfs")
        writer = AgentMemory(agent_id="writer-agent", adapter=adapter)
        reader = AgentMemory(agent_id="reader-agent", adapter=adapter)

        mock_upsert = MagicMock(return_value=GraphEdge(
            source_entity="", source_type="", relation="",
            target_entity="", target_type="",
        ))
        reader._adapter.upsert_graph_edge = mock_upsert

        writer.write("svc/auth", "task-login", "login impl")
        reader.read_from("writer-agent", "svc/auth", "task-login")

        learned_calls = [
            c for c in mock_upsert.call_args_list
            if c[0][0].relation == "learned_from"
        ]
        assert len(learned_calls) == 1
        edge = learned_calls[0][0][0]
        assert edge.source_entity == "reader-agent"
        assert edge.target_entity == "writer-agent"


class TestGraphNeighborsSDK:
    def test_graph_neighbors_delegates_to_adapter(self, tmp_path: Path):
        mem = _make_memory(tmp_path)
        expected = [GraphEdge(
            source_entity="svc/a",
            source_type="entry",
            relation="references",
            target_entity="svc/b",
            target_type="entry",
        )]
        mem._adapter.graph_neighbors = MagicMock(return_value=expected)

        result = mem.graph_neighbors("svc/a", relation="references")

        assert result == expected
        mem._adapter.graph_neighbors.assert_called_once()


class TestFSAdapterGraphNoOp:
    def test_fs_graph_methods_return_empty(self, tmp_path: Path):
        adapter = FilesystemAdapter(root=tmp_path / ".amfs")

        edge = GraphEdge(
            source_entity="x", source_type="entry",
            relation="references",
            target_entity="y", target_type="entry",
        )
        result = adapter.upsert_graph_edge(edge)
        assert result.source_entity == "x"

        neighbors = adapter.graph_neighbors(GraphNeighborQuery(entity="x"))
        assert neighbors == []

        edges = adapter.list_graph_edges(entity="x")
        assert edges == []

        stats = adapter.graph_stats()
        assert stats == {}
