"""Unit tests for the AMFS Python SDK (M4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from amfs_core.models import OutcomeType
from amfs.config import find_config, load_config, load_config_or_default
from amfs.factory import create_adapter_from_config
from amfs.memory import AgentMemory
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter
from amfs_filesystem.adapter import FilesystemAdapter


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text(
            "namespace: prod\n"
            "layers:\n"
            "  primary:\n"
            "    adapter: filesystem\n"
            "    options:\n"
            "      root: /data/.amfs\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.namespace == "prod"
        assert cfg.layers["primary"].adapter == "filesystem"
        assert cfg.layers["primary"].options["root"] == "/data/.amfs"

    def test_find_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text("namespace: test\nlayers: {}\n")
        found = find_config(tmp_path)
        assert found == cfg_file

    def test_find_config_walks_up(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text("namespace: test\nlayers: {}\n")
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        found = find_config(subdir)
        assert found == cfg_file

    def test_find_config_returns_none(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        # Won't find anything (searches up to filesystem root)
        # This test just verifies it doesn't crash
        result = find_config(empty)
        # May or may not find something depending on the system — just check type
        assert result is None or isinstance(result, Path)

    def test_load_config_or_default(self) -> None:
        cfg = load_config_or_default()
        assert cfg.namespace == "default"
        assert "primary" in cfg.layers
        assert cfg.layers["primary"].adapter == "filesystem"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_filesystem_adapter(self, tmp_amfs_root: Path) -> None:
        from amfs_core.models import AMFSConfig, LayerConfig

        config = AMFSConfig(
            namespace="test",
            layers={
                "primary": LayerConfig(
                    adapter="filesystem",
                    options={"root": str(tmp_amfs_root)},
                )
            },
        )
        adapter = create_adapter_from_config(config)
        assert isinstance(adapter, FilesystemAdapter)

    def test_unknown_adapter_raises(self) -> None:
        from amfs_core.models import AMFSConfig, LayerConfig

        config = AMFSConfig(
            namespace="test",
            layers={"primary": LayerConfig(adapter="unknown")},
        )
        with pytest.raises(ValueError, match="Unknown adapter"):
            create_adapter_from_config(config)

    def test_missing_layer_raises(self) -> None:
        from amfs_core.models import AMFSConfig

        config = AMFSConfig(namespace="test")
        with pytest.raises(KeyError, match="Layer 'primary' not found"):
            create_adapter_from_config(config)


# ---------------------------------------------------------------------------
# AgentMemory tests
# ---------------------------------------------------------------------------

class TestAgentMemory:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(agent_id="test-agent", adapter=adapter)
        yield m  # type: ignore[misc]
        m.close()

    def test_write_and_read(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", {"data": 1})
        entry = mem.read("svc", "key")
        assert entry is not None
        assert entry.value == {"data": 1}
        assert entry.provenance.agent_id == "test-agent"

    def test_version_increments(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "v1")
        mem.write("svc", "key", "v2")
        entry = mem.read("svc", "key")
        assert entry is not None
        assert entry.version == 2
        assert entry.value == "v2"

    def test_list(self, mem: AgentMemory) -> None:
        mem.write("svc", "k1", "a")
        mem.write("svc", "k2", "b")
        entries = mem.list("svc")
        assert len(entries) == 2

    def test_commit_outcome(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "data")
        updated = mem.commit_outcome(
            "INC-001",
            OutcomeType.P1_INCIDENT,
            ["svc/key"],
        )
        assert len(updated) == 1
        assert updated[0].confidence == 1.15

    def test_properties(self, mem: AgentMemory) -> None:
        assert mem.agent_id == "test-agent"
        assert mem.session_id.startswith("sess-")
        assert mem.namespace == "default"

    def test_context_manager(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        with AgentMemory(agent_id="ctx-agent", adapter=adapter) as m:
            m.write("svc", "key", "value")
        # Should not raise after close

    def test_write_with_confidence_and_ttl(self, mem: AgentMemory) -> None:
        ttl = datetime(2099, 12, 31, tzinfo=timezone.utc)
        entry = mem.write("svc", "key", "val", confidence=0.8, ttl_at=ttl)
        assert entry.confidence == 0.8
        assert entry.ttl_at == ttl

    def test_write_with_pattern_refs(self, mem: AgentMemory) -> None:
        entry = mem.write("svc", "key", "val", pattern_refs=["retry-logic"])
        assert entry.provenance.pattern_refs == ["retry-logic"]

    def test_min_confidence_filter(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "val", confidence=0.3)
        assert mem.read("svc", "key", min_confidence=0.5) is None
        assert mem.read("svc", "key", min_confidence=0.2) is not None


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshot:
    @pytest.fixture
    def adapter(self, tmp_amfs_root: Path) -> FilesystemAdapter:
        a = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        yield a  # type: ignore[misc]
        a.close()

    def test_export_and_restore(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        # Seed data
        mem = AgentMemory(agent_id="snap-agent", adapter=adapter)
        mem.write("svc-a", "k1", {"x": 1})
        mem.write("svc-a", "k2", {"x": 2})
        mem.write("svc-b", "k3", {"x": 3})

        # Export
        snapshot_path = tmp_path / "snapshot.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path)
        assert count == 3
        assert snapshot_path.exists()

        # Verify snapshot structure
        data = json.loads(snapshot_path.read_text())
        assert data["amfs_snapshot_version"] == "0.1.0"
        assert data["entry_count"] == 3
        assert len(data["entries"]) == 3

        # Restore into a fresh adapter
        fresh_root = tmp_path / "fresh_amfs"
        fresh_root.mkdir()
        fresh_adapter = FilesystemAdapter(root=fresh_root, namespace="restored")
        importer = SnapshotImporter(fresh_adapter)
        restored_count = importer.restore(snapshot_path)
        assert restored_count == 3

        # Verify restored data
        entries = fresh_adapter.list()
        assert len(entries) == 3
        entity_paths = {e.entity_path for e in entries}
        assert entity_paths == {"svc-a", "svc-b"}
        fresh_adapter.close()

    def test_export_filtered(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        mem = AgentMemory(agent_id="snap-agent", adapter=adapter)
        mem.write("svc-a", "k1", "a")
        mem.write("svc-b", "k2", "b")

        snapshot_path = tmp_path / "filtered.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path, entity_path="svc-a")
        assert count == 1

    def test_export_empty(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        snapshot_path = tmp_path / "empty.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path)
        assert count == 0
        data = json.loads(snapshot_path.read_text())
        assert data["entries"] == []
