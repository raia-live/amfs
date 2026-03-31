"""Snapshot export and import for AMFS memory state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry


class SnapshotExporter:
    """Exports the current state of an adapter to a JSON snapshot file."""

    def __init__(self, adapter: AdapterABC) -> None:
        self._adapter = adapter

    def export(
        self,
        output: Path,
        *,
        entity_path: str | None = None,
        include_superseded: bool = False,
    ) -> int:
        """Export entries to a JSON file. Returns the number of entries exported."""
        entries = self._adapter.list(
            entity_path, include_superseded=include_superseded
        )
        snapshot: dict[str, Any] = {
            "amfs_snapshot_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": [e.model_dump(mode="json") for e in entries],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
        )
        return len(entries)


class SnapshotImporter:
    """Imports a JSON snapshot into an adapter."""

    def __init__(self, adapter: AdapterABC) -> None:
        self._adapter = adapter

    def restore(self, input_path: Path) -> int:
        """Restore entries from a snapshot file. Returns the number of entries restored."""
        data = json.loads(input_path.read_text(encoding="utf-8"))
        entries_raw = data.get("entries", [])
        count = 0
        for entry_data in entries_raw:
            entry = MemoryEntry.model_validate(entry_data)
            # Reset version to 1 so the adapter assigns the correct next version
            entry = entry.model_copy(update={"version": 1})
            self._adapter.write(entry)
            count += 1
        return count
