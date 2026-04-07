"""FilesystemAdapter — AMFS adapter backed by local filesystem with CoW semantics."""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.exceptions import AdapterError, EntryNotFoundError, VersionConflictError
from amfs_core.lock import AdvisoryLock
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
)

from amfs_filesystem.layout import PathLayout
from amfs_filesystem.watcher import FilesystemWatcher

logger = logging.getLogger(__name__)


class FilesystemAdapter(AdapterABC):
    """Store AMFS entries as versioned JSON files with CoW via atomic rename.

    Parameters
    ----------
    root:
        Root directory (typically ``<project>/.amfs``).
    namespace:
        Logical namespace (e.g. ``"default"``).
    """

    def __init__(self, root: Path, namespace: str = "default") -> None:
        self._layout = PathLayout(root, namespace)
        self._watcher = FilesystemWatcher()
        # Ensure namespace dir exists
        self._layout.namespace_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str = "main",
    ) -> MemoryEntry | None:
        current_file = self._layout.current_version_file(entity_path, key)
        if current_file is None:
            return None
        entry = self._read_entry(current_file)
        if entry.confidence < min_confidence:
            return None
        return entry

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        layout = self._layout
        key_dir = layout.key_dir(entry.entity_path, entry.key)
        key_dir.mkdir(parents=True, exist_ok=True)

        with AdvisoryLock(layout.lock_path(entry.entity_path, entry.key)):
            current_version = layout.current_version_number(entry.entity_path, entry.key)
            new_version = current_version + 1

            # If caller specified a version, verify no conflict
            if entry.version > 1 and entry.version != new_version:
                raise VersionConflictError(
                    entry.entity_path, entry.key, entry.version, current_version
                )

            # Update entry with actual version
            entry = entry.model_copy(update={"version": new_version})

            # Supersede old current file
            old_current = layout.current_version_file(entry.entity_path, entry.key)
            if old_current is not None:
                superseded_path = old_current.with_name(
                    old_current.name.replace("_current.json", "_superseded.json")
                )
                old_current.rename(superseded_path)

            # Atomic write: write to tmp, then rename
            tmp_path = layout.tmp_file(entry.entity_path, entry.key, new_version)
            final_path = layout.version_file(entry.entity_path, entry.key, new_version, current=True)

            data = entry.model_dump(mode="json")
            tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            os.rename(tmp_path, final_path)

        return entry

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
        branch: str = "main",
    ) -> list[MemoryEntry]:
        layout = self._layout
        entity_paths = (
            [entity_path] if entity_path else layout.all_entity_paths()
        )
        entries: list[MemoryEntry] = []
        for ep in entity_paths:
            for key in layout.all_keys(ep):
                files = layout.all_version_files(
                    ep, key, include_superseded=include_superseded
                )
                for f in files:
                    try:
                        entries.append(self._read_entry(f))
                    except Exception:
                        logger.warning("Skipping unreadable file: %s", f)
        return entries

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        watch_dir = self._layout.entity_dir(entity_path)
        return self._watcher.watch(watch_dir, callback)

    # ------------------------------------------------------------------
    # commit_outcome
    # ------------------------------------------------------------------

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        multiplier = OUTCOME_MULTIPLIERS[record.outcome_type]
        updated: list[MemoryEntry] = []

        for entry_key_spec in record.causal_entry_keys:
            # entry_key_spec format: "entity_path/key"
            parts = entry_key_spec.rsplit("/", 1)
            if len(parts) != 2:
                logger.warning("Invalid causal_entry_key format: %s", entry_key_spec)
                continue
            entity_path, key = parts

            current = self.read(entity_path, key)
            if current is None:
                logger.warning("Causal entry not found: %s", entry_key_spec)
                continue

            new_confidence = current.confidence * multiplier * record.causal_confidence
            new_entry = current.model_copy(
                update={
                    "version": 1,
                    "confidence": new_confidence,
                    "outcome_count": current.outcome_count + 1,
                }
            )
            written = self.write(new_entry)
            updated.append(written)

        return updated

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def increment_recall_count(
        self,
        entity_path: str,
        key: str,
        *,
        branch: str = "main",
    ) -> None:
        current_file = self._layout.current_version_file(entity_path, key)
        if current_file is None:
            return
        try:
            data = json.loads(current_file.read_text(encoding="utf-8"))
            data["recall_count"] = data.get("recall_count", 0) + 1
            current_file.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            logger.warning("Failed to increment recall_count for %s/%s", entity_path, key)

    @staticmethod
    def _read_entry(path: Path) -> MemoryEntry:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MemoryEntry.model_validate(data)
        except Exception as exc:
            raise AdapterError(f"Failed to read entry from {path}") from exc

    def close(self) -> None:
        """Stop the filesystem watcher."""
        self._watcher.stop()
