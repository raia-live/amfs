"""Filesystem watcher using watchdog — observes writes to entity paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from watchdog.events import (
    FileCreatedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers.polling import PollingObserver

from amfs_core.abc import WatchHandle
from amfs_core.models import MemoryEntry

logger = logging.getLogger(__name__)


class _EntryCreatedHandler(FileSystemEventHandler):
    """Fires callback when a new *_current.json file appears (create or rename)."""

    def __init__(self, callback: Callable[[MemoryEntry], None]) -> None:
        self._callback = callback

    def _handle(self, path: Path) -> None:
        if not path.name.endswith("_current.json"):
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = MemoryEntry.model_validate(data)
            self._callback(entry)
        except Exception:
            logger.debug("Watcher: could not parse %s", path, exc_info=True)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # os.rename(tmp, final) triggers a moved event; check the destination
        dest = getattr(event, "dest_path", None)
        if dest:
            self._handle(Path(dest))


class FilesystemWatcher:
    """Manages watchdog observers for AMFS entity paths."""

    def __init__(self, poll_interval: float = 0.25) -> None:
        self._observer: PollingObserver | None = None
        self._poll_interval = poll_interval

    def _ensure_observer(self) -> PollingObserver:
        if self._observer is None or not self._observer.is_alive():
            self._observer = PollingObserver(timeout=self._poll_interval)
            self._observer.daemon = True
            self._observer.start()
        return self._observer

    def watch(
        self,
        directory: Path,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        """Start watching *directory* recursively for new current entries."""
        directory.mkdir(parents=True, exist_ok=True)
        observer = self._ensure_observer()
        handler = _EntryCreatedHandler(callback)
        watch_obj = observer.schedule(handler, str(directory), recursive=True)

        def cancel() -> None:
            try:
                observer.unschedule(watch_obj)
            except Exception:
                pass

        return WatchHandle(cancel)

    def stop(self) -> None:
        if self._observer is not None and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=2)
