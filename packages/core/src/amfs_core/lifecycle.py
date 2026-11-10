"""TTL lifecycle manager — archives expired entries in a background thread."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from amfs_core.abc import AdapterABC
from amfs_core.exceptions import VersionConflictError
from amfs_core.models import MemoryEntry

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Periodically scans for entries past their TTL and archives them.

    "Archiving" means writing a new version with confidence set to 0.0,
    effectively marking the entry as expired while preserving CoW history.

    Parameters
    ----------
    adapter:
        The storage adapter to scan and write to.
    interval:
        Seconds between TTL sweep runs.
    """

    def __init__(self, adapter: AdapterABC, *, interval: float = 60.0) -> None:
        self._adapter = adapter
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background TTL sweep thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="amfs-lifecycle")
        self._thread.start()
        logger.info("Lifecycle manager started (interval=%.1fs)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Lifecycle manager stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def sweep(self) -> list[MemoryEntry]:
        """Run a single TTL sweep. Returns list of archived entries.

        Asks the adapter for the expired rows (``list_expired``) where it has
        that — Postgres answers from a partial index on ``ttl_at``, the
        filesystem and S3 adapters by scanning their current entries — and
        otherwise lists the store and filters here. ``list_expired`` is not
        optional for an adapter whose ``list()`` hides expired entries, as
        those three do: the listing would never show this sweep anything to
        archive. The fallback is for an adapter that still lists them, and
        over HTTP, where the server archives its own store and the listing
        comes back without them.

        The archived copy is written as the version after the one that
        expired, which is the version every adapter expects of a writer that
        names one. If the key was written again in the meantime — a fresh
        value, or another sweeper's archive — the adapter refuses with
        ``VersionConflictError`` and the entry is skipped: the newer version
        stands, and the next sweep looks at it on its own terms.
        """
        now = datetime.now(timezone.utc)
        list_expired = getattr(self._adapter, "list_expired", None)
        entries = list_expired(now=now) if callable(list_expired) else self._adapter.list()
        archived: list[MemoryEntry] = []

        for entry in entries:
            if not entry.is_expired(now=now):
                continue
            logger.debug(
                "Archiving expired entry: %s/%s v%d (ttl_at=%s)",
                entry.entity_path,
                entry.key,
                entry.version,
                entry.ttl_at,
            )
            archived_entry = entry.model_copy(
                update={"confidence": 0.0, "ttl_at": None, "version": entry.version + 1}
            )
            try:
                written = self._adapter.write(archived_entry)
            except VersionConflictError:
                logger.debug(
                    "Not archiving %s/%s v%d: a newer version was written",
                    entry.entity_path,
                    entry.key,
                    entry.version,
                )
                continue
            archived.append(written)

        return archived

    def _run(self) -> None:
        """Background loop: sweep, then sleep until next interval or stop."""
        while not self._stop_event.is_set():
            try:
                archived = self.sweep()
                if archived:
                    logger.info("TTL sweep archived %d entries", len(archived))
            except Exception as exc:
                msg = str(exc)
                if "401" in msg or "Unauthorized" in msg:
                    logger.error(
                        "TTL sweep failed with 401 Unauthorized — check that "
                        "AMFS_API_KEY is set correctly in your MCP server env."
                    )
                else:
                    logger.exception("Error during TTL sweep")
            self._stop_event.wait(timeout=self._interval)
