"""TTL lifecycle manager — archives expired entries in a background thread."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from amfs_core.abc import AdapterABC
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

        Asks the adapter for the expired rows (``list_expired``) where it can
        answer that with a query — the Postgres adapters do, from a partial
        index on ``ttl_at`` — and otherwise lists the store and filters here.
        The listing is what the filesystem adapter needs and what every
        adapter did until 2026-09: over HTTP against a hosted tenant it was
        the whole tenant every ``interval`` seconds per client process, which
        on a 25k-entry store took 3-27 s, mostly timed out, and so never
        archived anything either.
        """
        now = datetime.now(timezone.utc)
        list_expired = getattr(self._adapter, "list_expired", None)
        entries = list_expired(now=now) if callable(list_expired) else self._adapter.list()
        archived: list[MemoryEntry] = []

        for entry in entries:
            if entry.ttl_at is not None and entry.ttl_at <= now:
                logger.debug(
                    "Archiving expired entry: %s/%s v%d (ttl_at=%s)",
                    entry.entity_path,
                    entry.key,
                    entry.version,
                    entry.ttl_at,
                )
                archived_entry = entry.model_copy(
                    update={"confidence": 0.0, "ttl_at": None}
                )
                written = self._adapter.write(archived_entry)
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
