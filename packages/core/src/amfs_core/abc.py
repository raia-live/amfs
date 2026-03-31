"""Abstract base class for AMFS storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from amfs_core.models import MemoryEntry, OutcomeRecord


class WatchHandle:
    """Handle returned by adapter.watch() — call .cancel() to stop watching."""

    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            self._cancel_fn()
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AdapterABC(ABC):
    """Interface that every AMFS storage adapter must implement.

    Five operations define the contract:
      - read: fetch the current version of a key
      - write: persist a new version of a key (CoW)
      - list: enumerate entries under an entity path
      - watch: observe writes to an entity path in real time
      - commit_outcome: record an outcome and back-propagate to entries
    """

    @abstractmethod
    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Read the current version of a key, or None if not found.

        If the entry's confidence is below *min_confidence*, return None.
        """

    @abstractmethod
    def write(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a memory entry. Returns the written entry (with final version)."""

    @abstractmethod
    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """List current entries, optionally filtered to an entity path.

        If *include_superseded* is True, also include older versions.
        """

    @abstractmethod
    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        """Watch for writes to any key under *entity_path*.

        Returns a WatchHandle that can be cancelled.
        """

    @abstractmethod
    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        """Record an outcome and back-propagate confidence to causal entries.

        Returns the list of entries whose confidence was updated.
        """
