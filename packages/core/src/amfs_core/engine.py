"""CoWEngine, CausalTagger, and ReadTracker — core write logic for AMFS."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, Provenance


class CausalTagger:
    """Stamps provenance metadata on every write.

    Parameters
    ----------
    agent_id:
        Identifier of the agent performing writes.
    session_id:
        Optional session identifier. Auto-generated if not provided.
    """

    def __init__(self, agent_id: str, session_id: str | None = None) -> None:
        self.agent_id = agent_id
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"

    def tag(self, *, pattern_refs: list[str] | None = None) -> Provenance:
        """Create a new Provenance with the current timestamp."""
        return Provenance(
            agent_id=self.agent_id,
            session_id=self.session_id,
            written_at=datetime.now(timezone.utc),
            pattern_refs=pattern_refs or [],
        )


class ReadTracker:
    """Automatically records every read within a session for causal linking
    and conflict detection.

    When an agent reads entries and later commits an outcome, the tracker
    provides the causal chain without the developer manually specifying
    which entries were involved.

    Also tracks the version at read time so the engine can detect stale
    writes (another agent modified the entry since we last read it).
    """

    def __init__(self) -> None:
        self._reads: dict[str, datetime] = {}
        self._versions: dict[str, int] = {}

    def record(self, entry: MemoryEntry) -> None:
        """Record that an entry was read during this session."""
        self._reads[entry.entry_key] = datetime.now(timezone.utc)
        self._versions[entry.entry_key] = entry.version

    @property
    def causal_keys(self) -> list[str]:
        """All entry keys read in this session, ordered by read time."""
        return [k for k, _ in sorted(self._reads.items(), key=lambda x: x[1])]

    @property
    def read_count(self) -> int:
        return len(self._reads)

    def read_version(self, entry_key: str) -> int | None:
        """Return the version we last read for an entry, or None if never read."""
        return self._versions.get(entry_key)

    def clear(self) -> None:
        """Reset the read log (e.g. between sub-tasks within a session)."""
        self._reads.clear()
        self._versions.clear()

    def contains(self, entry_key: str) -> bool:
        return entry_key in self._reads


class CoWEngine:
    """Copy-on-Write engine that reads existing entries, increments versions,
    and delegates writes to the underlying adapter.

    Parameters
    ----------
    adapter:
        The storage adapter to use.
    tagger:
        A CausalTagger for stamping provenance.
    read_tracker:
        Optional ReadTracker for auto-causal linking. If provided, every
        successful read is logged for later use in commit_outcome().
    """

    def __init__(
        self,
        adapter: AdapterABC,
        tagger: CausalTagger,
        read_tracker: ReadTracker | None = None,
    ) -> None:
        self._adapter = adapter
        self._tagger = tagger
        self._read_tracker = read_tracker

    @property
    def adapter(self) -> AdapterABC:
        return self._adapter

    @property
    def tagger(self) -> CausalTagger:
        return self._tagger

    @property
    def read_tracker(self) -> ReadTracker | None:
        return self._read_tracker

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Read the current version of a key from the adapter.

        If a ReadTracker is attached, the read is automatically logged
        for causal linking.
        """
        entry = self._adapter.read(entity_path, key, min_confidence=min_confidence)
        if entry is not None and self._read_tracker is not None:
            self._read_tracker.record(entry)
        return entry

    def write(
        self,
        entity_path: str,
        key: str,
        value: Any,
        *,
        confidence: float = 1.0,
        ttl_at: datetime | None = None,
        pattern_refs: list[str] | None = None,
    ) -> MemoryEntry:
        """Write a new version of a key with CoW semantics.

        - Reads the current version (if any) to determine the next version number.
        - Stamps provenance via the CausalTagger.
        - Delegates the actual write to the adapter.
        """
        current = self._adapter.read(entity_path, key)
        next_version = (current.version + 1) if current else 1

        entry = MemoryEntry(
            entity_path=entity_path,
            key=key,
            version=next_version,
            value=value,
            provenance=self._tagger.tag(pattern_refs=pattern_refs),
            confidence=confidence,
            outcome_count=current.outcome_count if current else 0,
            ttl_at=ttl_at,
        )

        return self._adapter.write(entry)

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """List entries from the adapter."""
        return self._adapter.list(entity_path, include_superseded=include_superseded)
