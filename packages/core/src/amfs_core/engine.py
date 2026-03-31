"""CoWEngine and CausalTagger — core write logic for AMFS."""

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


class CoWEngine:
    """Copy-on-Write engine that reads existing entries, increments versions,
    and delegates writes to the underlying adapter.

    Parameters
    ----------
    adapter:
        The storage adapter to use.
    tagger:
        A CausalTagger for stamping provenance.
    """

    def __init__(self, adapter: AdapterABC, tagger: CausalTagger) -> None:
        self._adapter = adapter
        self._tagger = tagger

    @property
    def adapter(self) -> AdapterABC:
        return self._adapter

    @property
    def tagger(self) -> CausalTagger:
        return self._tagger

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Read the current version of a key from the adapter."""
        return self._adapter.read(entity_path, key, min_confidence=min_confidence)

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
