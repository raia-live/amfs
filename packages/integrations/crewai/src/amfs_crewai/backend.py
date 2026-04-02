"""AMFS storage backend for CrewAI Memory."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from amfs import AgentMemory


class AMFSStorageBackend:
    """AMFS as a CrewAI Memory storage backend.

    Maps CrewAI scopes to AMFS entity_paths, giving CrewAI
    the benefit of AMFS versioning, provenance, and outcomes.

    Usage::

        from amfs import AgentMemory
        from amfs_crewai import AMFSStorageBackend

        mem = AgentMemory(agent_id="crewai-crew")
        backend = AMFSStorageBackend(mem)
        # Pass to CrewAI: Memory(storage=backend)
    """

    def __init__(self, memory: AgentMemory, default_scope: str = "crewai") -> None:
        self._memory = memory
        self._default_scope = default_scope

    def save(self, data: dict[str, Any], scope: str | None = None) -> None:
        entity_path = scope or self._default_scope
        key = data.get("id", data.get("key", str(uuid4())[:8]))
        content = data.get("content", data.get("value", ""))
        metadata = {k: v for k, v in data.items() if k not in ("id", "key", "content", "value")}
        self._memory.write(
            entity_path,
            key,
            {"content": content, **metadata},
            confidence=data.get("importance", 0.8),
        )

    def query(
        self, query_text: str, *, limit: int = 10, scope: str | None = None
    ) -> list[dict[str, Any]]:
        entity_path = scope or self._default_scope
        entries = self._memory.search(entity_path=entity_path, limit=limit)
        results: list[dict[str, Any]] = []
        for e in entries:
            val = str(e.value) if e.value else ""
            if (
                not query_text
                or query_text.lower() in val.lower()
                or query_text.lower() in e.key.lower()
            ):
                results.append({
                    "id": e.key,
                    "content": e.value,
                    "confidence": e.confidence,
                    "version": e.version,
                    "agent_id": e.provenance.agent_id,
                    "written_at": (
                        e.provenance.written_at.isoformat()
                        if hasattr(e.provenance.written_at, "isoformat")
                        else str(e.provenance.written_at)
                    ),
                })
        return results[:limit]

    def delete(self, record_id: str, scope: str | None = None) -> None:
        entity_path = scope or self._default_scope
        existing = self._memory.read(entity_path, record_id)
        if existing:
            self._memory.write(entity_path, record_id, {"_deleted": True}, confidence=0.0)

    def reset(self, scope: str | None = None) -> None:
        entity_path = scope or self._default_scope
        entries = self._memory.list(entity_path)
        for e in entries:
            self._memory.write(e.entity_path, e.key, {"_deleted": True}, confidence=0.0)
