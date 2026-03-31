"""AMFSMemoryStore — AutoGen-compatible memory store backed by AMFS."""

from __future__ import annotations

from typing import Any

from amfs.memory import AgentMemory


class AMFSMemoryStore:
    """A memory store for AutoGen agents backed by AMFS.

    Usage::

        from amfs import AgentMemory
        from amfs_autogen import AMFSMemoryStore

        mem = AgentMemory(agent_id="autogen-agent")
        store = AMFSMemoryStore(mem)
        store.add("user_preferences", {"theme": "dark"})
        prefs = store.get("user_preferences")
    """

    def __init__(
        self, memory: AgentMemory, entity_path: str = "_autogen_memory"
    ) -> None:
        self._memory = memory
        self._entity_path = entity_path

    def add(self, key: str, value: Any) -> None:
        """Store a value by key."""
        self._memory.write(self._entity_path, key, value)

    def get(self, key: str) -> Any | None:
        """Retrieve the latest value for a key."""
        entry = self._memory.read(self._entity_path, key)
        return entry.value if entry else None

    def get_all(self) -> dict[str, Any]:
        """Retrieve all stored key-value pairs."""
        entries = self._memory.list(self._entity_path)
        return {e.key: e.value for e in entries}

    def clear(self, key: str) -> None:
        """Clear a key by writing a None value with 0 confidence."""
        self._memory.write(self._entity_path, key, None, confidence=0.0)
