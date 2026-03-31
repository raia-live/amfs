"""AMFSCheckpointer — stores LangGraph checkpoint state in AMFS."""

from __future__ import annotations

import json
from typing import Any

from amfs.memory import AgentMemory


class AMFSCheckpointer:
    """A LangGraph-compatible checkpointer backed by AMFS.

    Stores graph state as AMFS memory entries keyed by thread_id.

    Usage::

        from amfs import AgentMemory
        from amfs_langgraph import AMFSCheckpointer

        mem = AgentMemory(agent_id="graph-agent")
        checkpointer = AMFSCheckpointer(mem)
        # Pass to your LangGraph graph builder
    """

    def __init__(self, memory: AgentMemory, entity_path: str = "_langgraph_checkpoints") -> None:
        self._memory = memory
        self._entity_path = entity_path

    def put(self, config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Save a checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        self._memory.write(self._entity_path, thread_id, checkpoint)
        return config

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Load the latest checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        entry = self._memory.read(self._entity_path, thread_id)
        if entry is None:
            return None
        return entry.value  # type: ignore[return-value]

    def list(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """List all checkpoints for a thread (including superseded)."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        entries = self._memory.list(self._entity_path, include_superseded=True)
        return [
            e.value
            for e in entries
            if e.key == thread_id
        ]
