"""Standalone AMFS tool for Strands Agents — drop-in mem0 replacement.

Provides a single ``amfs_memory`` tool with action-based dispatch
(store, retrieve, search, list) that mirrors mem0's ``mem0_memory``
interface.  Uses a module-level AgentMemory singleton.

Usage::

    from strands import Agent
    from amfs_strands import amfs_memory

    agent = Agent(tools=[amfs_memory])
    agent("Remember that I prefer Python over TypeScript")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_memory_instance: Any = None


def _get_memory() -> Any:
    """Lazy-init a module-level AgentMemory singleton."""
    global _memory_instance
    if _memory_instance is None:
        from amfs import AgentMemory

        agent_id = os.environ.get("AMFS_AGENT_ID", "strands-agent")
        _memory_instance = AgentMemory(agent_id=agent_id)
    return _memory_instance


try:
    from strands import tool as _strands_tool

    @_strands_tool
    def amfs_memory(
        action: str,
        entity_path: str = "default",
        key: str = "",
        value: str = "",
        query: str = "",
        confidence: float = 1.0,
        limit: int = 10,
    ) -> str:
        """Store, retrieve, search, and list persistent agent memory with AMFS.

        Args:
            action: One of 'store', 'retrieve', 'search', or 'list'
            entity_path: Entity path scope (e.g. 'myapp/auth')
            key: Key name (required for store and retrieve)
            value: Value to store (required for store action)
            query: Search query text (required for search action)
            confidence: Confidence score 0.0-1.0 (for store action)
            limit: Maximum results (for search and list actions)
        """
        mem = _get_memory()

        if action == "store":
            if not key:
                return "Error: 'key' is required for store action"
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed = value
            entry = mem.write(entity_path, key, parsed, confidence=confidence)
            return f"Stored {entity_path}/{key} v{entry.version}"

        elif action == "retrieve":
            if not key:
                return "Error: 'key' is required for retrieve action"
            entry = mem.read(entity_path, key)
            if entry is None:
                return f"No entry found at {entity_path}/{key}"
            return json.dumps(entry.model_dump(mode="json"), indent=2, default=str)

        elif action == "search":
            results = mem.search(
                query=query or None,
                entity_path=entity_path if entity_path != "default" else None,
                limit=limit,
            )
            if not results:
                return "No entries found."
            items = [
                {
                    "entity_path": e.entity_path,
                    "key": e.key,
                    "value": e.value,
                    "confidence": e.confidence,
                }
                for e in results
            ]
            return json.dumps(items, indent=2, default=str)

        elif action == "list":
            entries = mem.list(entity_path if entity_path != "default" else None)
            if not entries:
                return "No entries found."
            items = [
                {
                    "entity_path": e.entity_path,
                    "key": e.key,
                    "version": e.version,
                    "confidence": e.confidence,
                }
                for e in entries[:limit]
            ]
            return json.dumps(items, indent=2, default=str)

        else:
            return f"Unknown action '{action}'. Use 'store', 'retrieve', 'search', or 'list'."

except ImportError:

    def amfs_memory(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub when strands-agents is not installed."""
        raise ImportError(
            "strands-agents is not installed. "
            "Install with: pip install amfs-strands[strands]"
        )
