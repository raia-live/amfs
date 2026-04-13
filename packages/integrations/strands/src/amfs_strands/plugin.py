"""AMFSPlugin — a Strands Agents Plugin that gives agents persistent memory.

Provides @tool methods for read/write/search/recall/list and @hook methods
for automatic causal tracing of tool calls.  Goes beyond tool-only
integrations (like mem0) by passively building decision traces.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from strands import tool
    from strands.hooks import AfterToolCallEvent
    from strands.plugins import Plugin, hook

    _HAS_STRANDS = True
except ImportError:
    _HAS_STRANDS = False

if _HAS_STRANDS:

    class AMFSPlugin(Plugin):
        """Strands Plugin that wraps :class:`AgentMemory` for persistent memory.

        Provides six tools (read, write, search, recall, list, record_context)
        and a hook that auto-records tool call context into the AMFS causal
        chain so decision traces capture every tool the agent uses.

        Usage::

            from strands import Agent
            from amfs import AgentMemory
            from amfs_strands import AMFSPlugin

            mem = AgentMemory(agent_id="research-agent")
            agent = Agent(plugins=[AMFSPlugin(mem)])
            agent("Research the latest AI papers and remember key findings")
        """

        name = "amfs-memory"

        def __init__(self, memory: Any, *, auto_trace: bool = True) -> None:
            super().__init__()
            self._memory = memory
            self._auto_trace = auto_trace

        @property
        def memory(self) -> Any:
            """The underlying :class:`AgentMemory` instance."""
            return self._memory

        # ------------------------------------------------------------------
        # Hooks
        # ------------------------------------------------------------------

        @hook
        def _trace_tool_call(self, event: AfterToolCallEvent) -> None:
            """Auto-record every tool call into the AMFS causal chain."""
            if not self._auto_trace:
                return
            tool_name = event.tool_use.get("name", "unknown")
            if tool_name.startswith("amfs_"):
                return
            tool_input = event.tool_use.get("input", {})
            try:
                summary = f"Tool '{tool_name}' called with {json.dumps(tool_input, default=str)[:200]}"
                self._memory.record_context(
                    f"strands-tool-{tool_name}",
                    summary,
                    source="strands-agent",
                )
            except Exception:
                logger.debug("Failed to record tool context for %s", tool_name, exc_info=True)

        # ------------------------------------------------------------------
        # Tools
        # ------------------------------------------------------------------

        @tool
        def amfs_read(self, entity_path: str, key: str) -> str:
            """Read a memory entry from AMFS persistent storage.

            Args:
                entity_path: Entity path (e.g. 'myapp/auth')
                key: Key name to read
            """
            entry = self._memory.read(entity_path, key)
            if entry is None:
                return f"No entry found at {entity_path}/{key}"
            return json.dumps(entry.model_dump(mode="json"), indent=2, default=str)

        @tool
        def amfs_write(
            self,
            entity_path: str,
            key: str,
            value: str,
            confidence: float = 1.0,
            memory_type: str = "fact",
        ) -> str:
            """Write a memory entry to AMFS persistent storage.

            Args:
                entity_path: Entity path (e.g. 'myapp/auth')
                key: Key name to write
                value: The value to store (string or JSON)
                confidence: Confidence score 0.0–1.0
                memory_type: One of 'fact', 'belief', or 'experience'
            """
            from amfs_core.models import MemoryType

            mt = MemoryType(memory_type)
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed = value

            entry = self._memory.write(
                entity_path,
                key,
                parsed,
                confidence=confidence,
                memory_type=mt,
            )
            return f"Written {entity_path}/{key} v{entry.version} (confidence={entry.confidence})"

        @tool
        def amfs_search(
            self,
            query: str = "",
            entity_path: str = "",
            min_confidence: float = 0.0,
            limit: int = 10,
        ) -> str:
            """Search across AMFS memory entries with filters.

            Args:
                query: Text query for full-text search
                entity_path: Optional entity path to scope the search
                min_confidence: Minimum confidence threshold
                limit: Maximum number of results
            """
            results = self._memory.search(
                query=query or None,
                entity_path=entity_path or None,
                min_confidence=min_confidence,
                limit=limit,
            )
            if not results:
                return "No entries found matching the search criteria."
            items = []
            for entry in results:
                items.append({
                    "entity_path": entry.entity_path,
                    "key": entry.key,
                    "value": entry.value,
                    "confidence": entry.confidence,
                    "version": entry.version,
                    "agent_id": entry.provenance.agent_id,
                })
            return json.dumps(items, indent=2, default=str)

        @tool
        def amfs_recall(self, entity_path: str, key: str) -> str:
            """Recall this agent's own memory for a key (agent-scoped read).

            Unlike amfs_read which returns the latest version by any agent,
            recall returns only entries written by this agent.

            Args:
                entity_path: Entity path (e.g. 'myapp/auth')
                key: Key name to recall
            """
            entry = self._memory.recall(entity_path, key)
            if entry is None:
                return f"No personal memory found at {entity_path}/{key}"
            return json.dumps(entry.model_dump(mode="json"), indent=2, default=str)

        @tool
        def amfs_list(self, entity_path: str = "") -> str:
            """List all memory entries, optionally filtered by entity path.

            Args:
                entity_path: Optional entity path filter
            """
            entries = self._memory.list(entity_path or None)
            if not entries:
                return "No entries found."
            items = [
                {
                    "entity_path": e.entity_path,
                    "key": e.key,
                    "version": e.version,
                    "confidence": e.confidence,
                }
                for e in entries
            ]
            return json.dumps(items, indent=2, default=str)

        @tool
        def amfs_record_context(self, label: str, summary: str, source: str = "") -> str:
            """Record external context in the AMFS causal chain.

            Use this after consulting an external tool, API, or data source so
            that decision traces capture what informed the agent's decisions.

            Args:
                label: Short label for the context (e.g. 'api-response')
                summary: Description of what was found
                source: Where the information came from
            """
            self._memory.record_context(label, summary, source=source or None)
            return f"Context recorded: {label}"

else:

    class AMFSPlugin:  # type: ignore[no-redef]
        """Stub when strands-agents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "strands-agents is not installed. "
                "Install with: pip install amfs-strands[strands]"
            )
