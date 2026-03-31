"""AMFSTool — a CrewAI BaseTool that wraps AgentMemory for read/write/list."""

from __future__ import annotations

import json
from typing import Any, Type

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _ReadInput(BaseModel):
        entity_path: str = Field(description="Entity path to read from")
        key: str = Field(description="Key name to read")

    class _WriteInput(BaseModel):
        entity_path: str = Field(description="Entity path to write to")
        key: str = Field(description="Key name to write")
        value: str = Field(description="JSON value to write")

    class _ListInput(BaseModel):
        entity_path: str | None = Field(default=None, description="Optional entity path filter")

    class AMFSReadTool(BaseTool):
        name: str = "amfs_read"
        description: str = "Read a memory entry from the AMFS shared memory store"
        args_schema: Type[BaseModel] = _ReadInput

        def __init__(self, memory: Any, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._memory = memory

        def _run(self, entity_path: str, key: str) -> str:
            entry = self._memory.read(entity_path, key)
            if entry is None:
                return f"No entry found at {entity_path}/{key}"
            return json.dumps(entry.model_dump(mode="json"), indent=2, default=str)

    class AMFSWriteTool(BaseTool):
        name: str = "amfs_write"
        description: str = "Write a memory entry to the AMFS shared memory store"
        args_schema: Type[BaseModel] = _WriteInput

        def __init__(self, memory: Any, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._memory = memory

        def _run(self, entity_path: str, key: str, value: str) -> str:
            parsed = json.loads(value)
            entry = self._memory.write(entity_path, key, parsed)
            return f"Written {entity_path}/{key} v{entry.version}"

    class AMFSListTool(BaseTool):
        name: str = "amfs_list"
        description: str = "List memory entries in the AMFS shared memory store"
        args_schema: Type[BaseModel] = _ListInput

        def __init__(self, memory: Any, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._memory = memory

        def _run(self, entity_path: str | None = None) -> str:
            entries = self._memory.list(entity_path)
            result = [
                {"entity": e.entity_path, "key": e.key, "version": e.version, "confidence": e.confidence}
                for e in entries
            ]
            return json.dumps(result, indent=2)

except ImportError:
    pass


class AMFSTool:
    """Factory that creates CrewAI tools from an AgentMemory instance.

    Usage::

        from amfs import AgentMemory
        from amfs_crewai import AMFSTool

        mem = AgentMemory(agent_id="review-agent")
        tools = AMFSTool(mem).tools()
        # Pass `tools` to your CrewAI agent
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def tools(self) -> list[Any]:
        """Return a list of CrewAI BaseTool instances."""
        try:
            return [
                AMFSReadTool(self._memory),
                AMFSWriteTool(self._memory),
                AMFSListTool(self._memory),
            ]
        except NameError:
            raise ImportError(
                "CrewAI is not installed. Install with: pip install amfs-crewai[crewai]"
            )
