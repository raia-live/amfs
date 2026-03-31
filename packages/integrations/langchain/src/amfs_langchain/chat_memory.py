"""AMFSChatMemory — stores LangChain chat history in AMFS."""

from __future__ import annotations

from typing import Any

from amfs.memory import AgentMemory


class AMFSChatMemory:
    """A LangChain-compatible chat memory backed by AMFS.

    Stores chat messages as versioned AMFS entries so they benefit
    from CoW history and confidence tracking.

    Usage::

        from amfs import AgentMemory
        from amfs_langchain import AMFSChatMemory

        mem = AgentMemory(agent_id="chat-agent")
        chat_memory = AMFSChatMemory(mem, session_key="conv-123")
    """

    def __init__(
        self,
        memory: AgentMemory,
        session_key: str = "default",
        entity_path: str = "_langchain_chat",
    ) -> None:
        self._memory = memory
        self._entity_path = entity_path
        self._session_key = session_key

    @property
    def memory_variables(self) -> list[str]:
        return ["history"]

    def load_memory_variables(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load chat history."""
        entry = self._memory.read(self._entity_path, self._session_key)
        if entry is None:
            return {"history": []}
        return {"history": entry.value}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Append a turn to chat history."""
        current = self.load_memory_variables()
        history: list[dict[str, str]] = current.get("history", [])
        history.append({"input": str(inputs), "output": str(outputs)})
        self._memory.write(self._entity_path, self._session_key, history)

    def clear(self) -> None:
        """Clear chat history."""
        self._memory.write(self._entity_path, self._session_key, [])
