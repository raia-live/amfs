"""Integration tests for CrewAI, LangChain, LangGraph, and AutoGen AMFS wrappers.

Tests each framework wrapper against a real FilesystemAdapter-backed AgentMemory.
No LLM or API key required — validates the memory plumbing end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amfs import AgentMemory, OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter

try:
    from crewai.tools import BaseTool as _  # noqa: F401

    _HAS_CREWAI = True
except ImportError:
    _HAS_CREWAI = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter(tmp_amfs_root: Path) -> FilesystemAdapter:
    a = FilesystemAdapter(root=tmp_amfs_root, namespace="framework-test")
    yield a  # type: ignore[misc]
    a.close()


@pytest.fixture
def mem(adapter: FilesystemAdapter) -> AgentMemory:
    m = AgentMemory(agent_id="test-agent", adapter=adapter)
    yield m  # type: ignore[misc]
    m.close()


# ---------------------------------------------------------------------------
# CrewAI — AMFSTool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CREWAI, reason="crewai not installed")
class TestCrewAITool:
    """Test AMFSTool read/write/list tools (requires crewai)."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem: AgentMemory) -> None:
        from amfs_crewai.tool import AMFSReadTool, AMFSWriteTool, AMFSListTool

        self.read_tool = AMFSReadTool(mem)
        self.write_tool = AMFSWriteTool(mem)
        self.list_tool = AMFSListTool(mem)

    def test_tools_factory(self, mem: AgentMemory) -> None:
        from amfs_crewai import AMFSTool

        tools = AMFSTool(mem).tools()
        assert len(tools) == 3

    def test_write_and_read(self) -> None:
        result = self.write_tool._run("svc", "config", '{"retries": 3}')
        assert "v1" in result

        read_result = self.read_tool._run("svc", "config")
        data = json.loads(read_result)
        assert data["value"] == {"retries": 3}

    def test_read_not_found(self) -> None:
        result = self.read_tool._run("nope", "nada")
        assert "No entry found" in result

    def test_list(self) -> None:
        self.write_tool._run("svc", "k1", '"val1"')
        self.write_tool._run("svc", "k2", '"val2"')
        result = json.loads(self.list_tool._run("svc"))
        assert len(result) == 2
        keys = {e["key"] for e in result}
        assert keys == {"k1", "k2"}


# ---------------------------------------------------------------------------
# CrewAI — AMFSStorageBackend
# ---------------------------------------------------------------------------


class TestCrewAIStorageBackend:
    """Test AMFSStorageBackend save/query/delete/reset."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem: AgentMemory) -> None:
        from amfs_crewai import AMFSStorageBackend

        self.backend = AMFSStorageBackend(mem, default_scope="crewai-test")

    def test_save_and_query(self) -> None:
        self.backend.save({"id": "item-1", "content": "test data", "importance": 0.9})
        results = self.backend.query("test")
        assert len(results) == 1
        assert results[0]["id"] == "item-1"

    def test_query_empty(self) -> None:
        results = self.backend.query("nothing")
        assert results == []

    def test_delete(self) -> None:
        self.backend.save({"id": "to-delete", "content": "doomed"})
        self.backend.delete("to-delete")
        results = self.backend.query("doomed")
        assert results == []

    def test_reset(self) -> None:
        self.backend.save({"id": "a", "content": "one"})
        self.backend.save({"id": "b", "content": "two"})
        self.backend.reset()
        results = self.backend.query("one")
        assert results == []
        results = self.backend.query("two")
        assert results == []


# ---------------------------------------------------------------------------
# LangChain — AMFSChatMemory
# ---------------------------------------------------------------------------


class TestLangChainChatMemory:
    """Test AMFSChatMemory save/load/clear."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem: AgentMemory) -> None:
        from amfs_langchain import AMFSChatMemory

        self.chat = AMFSChatMemory(mem, session_key="test-conv")

    def test_empty_history(self) -> None:
        result = self.chat.load_memory_variables()
        assert result == {"history": []}

    def test_save_and_load(self) -> None:
        self.chat.save_context(
            {"input": "What is AMFS?"},
            {"output": "Agent Memory File System."},
        )
        result = self.chat.load_memory_variables()
        assert len(result["history"]) == 1
        assert "AMFS" in result["history"][0]["input"]

    def test_multi_turn(self) -> None:
        self.chat.save_context({"input": "Hi"}, {"output": "Hello!"})
        self.chat.save_context({"input": "How?"}, {"output": "Great."})
        result = self.chat.load_memory_variables()
        assert len(result["history"]) == 2

    def test_clear(self) -> None:
        self.chat.save_context({"input": "Hi"}, {"output": "Hello!"})
        self.chat.clear()
        result = self.chat.load_memory_variables()
        assert result["history"] == []

    def test_session_isolation(self, mem: AgentMemory) -> None:
        from amfs_langchain import AMFSChatMemory

        chat_a = AMFSChatMemory(mem, session_key="sess-a")
        chat_b = AMFSChatMemory(mem, session_key="sess-b")

        chat_a.save_context({"input": "A"}, {"output": "A-reply"})
        chat_b.save_context({"input": "B"}, {"output": "B-reply"})

        assert len(chat_a.load_memory_variables()["history"]) == 1
        assert len(chat_b.load_memory_variables()["history"]) == 1
        assert "A" in chat_a.load_memory_variables()["history"][0]["input"]
        assert "B" in chat_b.load_memory_variables()["history"][0]["input"]

    def test_memory_variables_property(self) -> None:
        assert self.chat.memory_variables == ["history"]


# ---------------------------------------------------------------------------
# LangGraph — AMFSCheckpointer
# ---------------------------------------------------------------------------


class TestLangGraphCheckpointer:
    """Test AMFSCheckpointer put/get/list."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem: AgentMemory) -> None:
        from amfs_langgraph import AMFSCheckpointer

        self.cp = AMFSCheckpointer(mem)

    def _config(self, thread_id: str = "thread-1") -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def test_get_empty(self) -> None:
        assert self.cp.get(self._config()) is None

    def test_put_and_get(self) -> None:
        checkpoint = {"step": 1, "state": {"counter": 10}}
        self.cp.put(self._config(), checkpoint)
        result = self.cp.get(self._config())
        assert result == checkpoint

    def test_overwrite(self) -> None:
        self.cp.put(self._config(), {"step": 1})
        self.cp.put(self._config(), {"step": 2})
        result = self.cp.get(self._config())
        assert result == {"step": 2}

    def test_list_versions(self) -> None:
        self.cp.put(self._config(), {"step": 1})
        self.cp.put(self._config(), {"step": 2})
        versions = self.cp.list(self._config())
        assert len(versions) == 2

    def test_thread_isolation(self) -> None:
        self.cp.put(self._config("t1"), {"data": "alpha"})
        self.cp.put(self._config("t2"), {"data": "beta"})
        assert self.cp.get(self._config("t1")) == {"data": "alpha"}
        assert self.cp.get(self._config("t2")) == {"data": "beta"}


# ---------------------------------------------------------------------------
# AutoGen — AMFSMemoryStore
# ---------------------------------------------------------------------------


class TestAutoGenMemoryStore:
    """Test AMFSMemoryStore add/get/get_all/clear."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem: AgentMemory) -> None:
        from amfs_autogen import AMFSMemoryStore

        self.store = AMFSMemoryStore(mem)

    def test_get_missing(self) -> None:
        assert self.store.get("nonexistent") is None

    def test_add_and_get(self) -> None:
        self.store.add("prefs", {"theme": "dark", "lang": "en"})
        result = self.store.get("prefs")
        assert result == {"theme": "dark", "lang": "en"}

    def test_overwrite(self) -> None:
        self.store.add("prefs", {"v": 1})
        self.store.add("prefs", {"v": 2})
        assert self.store.get("prefs") == {"v": 2}

    def test_get_all(self) -> None:
        self.store.add("a", "alpha")
        self.store.add("b", "beta")
        all_data = self.store.get_all()
        assert all_data["a"] == "alpha"
        assert all_data["b"] == "beta"

    def test_clear(self) -> None:
        self.store.add("temp", "data")
        self.store.clear("temp")
        assert self.store.get("temp") is None


# ---------------------------------------------------------------------------
# Cross-framework memory sharing
# ---------------------------------------------------------------------------


class TestCrossFrameworkSharing:
    """Verify that different framework wrappers can share the same AMFS store."""

    def test_crewai_writes_others_read(self, adapter: FilesystemAdapter) -> None:
        """One agent writes via SDK (simulating CrewAI), others read via their wrappers."""
        from amfs_autogen import AMFSMemoryStore

        writer_mem = AgentMemory(agent_id="crewai-writer", adapter=adapter)
        lc_mem = AgentMemory(agent_id="langchain-reader", adapter=adapter)
        ag_mem = AgentMemory(agent_id="autogen-reader", adapter=adapter)

        try:
            writer_mem.write(
                "shared-project", "risk",
                {"score": 0.85, "flags": ["retry-bug"]},
                confidence=0.9,
            )

            lc_entry = lc_mem.read("shared-project", "risk")
            assert lc_entry is not None
            assert lc_entry.value["score"] == 0.85
            assert lc_entry.provenance.agent_id == "crewai-writer"

            store = AMFSMemoryStore(ag_mem, entity_path="shared-project")
            ag_data = store.get("risk")
            assert ag_data["score"] == 0.85
            assert ag_data["flags"] == ["retry-bug"]
        finally:
            writer_mem.close()
            lc_mem.close()
            ag_mem.close()

    def test_outcome_propagates_across_agents(self, adapter: FilesystemAdapter) -> None:
        """Outcome committed by one agent affects entries written by another."""
        writer = AgentMemory(agent_id="writer-agent", adapter=adapter)
        reader = AgentMemory(agent_id="reader-agent", adapter=adapter)

        try:
            writer.write("svc", "config", {"timeout": 5000}, confidence=1.0)

            entry = reader.read("svc", "config")
            assert entry is not None

            updated = reader.commit_outcome("INC-999", OutcomeType.CRITICAL_FAILURE)
            assert len(updated) == 1
            assert updated[0].confidence > 1.0

            boosted = writer.read("svc", "config")
            assert boosted.confidence > 1.0
        finally:
            writer.close()
            reader.close()

    def test_all_frameworks_write_to_shared_store(self, adapter: FilesystemAdapter) -> None:
        """Each framework writes via its wrapper; all entries are visible globally."""
        from amfs_langchain import AMFSChatMemory
        from amfs_autogen import AMFSMemoryStore

        crew_mem = AgentMemory(agent_id="crew-agent", adapter=adapter)
        lc_mem = AgentMemory(agent_id="lc-agent", adapter=adapter)
        ag_mem = AgentMemory(agent_id="ag-agent", adapter=adapter)
        observer = AgentMemory(agent_id="observer", adapter=adapter)

        try:
            crew_mem.write("project", "analysis", "CrewAI findings")

            chat = AMFSChatMemory(lc_mem, session_key="s1")
            chat.save_context({"input": "q"}, {"output": "a"})

            store = AMFSMemoryStore(ag_mem, entity_path="project")
            store.add("decision", "ship it")

            stats = observer.stats()
            assert stats.total_entries >= 3
            assert stats.total_agents >= 3
        finally:
            crew_mem.close()
            lc_mem.close()
            ag_mem.close()
            observer.close()
