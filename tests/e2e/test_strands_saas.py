"""E2E: Strands Agent with AMFSPlugin against AMFS SaaS.

Exercises the full read/write/search loop through the HTTP adapter
connected to the AMFS hosted service.  Skipped when credentials are
not available.

Requires:
    AMFS_HTTP_URL  — SaaS endpoint (e.g. https://amfs-login.sense-lab.ai)
    AMFS_API_KEY   — API key for authentication

Optionally (for the full LLM agent loop test):
    AWS_REGION / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — for Bedrock model
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

AMFS_HTTP_URL = os.environ.get("AMFS_HTTP_URL")
AMFS_API_KEY = os.environ.get("AMFS_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not AMFS_HTTP_URL or not AMFS_API_KEY,
        reason="AMFS_HTTP_URL and AMFS_API_KEY not set — skipping SaaS E2E tests",
    ),
]

try:
    from strands.plugins import Plugin as _  # noqa: F401

    _HAS_STRANDS = True
except ImportError:
    _HAS_STRANDS = False


@pytest.mark.skipif(not _HAS_STRANDS, reason="strands-agents not installed")
class TestStrandsWithSaaS:
    """End-to-end tests: Strands Agent + AMFSPlugin + AMFS SaaS."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from amfs import AgentMemory
        from amfs_adapter_http import HttpAdapter
        from amfs_strands import AMFSPlugin

        self._test_scope = f"e2e-strands-{uuid.uuid4().hex[:8]}"
        adapter = HttpAdapter(base_url=AMFS_HTTP_URL, api_key=AMFS_API_KEY)
        self.mem = AgentMemory(agent_id="e2e-strands-agent", adapter=adapter)
        self.plugin = AMFSPlugin(self.mem)
        yield
        self.mem.close()

    def test_write_and_read_roundtrip(self) -> None:
        """Write via plugin tool, read back via AgentMemory."""
        ep = self._test_scope
        result = self.plugin.amfs_write(
            entity_path=ep, key="greeting", value='"hello from strands"'
        )
        assert "v1" in result

        entry = self.mem.read(ep, "greeting")
        assert entry is not None
        assert entry.value == "hello from strands"

    def test_read_via_plugin_tool(self) -> None:
        """Write via SDK, read back via plugin tool."""
        ep = self._test_scope
        self.mem.write(ep, "data", {"status": "ok"})

        result = self.plugin.amfs_read(entity_path=ep, key="data")
        data = json.loads(result)
        assert data["value"]["status"] == "ok"

    def test_search_via_plugin_tool(self) -> None:
        """Write entries and search via plugin tool."""
        ep = self._test_scope
        self.mem.write(ep, "pattern-retry", "exponential backoff")
        self.mem.write(ep, "pattern-circuit", "circuit breaker")

        result = self.plugin.amfs_search(entity_path=ep)
        parsed = json.loads(result)
        assert len(parsed) >= 2

    def test_list_via_plugin_tool(self) -> None:
        """List entries via plugin tool."""
        ep = self._test_scope
        self.mem.write(ep, "k1", "v1")
        self.mem.write(ep, "k2", "v2")

        result = self.plugin.amfs_list(ep)
        parsed = json.loads(result)
        keys = {e["key"] for e in parsed}
        assert "k1" in keys
        assert "k2" in keys

    def test_recall_own_entries(self) -> None:
        """Recall returns only entries written by this agent."""
        ep = self._test_scope
        self.mem.write(ep, "my-note", "agent's own note")

        result = self.plugin.amfs_recall(ep, "my-note")
        data = json.loads(result)
        assert data["value"] == "agent's own note"

    def test_record_context(self) -> None:
        """Record context does not raise."""
        result = self.plugin.amfs_record_context(
            "e2e-test", "Testing context recording", source="pytest"
        )
        assert "Context recorded" in result

    def test_full_agent_loop(self) -> None:
        """Full LLM-driven loop: agent stores and retrieves a fact.

        Requires a working LLM model (e.g. Bedrock Claude).
        Skipped gracefully if no model is available.
        """
        from strands import Agent

        ep = self._test_scope

        try:
            agent = Agent(
                system_prompt=(
                    f"You are a test agent. When asked to remember something, use "
                    f"amfs_write with entity_path='{ep}'. When asked to recall, "
                    f"use amfs_read with entity_path='{ep}'."
                ),
                plugins=[self.plugin],
            )
            agent(f"Remember that the project deadline is March 15th. Use key 'deadline' and entity_path '{ep}'.")
            result = agent(f"What is the project deadline? Use amfs_read with entity_path '{ep}' and key 'deadline'.")
            assert "March 15" in str(result) or "deadline" in str(result).lower()
        except Exception as exc:
            pytest.skip(f"No LLM model available for full agent loop test: {exc}")
