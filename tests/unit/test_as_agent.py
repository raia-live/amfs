"""Acting as another agent, without mutating a handle everyone shares.

A server holds one ``AgentMemory`` for the whole process and must sometimes act
for one of its callers: write a room's join briefing as the agent being briefed,
list an agent's own entries to render its page. The way it was done was::

    original = mem._tagger.agent_id
    mem._tagger.agent_id = agent_id
    try:
        mem.write(...)
    finally:
        mem._tagger.agent_id = original

which stamps every concurrent write in that window with the wrong agent. In
production, 1,994 room-join briefings sat under ``<agent>/briefings/rooms/...``
while the row belonged to someone else, and 14 real agents were credited with
briefings for rooms they had never joined — which is how one agent's page came to
list a stranger's topics.

``as_agent`` returns a separate handle onto the same store, so the caller's
identity is never touched and there is no window to race.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter

SERVER = "amfs-server"
BRIEFED = "continual-learning-audit-agent"


@pytest.fixture
def mem(tmp_path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    return AgentMemory(agent_id=SERVER, adapter=adapter)


class TestTheWriteIsAttributedToTheAgentActedFor:
    def test_provenance_names_the_agent_not_the_server(self, mem) -> None:
        mem.as_agent(BRIEFED).write(
            f"{BRIEFED}/briefings/rooms/bruno/seed-investors",
            "join-briefing-20260914T113732",
            "room history",
        )

        entry = mem.read(
            f"{BRIEFED}/briefings/rooms/bruno/seed-investors",
            "join-briefing-20260914T113732",
        )
        assert entry is not None
        assert entry.provenance.agent_id == BRIEFED, (
            "the path named the briefed agent but the row belonged to "
            f"{entry.provenance.agent_id} — the production bug exactly"
        )

    def test_the_shared_handle_keeps_its_own_identity(self, mem) -> None:
        """The point of the change: nothing about ``mem`` moves."""
        mem.as_agent(BRIEFED).write("some/path", "k", "v")

        assert mem.agent_id == SERVER
        mem.write("server/own", "k", "v")
        assert mem.read("server/own", "k").provenance.agent_id == SERVER

    def test_two_impersonations_do_not_interfere(self, mem) -> None:
        """Held at once, as concurrent requests would hold them."""
        a, b = mem.as_agent("agent-a"), mem.as_agent("agent-b")

        a.write("a/path", "k", "written by a")
        b.write("b/path", "k", "written by b")

        assert mem.read("a/path", "k").provenance.agent_id == "agent-a"
        assert mem.read("b/path", "k").provenance.agent_id == "agent-b"
        assert mem.agent_id == SERVER


class TestReadingAsTheAgentSeesWhatTheAgentSees:
    def test_its_private_entries_become_visible(self, mem) -> None:
        """``list`` filters by the listing identity, so identity is a read concern
        too — this is what the agent detail page needed."""
        mem.as_agent(BRIEFED).write("acme/billing", "runbook", "rotate then redeploy", shared=False)

        assert mem.list("acme/billing") == [], "setup wrong: should be invisible to the server"
        as_them = mem.as_agent(BRIEFED).list("acme/billing")
        assert [e.key for e in as_them] == ["runbook"]


class TestItSharesTheStoreAndNotTheSession:
    def test_the_adapter_is_the_same_object(self, mem) -> None:
        """Cheap enough to call per request, and writes land in one store."""
        assert mem.as_agent(BRIEFED).adapter is mem.adapter

    def test_reads_do_not_enter_the_callers_causal_chain(self, mem) -> None:
        """Work done for another agent must not reinforce the caller's entries.

        Sharing the tracker would put entries the caller never read into its next
        ``commit_outcome`` — the reinforcement loop crediting the wrong memory.
        """
        mem.write("acme/billing", "runbook", "rotate then redeploy")
        other = mem.as_agent(BRIEFED)

        other.read("acme/billing", "runbook")

        assert other.read_log == ["acme/billing/runbook"]
        assert mem.read_log == [], (
            f"the caller's chain picked up a read it never made: {mem.read_log}"
        )

    def test_session_state_is_not_inherited(self, mem) -> None:
        mem.record_context("upstream", "three SEV-1s in the last 24h", source="PagerDuty")

        assert mem.as_agent(BRIEFED)._read_tracker._contexts == []
        assert mem._read_tracker._contexts, "setup wrong: the caller recorded one"
