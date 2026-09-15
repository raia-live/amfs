"""Whose eyes the agent detail page looks through.

An agent's card in the directory is counted by SQL over the rows themselves. Its
page was assembled from ``mem.list()`` on the server's process-wide handle, and
``list`` keeps an entry only if it is shared or belongs to the listing identity::

    if e.shared or e.provenance.agent_id == self.agent_id

The listing identity there is the server, never the agent being viewed, so an
agent's own PRIVATE entries were invisible on its own page. Measured in
production: ``sidekick-sandbox-v1-guard-03a3777260`` holds four entries under one
path, every one of them ``shared=False``; its card read 4 memories and 1 topic and
its page read 0 and 0. Nothing about branches or ``_system/`` paths was involved,
which is what the first reading of the code had concluded.

The fix asks as the agent — ``mem.as_agent(agent_id).list()`` — which is also the
identity a write on that agent's behalf needs, so one primitive covers both.

These go through the ASGI stack because the route is what was wrong, and the
counts a page shows are what a user compares against the card.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_core.models import MemoryType
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_http import server
from fastapi.testclient import TestClient

GUARD = "sidekick-sandbox-v1-guard-03a3777260"
GUARD_PATH = "sidekick/sandbox/v1/agents/guard/episodes"
OTHER_AGENT = "some-other-agent"


@pytest.fixture
def mem(monkeypatch, tmp_path) -> AgentMemory:
    """The server's one shared handle, as ``_get_memory`` returns it."""
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="amfs-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
    return handle


@pytest.fixture
def client(mem) -> TestClient:
    return TestClient(server.app)


def _guard_saves_four_private_episodes(mem: AgentMemory) -> None:
    """The production shape: one path, four keys, none of them shared."""
    author = mem.as_agent(GUARD)
    for i in range(4):
        author.write(
            GUARD_PATH,
            f"episodes-{i}",
            f"episode {i}: guard route held, noise suppressed",
            memory_type=MemoryType.FACT,
            shared=False,
        )


class TestTheAgentsOwnPrivateEntriesAppearOnItsPage:
    def test_the_page_counts_what_the_card_counts(self, client, mem) -> None:
        _guard_saves_four_private_episodes(mem)

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        written = [n for n in body["nodes"] if n.get("writtenEntries")]
        total = sum(len(n["writtenEntries"]) for n in written)
        assert total == 4, (
            "the page found none of the agent's four private entries, which is "
            f"the 4-vs-0 divergence: {body['nodes']}"
        )
        assert len(written) == 1, f"one path, so one topic: {[n['id'] for n in written]}"

    def test_it_is_the_private_flag_that_used_to_decide(self, client, mem) -> None:
        """The same four entries, shared this time, were always visible.

        Pins the mechanism rather than the symptom: if a later change makes the
        two cases differ again, the shared case passing while the private one
        fails is the signature to look for.
        """
        author = mem.as_agent(GUARD)
        for i in range(4):
            author.write(
                GUARD_PATH, f"episodes-{i}", f"episode {i}", shared=True
            )

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()
        total = sum(len(n.get("writtenEntries") or []) for n in body["nodes"])
        assert total == 4


class TestItStillShowsOnlyThisAgentsWork:
    def test_another_agents_private_entries_stay_out(self, client, mem) -> None:
        _guard_saves_four_private_episodes(mem)
        mem.as_agent(OTHER_AGENT).write(
            "someone/else", "secret", "not for this page", shared=False
        )

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        paths = [n["entityPath"] for n in body["nodes"] if n.get("writtenEntries")]
        assert "someone/else" not in paths, f"leaked another agent's entry: {paths}"

    def test_another_agents_shared_entries_stay_out_too(self, client, mem) -> None:
        """Shared means readable, not authored — the page lists authorship."""
        _guard_saves_four_private_episodes(mem)
        mem.as_agent(OTHER_AGENT).write(
            "someone/else", "public", "shared with everyone", shared=True
        )

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        paths = [n["entityPath"] for n in body["nodes"] if n.get("writtenEntries")]
        assert "someone/else" not in paths, f"listed as this agent's work: {paths}"

    def test_system_paths_are_still_dropped(self, client, mem) -> None:
        _guard_saves_four_private_episodes(mem)
        mem.as_agent(GUARD).write("_system/telemetry", "beat", "internal", shared=False)

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        paths = [n["entityPath"] for n in body["nodes"] if n.get("writtenEntries")]
        assert not any(p.startswith("_system/") for p in paths), paths
