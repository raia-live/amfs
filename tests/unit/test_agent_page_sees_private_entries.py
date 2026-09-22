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


class TestThePageNeverScansTheNamespace:
    """208 s in production for an agent with 69 entries.

    ``handle.list()`` with no entity path is every current entry the agent may
    see — 400k rows on a large account — filtered down to the agent's own in
    Python afterwards. The agent's entries come from ``search(agent_id=…)``,
    filtered by author in SQL, and the authors of what it read come from
    listing only the entity paths it actually read.
    """

    def test_no_unscoped_list_and_the_search_is_by_author(self, client, mem, monkeypatch) -> None:
        _guard_saves_four_private_episodes(mem)
        for i in range(50):
            mem.as_agent(OTHER_AGENT).write(f"noise/{i}", "k", f"row {i}", shared=True)

        list_paths: list = []
        search_calls: list = []
        real_list, real_search = AgentMemory.list, AgentMemory.search

        def spy_list(self, entity_path=None, **kw):
            list_paths.append(entity_path)
            return real_list(self, entity_path, **kw)

        def spy_search(self, **kw):
            search_calls.append(kw)
            return real_search(self, **kw)

        monkeypatch.setattr(AgentMemory, "list", spy_list)
        monkeypatch.setattr(AgentMemory, "search", spy_search)

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        assert None not in list_paths, "listed the whole namespace"
        assert search_calls and all(c.get("agent_id") == GUARD for c in search_calls)
        assert body["totalWritten"] == 4
        assert body["truncated"] is False

    def test_cross_agent_reads_still_name_the_author(self, client, mem, monkeypatch) -> None:
        """The namespace list existed to find who wrote what this agent read.
        Listing just the read paths has to give the same answer."""
        _guard_saves_four_private_episodes(mem)
        mem.as_agent(OTHER_AGENT).write("someone/else", "public", "shared", shared=True)
        monkeypatch.setattr(
            mem._adapter, "trace_read_counts",
            lambda agent_id: {"someone/else": {"public": 3}, GUARD_PATH: {"episodes-1": 1}},
            raising=False,
        )

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        assert body["crossAgentReads"] == {
            OTHER_AGENT: [{"entityPath": "someone/else", "key": "public", "readCount": 3}],
        }, body["crossAgentReads"]
        # Its own entry read back is not a cross-agent read.
        assert GUARD not in body["crossAgentReads"]
        assert body["nodes"][0]["readCounts"] or body["nodes"][1]["readCounts"]

    def test_reading_across_too_many_paths_is_bounded_and_flagged(self, client, mem, monkeypatch) -> None:
        _guard_saves_four_private_episodes(mem)
        monkeypatch.setattr(server, "_MEMORY_GRAPH_MAX_READ_PATHS", 3)
        monkeypatch.setattr(
            mem._adapter, "trace_read_counts",
            lambda agent_id: {f"elsewhere/{i}": {"k": 1} for i in range(10)},
            raising=False,
        )
        listed: list = []
        real_list = AgentMemory.list
        monkeypatch.setattr(
            AgentMemory, "list",
            lambda self, entity_path=None, **kw: (listed.append(entity_path), real_list(self, entity_path, **kw))[1],
        )

        body = client.get(f"/api/v1/agents/{GUARD}/memory-graph").json()

        assert len([p for p in listed if p and p.startswith("elsewhere/")]) == 3
        assert body["truncated"] is True
