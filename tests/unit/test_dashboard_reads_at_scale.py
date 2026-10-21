"""The whole-account reads a dashboard page makes, at scale.

One Agents page mount on a 6,000-agent, 430K-entry account fired a cluster
compile (minutes of Python), a consolidation status that loaded every entry
to count ten paths, a 3 MB roster and a 600 KB stats payload — and stalled
every other request on the instance while it did. These pin what replaced
that: paging and search in SQL with a ``total``, a status counted in SQL,
stats without the per-agent and per-entity maps, digests fetched by scope,
and a cluster graph built from candidate pairs rather than every pair.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
import amfs_http.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _client(monkeypatch, adapter, *, vis=None, visible_agents=None):
    mem = types.SimpleNamespace(
        namespace="ns",
        _adapter=adapter,
        list=lambda entity_path=None, branch="main": list(getattr(adapter, "entries", [])),
    )
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_active_visibility_filter", lambda request: vis)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
    monkeypatch.setattr(
        server, "_visible_agent_ids",
        lambda request: None if visible_agents is None else set(visible_agents),
    )
    return TestClient(server.app)


# ── /agents/enriched pages in SQL ─────────────────────────────────────────


class _PagingAgentsAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def list_agents_enriched(self, namespace="default", *, agent_ids=None, q=None,
                             platform=None, sort="last_active", limit=None, offset=0):
        self.calls.append(dict(
            agent_ids=agent_ids, q=q, platform=platform, sort=sort, limit=limit, offset=offset,
        ))
        if sort not in ("last_active", "entries_written", "created_at", "agent_id"):
            raise ValueError(f"unknown sort {sort!r}")
        if offset >= 3:
            return []
        rows = [{"agent_id": f"a{i}", "entries_written": 10 - i} for i in range(offset, 3)]
        if limit is not None:
            rows = rows[:limit]
            for r in rows:
                r["total"] = 3
                r["active_7d"] = 2
        return rows

    def get_agent_activity_histogram(self, aid, days=7, namespace="default"):
        return [1] * days


class TestEnrichedAgentsPage:
    def test_a_page_carries_the_total_and_the_parameters_reach_sql(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter, visible_agents={"a0", "a2"})
        res = client.get(
            "/api/v1/agents/enriched?limit=2&offset=0&q=bil&platform=cursor"
            "&sort=entries_written&histograms=false"
        )
        assert res.status_code == 200
        body = res.json()
        assert [a["agent_id"] for a in body["agents"]] == ["a0", "a1"]
        assert (body["total"], body["active_7d"], body["limit"], body["offset"]) == (3, 2, 2, 0)
        assert all("total" not in a and "active_7d" not in a for a in body["agents"]), (
            "the counts are on the page only"
        )
        assert adapter.calls == [dict(
            agent_ids=["a0", "a2"], q="bil", platform="cursor", sort="entries_written",
            limit=2, offset=0,
        )]

    def test_a_page_past_the_end_still_reports_the_count(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/agents/enriched?limit=2&offset=10&histograms=false").json()
        assert body["agents"] == [] and body["total"] == 3 and body["active_7d"] == 2
        assert adapter.calls[-1]["limit"] == 1 and adapter.calls[-1]["offset"] == 0

    def test_unpaged_is_the_whole_roster_as_before(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/agents/enriched?histograms=false").json()
        assert len(body["agents"]) == 3 and "total" not in body
        assert adapter.calls == [dict(
            agent_ids=None, q=None, platform=None, sort="last_active", limit=None, offset=0,
        )]

    def test_an_unknown_sort_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _PagingAgentsAdapter())
        assert client.get("/api/v1/agents/enriched?sort=karma").status_code == 400

    def test_an_adapter_without_paging_is_filtered_here(self, monkeypatch):
        class _Old:
            def list_agents_enriched(self, namespace="default"):
                return [{"agent_id": "a0"}, {"agent_id": "hidden"}]

            def get_agent_activity_histogram(self, aid, days=7, namespace="default"):
                return [0] * days

        client = _client(monkeypatch, _Old(), visible_agents={"a0"})
        body = client.get("/api/v1/agents/enriched?limit=1&histograms=false").json()
        assert [a["agent_id"] for a in body["agents"]] == ["a0"]
        assert "total" not in body, "no SQL paging happened, so no page count is claimed"


# ── /entities pages in SQL ────────────────────────────────────────────────


class _PagingEntitiesAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def entity_summaries(self, *, scope=None, q=None, agent=None, min_confidence=None,
                         sort="last_updated", limit=None, offset=0):
        self.calls.append(dict(
            scope=scope, q=q, agent=agent, min_confidence=min_confidence, sort=sort,
            limit=limit, offset=offset,
        ))
        rows = [{"entity_path": f"acme/{p}", "entry_count": 5 - i} for i, p in enumerate("abcd")]
        if q:
            rows = [r for r in rows if q in r["entity_path"]]
        total = len(rows)
        if limit is not None:
            rows = rows[offset:offset + limit]
            for r in rows:
                r["total"] = total
        return rows


class TestEntitySummariesPage:
    def test_a_page_with_search_and_total(self, monkeypatch):
        adapter = _PagingEntitiesAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get(
            "/api/v1/entities?limit=2&offset=1&q=acme&agent=bob&min_confidence=0.7"
            "&sort=entry_count"
        ).json()
        assert [e["entity_path"] for e in body["entities"]] == ["acme/b", "acme/c"]
        assert (body["total"], body["limit"], body["offset"]) == (4, 2, 1)
        assert all("total" not in e for e in body["entities"])
        assert adapter.calls == [dict(
            scope=None, q="acme", agent="bob", min_confidence=0.7, sort="entry_count",
            limit=2, offset=1,
        )]

    def test_unpaged_unchanged(self, monkeypatch):
        adapter = _PagingEntitiesAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/entities").json()
        assert len(body["entities"]) == 4 and "total" not in body
        assert adapter.calls == [dict(
            scope=None, q=None, agent=None, min_confidence=None, sort="last_updated",
            limit=None, offset=0,
        )]

    def test_an_older_adapter_is_searched_and_paged_here(self, monkeypatch):
        class _Old:
            def entity_summaries(self):
                return [
                    {"entity_path": "acme/a", "entry_count": 1, "agents": ["bob"],
                     "avg_confidence": 0.95},
                    {"entity_path": "beta/b", "entry_count": 9, "agents": ["bob"],
                     "avg_confidence": 0.9},
                    {"entity_path": "acme/c", "entry_count": 4, "agents": ["bob"],
                     "avg_confidence": 0.8},
                    {"entity_path": "acme/d", "entry_count": 7, "agents": ["eve"],
                     "avg_confidence": 0.99},
                    {"entity_path": "acme/e", "entry_count": 8, "agents": ["bob"],
                     "avg_confidence": 0.3},
                ]

        client = _client(monkeypatch, _Old())
        body = client.get(
            "/api/v1/entities?limit=1&offset=1&q=acme&agent=bob&min_confidence=0.7"
            "&sort=entry_count"
        ).json()
        assert [e["entity_path"] for e in body["entities"]] == ["acme/a"]
        assert body["total"] == 2, "eve's path and the low-confidence one are filtered here"
        low_first = client.get("/api/v1/entities?sort=avg_confidence_asc&limit=2").json()
        assert [e["entity_path"] for e in low_first["entities"]] == ["acme/e", "acme/c"]

    def test_an_unknown_sort_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _PagingEntitiesAdapter())
        assert client.get("/api/v1/entities?sort=vibes").status_code == 400


# ── /stats without the maps ───────────────────────────────────────────────


class _StatsAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def stats_extended(self, *, scope=None, breakdown=True):
        self.calls.append(dict(scope=scope, breakdown=breakdown))
        out = {"total_entries": 7, "total_agents": 2, "total_entities": 3, "total_recalls": 1}
        if breakdown:
            out["agents"] = {"a": 4, "b": 3}
            out["entities"] = {"x": 7}
        return out


class TestStatsBreakdown:
    def test_breakdown_false_leaves_the_maps_out_and_keeps_the_totals(self, monkeypatch):
        adapter = _StatsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/stats?breakdown=false").json()
        assert "agents" not in body and "entities" not in body
        assert (body["total_agents"], body["total_entities"]) == (2, 3)
        assert adapter.calls == [dict(scope=None, breakdown=False)]

    def test_default_is_the_full_shape(self, monkeypatch):
        adapter = _StatsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/stats").json()
        assert body["agents"] == {"a": 4, "b": 3}
        assert adapter.calls == [dict(scope=None, breakdown=True)]

    def test_an_adapter_without_the_option_is_trimmed_here(self, monkeypatch):
        class _Old:
            def stats_extended(self):
                return {"total_entries": 1, "total_agents": 1, "total_entities": 1,
                        "agents": {"a": 1}, "entities": {"x": 1}}

        client = _client(monkeypatch, _Old())
        body = client.get("/api/v1/stats?breakdown=false").json()
        assert "agents" not in body and body["total_agents"] == 1

    def test_the_loaded_and_filtered_route_carries_the_buckets_too(self, monkeypatch):
        """A caller whose visibility is not a SQL predicate gets its stats
        from loaded entries. ``breakdown=false`` drops the per-entity map
        that the health donut used to count from, so the buckets must be
        on this route as well, from the same definition."""
        from datetime import UTC, datetime

        from amfs_core.models import MemoryEntry, Provenance

        def _e(key, path, conf, agent="a"):
            return MemoryEntry(
                entity_path=path, key=key, value="v", confidence=conf,
                provenance=Provenance(
                    agent_id=agent, session_id="s", written_at=datetime.now(UTC),
                ),
            )

        rows = [
            _e("k1", "acme/x", 1.0), _e("k2", "acme/x", 0.9),     # mean 0.95 → healthy
            _e("k3", "acme/y", 0.5),                              # low
            _e("k4", "acme/z", 0.2, agent="hidden"),              # filtered out below
        ]

        class _Adapter:
            entries = rows

            def stats_extended(self, *, scope=None, breakdown=True):
                raise AssertionError("the SQL aggregate must not run for this caller")

        class _Vis:
            def filter_entries(self, rows):
                return [r for r in rows if r.provenance.agent_id != "hidden"]

        monkeypatch.setattr(server, "_visibility_scope", lambda request: (None, _Vis()))
        client = _client(monkeypatch, _Adapter())
        body = client.get("/api/v1/stats?breakdown=false").json()
        assert "entities" not in body
        assert body["total_entities"] == 2
        assert body["entity_confidence"] == {
            "avg": pytest.approx(0.725), "healthy": 1, "moderate": 0, "low": 1, "critical": 0,
        }


def test_the_bucket_edges_are_one_definition():
    from amfs_core.aggregates import entity_confidence_buckets

    assert entity_confidence_buckets([0.9, 0.7, 0.4, 0.39, 0.89]) == {
        "avg": pytest.approx(0.656), "healthy": 1, "moderate": 2, "low": 1, "critical": 1,
    }
    assert entity_confidence_buckets([]) == {
        "avg": 0.0, "healthy": 0, "moderate": 0, "low": 0, "critical": 0,
    }


# ── consolidation status counts in SQL ────────────────────────────────────


class TestConsolidationStatus:
    def test_ready_entities_come_from_one_grouped_query(self, monkeypatch):
        calls: list[dict] = []

        class _Adapter:
            def list_branches(self, namespace="default", status="active"):
                return []

            def consolidation_ready_entities(self, *, min_entries, limit, scope=None):
                calls.append(dict(min_entries=min_entries, limit=limit, scope=scope))
                return ["acme/billing", "acme/auth"]

            def list(self, *a, **kw):
                raise AssertionError("the status must not load the account's entries")

        client = _client(monkeypatch, _Adapter())
        monkeypatch.setattr(server, "_cortex_worker", None)
        body = client.get("/api/v1/cortex/consolidation/status").json()
        assert body["entities_ready"] == ["acme/billing", "acme/auth"]
        assert calls == [dict(min_entries=10, limit=10, scope=None)]

    def test_an_adapter_without_the_query_counts_loaded_entries(self, monkeypatch):
        entries = [types.SimpleNamespace(entity_path="big") for _ in range(10)]
        entries += [types.SimpleNamespace(entity_path="small") for _ in range(3)]

        class _Adapter:
            def list_branches(self, namespace="default", status="active"):
                return []

            def list(self, *a, **kw):
                return list(entries)

        client = _client(monkeypatch, _Adapter())
        monkeypatch.setattr(server, "_cortex_worker", None)
        assert client.get("/api/v1/cortex/consolidation/status").json()["entities_ready"] == ["big"]


# ── /cortex/digests filters in SQL ────────────────────────────────────────


def test_digests_scope_and_limit_reach_the_adapter(monkeypatch):
    from amfs_postgres.adapter import PostgresAdapter

    calls: list[dict] = []
    adapter = object.__new__(PostgresAdapter)
    adapter._namespace = "ns"

    def list_digests(digest_type=None, namespace="default", branch="main", *, scope=None,
                     written_by=None, relevant_to=None, limit=None):
        calls.append(dict(
            digest_type=digest_type, namespace=namespace, scope=scope, written_by=written_by,
            limit=limit,
        ))
        return []

    adapter.list_digests = list_digests
    client = _client(monkeypatch, adapter)
    body = client.get(
        "/api/v1/cortex/digests?digest_type=agent_brief&scope=billing-agent&limit=1"
    ).json()
    assert body == {"digests": [], "total": 0}
    client.get("/api/v1/cortex/digests?digest_type=entity&written_by=billing-agent&limit=50")
    assert len(calls) == 2
    assert (calls[0]["digest_type"].value, calls[0]["scope"], calls[0]["limit"]) == (
        "agent_brief", "billing-agent", 1,
    )
    assert calls[0]["written_by"] is None
    assert (calls[1]["digest_type"].value, calls[1]["written_by"], calls[1]["limit"]) == (
        "entity", "billing-agent", 50,
    )


def test_written_by_resolves_the_agents_topics_as_the_topics_page_would():
    """The subquery is the set ``entity_summaries`` lists for the agent: the
    digests' branch, live rows only, no benchmark or system rows. Without
    that, a main-branch entity digest showed up on an agent's list from a
    write on another branch or under ``_system/``."""
    import contextlib

    from amfs_postgres.adapter import _EXCLUDE_SYSTEM_ROWS, PostgresAdapter

    executed: list[tuple[str, tuple]] = []

    class _Cur:
        def execute(self, sql, params):
            executed.append((sql, params))
            return self

        def fetchall(self):
            return []

    class _Pool:
        @contextlib.contextmanager
        def connection(self):
            conn = types.SimpleNamespace(
                cursor=lambda row_factory=None: contextlib.nullcontext(_Cur()),
            )
            yield conn

    adapter = object.__new__(PostgresAdapter)
    adapter._pool = _Pool()
    adapter._get_current_account_id = lambda: None
    adapter.list_digests(namespace="ns", branch="feature", written_by="billing-agent", limit=5)

    (sql, params), = executed
    sub = sql[sql.index("scope IN ("):]
    assert sub.startswith(
        "scope IN (SELECT DISTINCT entity_path FROM amfs_memory_entries"
        " WHERE namespace = %s AND branch = %s AND agent_id = %s"
        f" AND superseded_at IS NULL AND {_EXCLUDE_SYSTEM_ROWS})"
    )
    # namespace, branch (outer); namespace, branch, agent (subquery); limit.
    assert params == ("ns", "feature", "ns", "feature", "billing-agent", 5)


# ── the cluster graph is built from candidate pairs ───────────────────────


class TestClusterCandidates:
    @pytest.fixture
    def strategy(self):
        pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")
        from amfs_cortex.clustering import AgentClusterStrategy

        return AgentClusterStrategy

    def test_pairs_share_a_path_or_an_edge_and_nothing_else_is_scored(self, strategy):
        ids = ["a", "b", "c", "d"]
        paths = {"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": {"z"}}
        collab = {"a": set(), "b": set(), "c": set(), "d": {"a"}}
        assert strategy.candidate_pairs(ids, paths, collab) == [("a", "b"), ("a", "d"), ("b", "c")]

    def test_a_fleet_with_nothing_in_common_proposes_no_pairs(self, strategy):
        ids = [f"agent-{i}" for i in range(5000)]
        paths = {aid: {f"repo-{aid}"} for aid in ids}
        collab = {aid: set() for aid in ids}
        assert strategy.candidate_pairs(ids, paths, collab) == []

    def test_a_hub_path_proposes_nothing_but_still_counts_in_the_score(self, strategy):
        ids = [f"agent-{i}" for i in range(strategy.HUB_PATH_AGENTS + 1)]
        paths = {aid: {"everyone/writes/here"} for aid in ids}
        paths["agent-0"].add("pair")
        paths["agent-1"].add("pair")
        collab = {aid: set() for aid in ids}
        assert strategy.candidate_pairs(ids, paths, collab) == [("agent-0", "agent-1")]

    def test_edges_to_unknown_agents_and_self_are_ignored_and_order_is_stable(self, strategy):
        ids = ["b", "a"]
        paths = {"a": {"x"}, "b": {"x"}}
        collab = {"a": {"a", "ghost", "b"}, "b": set()}
        assert strategy.candidate_pairs(ids, paths, collab) == [("b", "a")]

    def test_type_similarity_alone_no_longer_makes_an_edge(self, strategy):
        """Two agents that share nothing but a taste for facts are not a cluster."""
        from amfs_core.models import GraphEdge

        agents = [
            {"agent_id": a, "entity_paths": [f"only/{a}"], "type_dist": {"fact": 10}}
            for a in ("a", "b", "c")
        ]
        edge = GraphEdge(
            source_entity="x", source_type="agent", relation="learned_from",
            target_entity="y", target_type="agent",
        )
        digest = strategy(scope="account:ns").compute_clusters(agents, [edge])
        assert digest.summary["clusters"] == []


# ── /pro/tiers in SQL ─────────────────────────────────────────────────────


def _tier_rows(n: int):
    """*n* entries whose priority order is their index: a newer, more recalled
    row scores higher, so ``e0`` ranks first and ``e{n-1}`` last."""
    from datetime import UTC, datetime, timedelta

    from amfs_core.models import MemoryEntry, Provenance

    now = datetime.now(UTC)
    return [
        MemoryEntry(
            entity_path="acme/x", key=f"e{i}", value="v", confidence=0.9,
            recall_count=max(0, n - i),
            provenance=Provenance(
                agent_id="a" if i % 2 else "b", session_id="s",
                written_at=now - timedelta(days=i),
            ),
        )
        for i in range(n)
    ]


class _RankingAdapter:
    """Orders by the score in the database; ``list`` must never be needed."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple] = []

    def priority_stats(self, *, scope=None, agent_id=None, now=None, branch="main"):
        self.calls.append(("stats", scope, agent_id))
        rows = [r for r in self.rows if not agent_id or r.provenance.agent_id == agent_id]
        return len(rows), (0.5 if rows else None)

    def list_by_priority(self, *, scope=None, agent_id=None, limit, offset=0, now=None,
                         branch="main"):
        self.calls.append(("page", scope, agent_id, limit, offset))
        rows = [r for r in self.rows if not agent_id or r.provenance.agent_id == agent_id]
        return rows[offset:offset + limit]

    def list(self, *args, **kwargs):
        raise AssertionError("the ranked reads must not load the account")


class TestTiersInSql:
    def test_the_distribution_is_a_count_and_a_mean(self, monkeypatch):
        adapter = _RankingAdapter(_tier_rows(300))
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert (body["total"], body["hot"], body["warm"], body["archive"]) == (300, 50, 200, 50)
        assert body["scored"] == 300 and body["avg_priority_score"] == 0.5
        assert adapter.calls == [("stats", None, None)]

    def test_a_small_account_fills_hot_first(self, monkeypatch):
        client = _client(monkeypatch, _RankingAdapter(_tier_rows(30)))
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert (body["hot"], body["warm"], body["archive"]) == (30, 0, 0)

    def test_an_empty_account_has_no_mean(self, monkeypatch):
        client = _client(monkeypatch, _RankingAdapter([]))
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert body["total"] == 0 and body["avg_priority_score"] is None

    def test_each_tier_is_one_page_at_its_offset(self, monkeypatch):
        adapter = _RankingAdapter(_tier_rows(300))
        client = _client(monkeypatch, adapter)
        hot = client.get("/api/v1/pro/tiers/entries?tier=1&limit=10").json()
        warm = client.get("/api/v1/pro/tiers/entries?tier=2&limit=10").json()
        archive = client.get("/api/v1/pro/tiers/entries?tier=3&limit=10").json()
        assert [e["key"] for e in hot] == [f"e{i}" for i in range(10)]
        assert [e["key"] for e in warm] == [f"e{i}" for i in range(50, 60)]
        assert [e["key"] for e in archive] == [f"e{i}" for i in range(250, 260)]
        assert all(e["tier"] == 2 for e in warm)
        assert [c for c in adapter.calls if c[0] == "page"] == [
            ("page", None, None, 10, 0),
            ("page", None, None, 10, 50),
            ("page", None, None, 10, 250),
        ]
        # Scores are real and descend within the page.
        scores = [e["priority_score"] for e in hot]
        assert scores == sorted(scores, reverse=True) and scores[0] > 0

    def test_a_tier_page_never_exceeds_its_band(self, monkeypatch):
        adapter = _RankingAdapter(_tier_rows(300))
        client = _client(monkeypatch, adapter)
        hot = client.get("/api/v1/pro/tiers/entries?tier=1&limit=500").json()
        assert len(hot) == 50 and adapter.calls[-1] == ("page", None, None, 50, 0)

    def test_the_agent_filter_reaches_sql(self, monkeypatch):
        adapter = _RankingAdapter(_tier_rows(20))
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/pro/tiers/distribution?agent_id=a").json()
        assert body["total"] == 10 and adapter.calls == [("stats", None, "a")]
        page = client.get("/api/v1/pro/tiers/entries?tier=1&limit=3&agent_id=a").json()
        assert [e["agent_id"] for e in page] == ["a", "a", "a"]

    def test_the_scope_reaches_sql(self, monkeypatch):
        from amfs_core.scope import SqlScope

        scope = SqlScope("agent_id = %s", ("a",))

        class _Vis:
            def should_filter(self):
                return True

            def sql_predicate(self):
                return scope

        adapter = _RankingAdapter(_tier_rows(20))
        client = _client(monkeypatch, adapter, vis=_Vis())
        client.get("/api/v1/pro/tiers/distribution")
        client.get("/api/v1/pro/tiers/entries?tier=1&limit=3")
        assert adapter.calls == [("stats", scope, None), ("page", scope, None, 3, 0)]


class TestTiersScoredHere:
    """A visibility rule that only exists in Python, or an adapter that cannot
    order by the score, is scored here — a page at a time, keeping only the
    rows the response needs."""

    class _PagingAdapter:
        def __init__(self, rows):
            self.rows = rows
            self.pages: list[tuple[int, int]] = []

        def list(self, entity_path=None, *, agent_id=None, limit=None, offset=0, **_):
            self.pages.append((limit, offset))
            rows = [r for r in self.rows if not agent_id or r.provenance.agent_id == agent_id]
            return rows[offset:offset + limit] if limit is not None else rows

    class _Vis:
        """Hides every row written by agent ``b``; no SQL form."""

        def should_filter(self):
            return True

        def sql_predicate(self):
            return None

        def filter_entries(self, rows):
            return [r for r in rows if r.provenance.agent_id != "b"]

    def test_the_fallback_pages_and_agrees_with_the_full_ranking(self, monkeypatch):
        from amfs_core.tiering import PriorityScorer, TierAssigner

        rows = _tier_rows(4500)
        adapter = self._PagingAdapter(rows)
        monkeypatch.setattr(server, "_TIER_SCAN_PAGE", 1000)
        client = _client(monkeypatch, adapter, vis=self._Vis())

        body = client.get("/api/v1/pro/tiers/distribution").json()
        visible = [r for r in rows if r.provenance.agent_id == "a"]
        assert body["total"] == len(visible) == 2250
        assert (body["hot"], body["warm"], body["archive"]) == (50, 200, 2000)
        assert adapter.pages == [(1000, 0), (1000, 1000), (1000, 2000), (1000, 3000), (1000, 4000)]

        warm = client.get("/api/v1/pro/tiers/entries?tier=2&limit=5").json()
        tiers, scores = TierAssigner().assign_with_scores(visible, PriorityScorer())
        expected = sorted(
            (r for r in visible if tiers[r.entry_key] == 2),
            key=lambda r: scores[r.entry_key], reverse=True,
        )[:5]
        assert [e["key"] for e in warm] == [r.key for r in expected]
        assert all(e["agent_id"] == "a" for e in warm)

    class _SqlVis:
        """Hides agent ``b`` — as a predicate, and as a Python rule for an
        adapter that cannot take the predicate."""

        def __init__(self):
            from amfs_core.scope import SqlScope

            self.scope = SqlScope("agent_id = %s", ("a",))

        def should_filter(self):
            return True

        def sql_predicate(self):
            return self.scope

        def filter_entries(self, rows):
            return [r for r in rows if r.provenance.agent_id != "b"]

    def test_a_sql_rule_reaches_a_paging_adapter_that_cannot_rank(self, monkeypatch):
        """``_visibility_scope`` hands back the predicate and no Python filter;
        an adapter that pages but cannot order by the score must still be
        narrowed by it, page by page."""
        rows = _tier_rows(40)
        vis = self._SqlVis()

        class _Scoped(self._PagingAdapter):
            def __init__(self, rows):
                super().__init__(rows)
                self.scopes: list = []

            def list(self, entity_path=None, *, scope=None, **kw):
                self.scopes.append(scope)
                rows = super().list(entity_path, **kw)
                if scope is not None:
                    rows = [r for r in rows if r.provenance.agent_id == scope.params[0]]
                return rows

        adapter = _Scoped(rows)
        client = _client(monkeypatch, adapter, vis=vis)
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert body["total"] == 20
        assert adapter.scopes and all(s is vis.scope for s in adapter.scopes)
        hot = client.get("/api/v1/pro/tiers/entries?tier=1&limit=50").json()
        assert len(hot) == 20 and all(e["agent_id"] == "a" for e in hot)

    def test_a_sql_rule_runs_here_when_the_adapter_cannot_take_it(self, monkeypatch):
        """An adapter with neither scope nor paging lists the whole account;
        the rule the predicate came from is then applied over it, so the
        caller never sees rows the predicate would have hidden."""
        rows = _tier_rows(40)

        class _Old:
            entries = rows

            def list(self, entity_path=None, *, include_superseded=False, branch="main"):
                return list(rows)

        client = _client(monkeypatch, _Old(), vis=self._SqlVis())
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert body["total"] == 20
        hot = client.get("/api/v1/pro/tiers/entries?tier=1&limit=50").json()
        assert len(hot) == 20 and all(e["agent_id"] == "a" for e in hot)

    def test_an_adapter_that_cannot_page_is_listed_whole(self, monkeypatch):
        rows = _tier_rows(60)

        class _Old:
            entries = rows

            def list(self, entity_path=None, *, include_superseded=False, branch="main"):
                return list(rows)

        client = _client(monkeypatch, _Old())
        body = client.get("/api/v1/pro/tiers/distribution").json()
        assert (body["total"], body["hot"], body["warm"]) == (60, 50, 10)
        hot = client.get("/api/v1/pro/tiers/entries?tier=1&limit=3").json()
        assert [e["key"] for e in hot] == ["e0", "e1", "e2"]
        archive = client.get("/api/v1/pro/tiers/entries?tier=3&limit=3").json()
        assert archive == [], "60 rows fill Hot and part of Warm; nothing reaches Archive"
