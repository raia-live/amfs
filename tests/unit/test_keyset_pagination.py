"""Keyset pagination: the cursor codec, ``has_more`` semantics, the paginated
routes, and the counts that used to be capped at 10,000 rows.

These run without Postgres. The SQL side of the same contract is exercised in
tests/integration/test_postgres_adapter.py when ``AMFS_TEST_PG_DSN`` is set.
"""

from __future__ import annotations

import types
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from amfs_core.abc import AdapterABC
from amfs_core.models import (
    DecisionTrace,
    Event,
    EventType,
    MemoryEntry,
    OutcomeRecord,
    Provenance,
    TraceEntry,
)
from amfs_core.pagination import (
    DEFAULT_MAX_SCAN_ROWS,
    MAX_PAGE_SIZE,
    InvalidCursorError,
    clamp_limit,
    decode_cursor,
    encode_cursor,
    entry_tiebreak,
    max_scan_rows,
    page_from_overfetch,
    paginate_desc,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(
    i: int,
    *,
    agent: str = "agent-a",
    entity_path: str = "svc/api",
    key: str | None = None,
    written_at: datetime | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key or f"k{i:03d}",
        value={"i": i},
        provenance=Provenance(
            agent_id=agent,
            session_id="s",
            written_at=written_at or (T0 + timedelta(minutes=i)),
        ),
    )


def _trace(i: int, *, agent: str = "agent-a", outcome_ref: str | None = None) -> DecisionTrace:
    return DecisionTrace(
        id=str(uuid.UUID(int=i)),
        agent_id=agent,
        session_id="s",
        outcome_ref=outcome_ref if outcome_ref is not None else f"OUT-{i}",
        outcome_type="success",
        causal_entries=[TraceEntry(entity_path="svc/api", key="k", version=1, confidence=1.0)],
        created_at=T0 + timedelta(minutes=i),
    )


def _event(i: int, *, agent: str = "agent-a") -> Event:
    return Event(
        id=str(uuid.UUID(int=10_000 + i)),
        agent_id=agent,
        event_type=EventType.READ,
        summary=f"read {i}",
        created_at=T0 + timedelta(minutes=i, seconds=30),
    )


# ── cursor codec ──────────────────────────────────────────────────────


class TestCursorCodec:
    def test_round_trip_with_string_tiebreak(self):
        ts = datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=UTC)
        cur = encode_cursor(ts, "abc-123")
        assert isinstance(cur, str) and "=" not in cur  # url-safe, unpadded
        assert decode_cursor(cur) == (ts, "abc-123")

    def test_round_trip_with_composite_tiebreak(self):
        ts = datetime(2026, 3, 4, tzinfo=UTC)
        tb = ["svc/api", "k001", 3]
        assert decode_cursor(encode_cursor(ts, tb)) == (ts, tb)

    def test_naive_timestamp_is_read_back_as_utc(self):
        cur = encode_cursor(datetime(2026, 3, 4, 12, 0, 0), "x")
        ts, _ = decode_cursor(cur)
        assert ts.tzinfo is not None
        assert ts == datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)

    def test_non_utc_timestamp_keeps_its_instant(self):
        from datetime import timezone

        plus9 = timezone(timedelta(hours=9))
        ts = datetime(2026, 3, 4, 21, 0, 0, tzinfo=plus9)
        decoded, _ = decode_cursor(encode_cursor(ts, "x"))
        assert decoded == ts

    @pytest.mark.parametrize("bad", ["", "not base64!", "AAAA", "e30", "WzEsMl0"])
    def test_malformed_cursors_raise(self, bad):
        with pytest.raises(InvalidCursorError):
            decode_cursor(bad)

    def test_entry_tiebreak_is_json_shaped(self):
        e = _entry(1)
        assert entry_tiebreak(e) == ["svc/api", "k001", 1]


class TestLimits:
    def test_clamp_limit(self):
        assert clamp_limit(None) == 100
        assert clamp_limit(0) == 1
        assert clamp_limit(-5) == 1
        assert clamp_limit(50) == 50
        assert clamp_limit(MAX_PAGE_SIZE + 1) == MAX_PAGE_SIZE

    def test_max_scan_rows_reads_env(self, monkeypatch):
        monkeypatch.delenv("AMFS_MAX_SCAN_ROWS", raising=False)
        assert max_scan_rows() == DEFAULT_MAX_SCAN_ROWS
        monkeypatch.setenv("AMFS_MAX_SCAN_ROWS", "250")
        assert max_scan_rows() == 250
        monkeypatch.setenv("AMFS_MAX_SCAN_ROWS", "garbage")
        assert max_scan_rows() == DEFAULT_MAX_SCAN_ROWS
        monkeypatch.setenv("AMFS_MAX_SCAN_ROWS", "0")
        assert max_scan_rows() == DEFAULT_MAX_SCAN_ROWS


# ── in-memory pagination primitives ───────────────────────────────────


class TestPaginateDesc:
    def _walk(self, items, limit, **kw):
        seen, cursor, pages = [], None, 0
        while True:
            page = paginate_desc(
                items,
                timestamp=lambda e: e.provenance.written_at,
                tiebreak=entry_tiebreak,
                limit=limit,
                cursor=cursor,
                **kw,
            )
            pages += 1
            seen.extend(page.items)
            if not page.has_more:
                assert page.next_cursor is None
                return seen, pages
            assert page.next_cursor is not None
            cursor = page.next_cursor

    def test_pages_cover_everything_once_newest_first(self):
        items = [_entry(i) for i in range(23)]
        seen, pages = self._walk(items, 5)
        assert pages == 5
        assert [e.key for e in seen] == [f"k{i:03d}" for i in reversed(range(23))]

    def test_exact_multiple_has_no_phantom_page(self):
        items = [_entry(i) for i in range(10)]
        page = paginate_desc(
            items,
            timestamp=lambda e: e.provenance.written_at,
            tiebreak=entry_tiebreak,
            limit=10,
        )
        assert len(page.items) == 10
        assert page.has_more is False
        assert page.next_cursor is None

    def test_ties_on_timestamp_are_broken_deterministically(self):
        same = T0 + timedelta(hours=1)
        items = [_entry(i, written_at=same) for i in range(7)]
        seen, _ = self._walk(items, 3)
        assert len(seen) == 7
        assert len({e.key for e in seen}) == 7

    def test_offset_is_honoured_without_a_cursor(self):
        items = [_entry(i) for i in range(10)]
        page = paginate_desc(
            items,
            timestamp=lambda e: e.provenance.written_at,
            tiebreak=entry_tiebreak,
            limit=3,
            offset=4,
        )
        assert [e.key for e in page.items] == ["k005", "k004", "k003"]
        assert page.has_more is True

    def test_page_from_overfetch(self):
        rows = [_trace(i) for i in range(6)]
        page = page_from_overfetch(
            rows, limit=5, timestamp=lambda t: t.created_at, tiebreak=lambda t: t.id
        )
        assert len(page.items) == 5 and page.has_more
        assert decode_cursor(page.next_cursor) == (rows[4].created_at, rows[4].id)
        page = page_from_overfetch(
            rows[:5], limit=5, timestamp=lambda t: t.created_at, tiebreak=lambda t: t.id
        )
        assert len(page.items) == 5 and not page.has_more and page.next_cursor is None


# ── filesystem adapter: has_more and agent filtering ─────────────────


@pytest.fixture
def fs_adapter(tmp_path):
    from amfs_filesystem.adapter import FilesystemAdapter

    return FilesystemAdapter(root=tmp_path, namespace="test")


class TestFilesystemListEntriesForAgent:
    # The filesystem layout keys entities by directory name, so these use flat
    # entity paths and an explicit hidden prefix rather than ``_system/``.
    HIDDEN = ("hidden-",)

    def _seed(self, adapter):
        for i in range(12):
            adapter.write(_entry(i, entity_path="svc-api"))
        for i in range(3):
            adapter.write(_entry(100 + i, agent="agent-b", entity_path="svc-other"))
        adapter.write(_entry(200, entity_path="hidden-internal"))
        adapter.write(_entry(300, entity_path="hidden-internal", key="k300"))

    def test_has_more_semantics_across_pages(self, fs_adapter):
        self._seed(fs_adapter)
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            rows = fs_adapter.list_entries_for_agent(
                "agent-a", limit=5 + 1, cursor=cursor, exclude_prefixes=self.HIDDEN
            )
            page = page_from_overfetch(
                rows, limit=5, timestamp=lambda e: e.provenance.written_at, tiebreak=entry_tiebreak
            )
            pages += 1
            seen.extend(e.key for e in page.items)
            if not page.has_more:
                break
            cursor = page.next_cursor
        # 12 of agent-a's entries; the hidden ones are excluded.
        assert pages == 3
        assert seen == [f"k{i:03d}" for i in reversed(range(12))]

    def test_exact_page_boundary_reports_no_more(self, fs_adapter):
        for i in range(5):
            fs_adapter.write(_entry(i, entity_path="svc-api"))
        rows = fs_adapter.list_entries_for_agent("agent-a", limit=6)
        page = page_from_overfetch(
            rows, limit=5, timestamp=lambda e: e.provenance.written_at, tiebreak=entry_tiebreak
        )
        assert len(page.items) == 5 and page.has_more is False

    def test_filters_by_agent_time_window_and_prefix(self, fs_adapter):
        self._seed(fs_adapter)
        rows = fs_adapter.list_entries_for_agent(
            "agent-a",
            since=T0 + timedelta(minutes=3),
            until=T0 + timedelta(minutes=8),
            limit=100,
            exclude_prefixes=self.HIDDEN,
        )
        assert [e.key for e in rows] == ["k007", "k006", "k005", "k004", "k003"]
        assert all(e.provenance.agent_id == "agent-a" for e in rows)

        rows = fs_adapter.list_entries_for_agent("agent-a", limit=100, exclude_prefixes=self.HIDDEN)
        assert not any(e.entity_path.startswith("hidden-") for e in rows)
        assert len(rows) == 12

        rows = fs_adapter.list_entries_for_agent("agent-b", limit=100)
        assert [e.key for e in rows] == ["k102", "k101", "k100"]

        rows = fs_adapter.list_entries_for_agent("agent-a", limit=100, exclude_prefixes=())
        assert "k300" in {e.key for e in rows}

    def test_only_current_versions_are_listed(self, fs_adapter):
        fs_adapter.write(_entry(1, key="dup", entity_path="svc-api"))
        fs_adapter.write(
            _entry(2, key="dup", entity_path="svc-api", written_at=T0 + timedelta(hours=1))
        )
        rows = fs_adapter.list_entries_for_agent("agent-a", limit=10)
        assert [(e.key, e.version) for e in rows] == [("dup", 2)]

    def test_matches_the_abc_default_ordering(self, fs_adapter):
        """The streaming override must page identically to the base class."""
        self._seed(fs_adapter)
        base = AdapterABC.list_entries_for_agent(
            fs_adapter, "agent-a", limit=7, exclude_prefixes=self.HIDDEN
        )
        fast = fs_adapter.list_entries_for_agent("agent-a", limit=7, exclude_prefixes=self.HIDDEN)
        assert [(e.entity_path, e.key, e.version) for e in base] == [
            (e.entity_path, e.key, e.version) for e in fast
        ]


# ── a small in-memory adapter for the route and count tests ──────────


class _FakeAdapter(AdapterABC):
    """Enough of the adapter contract for the paginated routes.

    Lists are held in memory; ``list_traces``/``list_events`` page with the same
    primitive the base class uses, so the routes see the sync contract exactly
    as a filesystem-backed server would.
    """

    def __init__(self, *, traces=(), events=(), entries=(), outcome_total=0):
        self.traces = list(traces)
        self.events = list(events)
        self.entries = list(entries)
        self.outcome_total = outcome_total
        self.calls: list[tuple[str, dict]] = []

    # unused parts of the contract
    def read(self, *a, **k):
        return None

    def write(self, entry):
        return entry

    def watch(self, *a, **k):
        raise NotImplementedError

    def commit_outcome(self, record):
        return []

    def write_batch(self, entries):
        return entries

    def list(self, entity_path=None, **k):
        return list(self.entries)

    def search(self, *a, **k):
        return []

    def stats(self):
        raise NotImplementedError

    def record_outcome(self, *a, **k):
        return []

    def list_outcomes(self, *, entity_path=None, since=None, limit=1000, outcome_ref=None):
        self.calls.append(("list_outcomes", {"limit": limit}))
        return [
            OutcomeRecord(
                outcome_ref=f"O-{i}", outcome_type="success", committed_at=T0, agent_id="agent-a",
            )
            for i in range(min(limit, self.outcome_total))
        ]

    def count_outcomes(self, *, entity_path=None, since=None):
        self.calls.append(("count_outcomes", {}))
        return self.outcome_total

    def list_traces(
        self, *, entity_path=None, agent_id=None, outcome_type=None,
        limit=100, offset=0, cursor=None, since=None, until=None,
    ):
        self.calls.append(("list_traces", {"limit": limit, "offset": offset, "cursor": cursor}))
        rows = [
            t for t in self.traces
            if (agent_id is None or t.agent_id == agent_id)
            and (outcome_type is None or t.outcome_type == outcome_type)
            and (entity_path is None or any(c.entity_path == entity_path for c in t.causal_entries))
            and (since is None or t.created_at >= since)
            and (until is None or t.created_at < until)
        ]
        return paginate_desc(
            rows, timestamp=lambda t: t.created_at, tiebreak=lambda t: t.id,
            limit=limit, cursor=cursor, offset=offset,
        ).items

    def list_events(
        self, agent_id, namespace="default", *, branch=None, event_type=None,
        since=None, limit=100, offset=0, cursor=None, until=None,
    ):
        self.calls.append(("list_events", {"limit": limit, "offset": offset, "cursor": cursor}))
        rows = [
            e for e in self.events
            if e.agent_id == agent_id
            and (event_type is None or e.event_type.value == event_type)
            and (since is None or e.created_at >= since)
            and (until is None or e.created_at < until)
        ]
        return paginate_desc(
            rows, timestamp=lambda e: e.created_at, tiebreak=lambda e: e.id,
            limit=limit, cursor=cursor, offset=offset,
        ).items


# ── the routes ────────────────────────────────────────────────────────


pytest.importorskip("fastapi", reason="fastapi not installed")
import amfs_http.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _client(monkeypatch, adapter):
    stats = types.SimpleNamespace(
        total_entries=3, total_agents=1, agents={"agent-a": 3}, entities={"svc/api": 3},
    )
    mem = types.SimpleNamespace(
        namespace="test-ns",
        _adapter=adapter,
        list=lambda entity_path=None, branch="main": list(adapter.entries),
        stats=lambda: stats,
    )
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: None)
    monkeypatch.setattr(server, "_active_visibility_filter", lambda request: None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    monkeypatch.setattr(server, "_get_db_pool", lambda: None)
    return TestClient(server.app)


class TestTracesRoute:
    def test_pages_with_cursor_and_reports_has_more(self, monkeypatch):
        adapter = _FakeAdapter(traces=[_trace(i) for i in range(7)])
        client = _client(monkeypatch, adapter)

        r = client.get("/api/v1/traces", params={"limit": 3})
        body = r.json()
        assert r.status_code == 200
        assert [t["outcome_ref"] for t in body["traces"]] == ["OUT-6", "OUT-5", "OUT-4"]
        assert body["has_more"] is True and body["next_cursor"]
        # overfetch by one so has_more is exact
        assert adapter.calls[-1] == ("list_traces", {"limit": 4, "offset": 0, "cursor": None})

        r = client.get("/api/v1/traces", params={"limit": 3, "cursor": body["next_cursor"]})
        body = r.json()
        assert [t["outcome_ref"] for t in body["traces"]] == ["OUT-3", "OUT-2", "OUT-1"]
        assert body["has_more"] is True

        r = client.get("/api/v1/traces", params={"limit": 3, "cursor": body["next_cursor"]})
        body = r.json()
        assert [t["outcome_ref"] for t in body["traces"]] == ["OUT-0"]
        assert body["has_more"] is False and body["next_cursor"] is None

    def test_offset_still_works_without_a_cursor(self, monkeypatch):
        adapter = _FakeAdapter(traces=[_trace(i) for i in range(7)])
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/traces", params={"limit": 2, "offset": 2}).json()
        assert [t["outcome_ref"] for t in body["traces"]] == ["OUT-4", "OUT-3"]
        assert body["has_more"] is True

    def test_bad_cursor_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _FakeAdapter())
        r = client.get("/api/v1/traces", params={"cursor": "definitely-not-a-cursor"})
        assert r.status_code == 400
        assert "cursor" in r.json()["detail"].lower()

    def test_limit_is_clamped(self, monkeypatch):
        adapter = _FakeAdapter(traces=[_trace(i) for i in range(3)])
        client = _client(monkeypatch, adapter)
        assert client.get("/api/v1/traces", params={"limit": 50_000}).status_code == 200
        assert adapter.calls[-1][1]["limit"] == MAX_PAGE_SIZE + 1

    def test_since_until_bound_the_query_not_the_page(self, monkeypatch):
        # A window that excludes the newest traces must still return the older
        # matches on the first page. Filtering the page after the fact would
        # return [] here and stop any cursor walk before reaching OUT-1/OUT-2.
        adapter = _FakeAdapter(traces=[_trace(i) for i in range(7)])
        client = _client(monkeypatch, adapter)
        r = client.get(
            "/api/v1/traces",
            params={
                "limit": 2,
                "since": (T0 + timedelta(minutes=1)).isoformat(),
                "until": (T0 + timedelta(minutes=3)).isoformat(),  # exclusive
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert [t["outcome_ref"] for t in body["traces"]] == ["OUT-2", "OUT-1"]
        assert body["has_more"] is False

    def test_naive_since_until_are_taken_as_utc(self, monkeypatch):
        # Stored timestamps are aware; a naive query value must not reach the
        # adapter's comparison naive (TypeError) or be read in the session
        # timezone. Same window as above, written without an offset.
        adapter = _FakeAdapter(traces=[_trace(i) for i in range(7)])
        client = _client(monkeypatch, adapter)
        r = client.get(
            "/api/v1/traces",
            params={"since": "2026-01-01T00:01:00", "until": "2026-01-01T00:03:00"},
        )
        assert r.status_code == 200
        assert [t["outcome_ref"] for t in r.json()["traces"]] == ["OUT-2", "OUT-1"]

    def test_bad_timestamp_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _FakeAdapter())
        r = client.get("/api/v1/traces", params={"since": "yesterday"})
        assert r.status_code == 400
        assert "timestamp" in r.json()["detail"].lower()


class TestTimelineRoute:
    def test_pages_with_cursor(self, monkeypatch):
        adapter = _FakeAdapter(events=[_event(i) for i in range(5)])
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/agents/agent-a/timeline", params={"limit": 2}).json()
        assert [e["summary"] for e in body["events"]] == ["read 4", "read 3"]
        assert body["count"] == 2 and body["has_more"] is True
        body = client.get(
            "/api/v1/agents/agent-a/timeline",
            params={"limit": 2, "cursor": body["next_cursor"]},
        ).json()
        assert [e["summary"] for e in body["events"]] == ["read 2", "read 1"]
        body = client.get(
            "/api/v1/agents/agent-a/timeline",
            params={"limit": 2, "cursor": body["next_cursor"]},
        ).json()
        assert [e["summary"] for e in body["events"]] == ["read 0"]
        assert body["has_more"] is False and body["next_cursor"] is None

    def test_since_until_window(self, monkeypatch):
        adapter = _FakeAdapter(events=[_event(i) for i in range(5)])
        client = _client(monkeypatch, adapter)
        body = client.get(
            "/api/v1/agents/agent-a/timeline",
            params={
                "since": (T0 + timedelta(minutes=1)).isoformat(),
                "until": (T0 + timedelta(minutes=3)).isoformat(),
            },
        ).json()
        assert [e["summary"] for e in body["events"]] == ["read 2", "read 1"]

    def test_bad_timestamp_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _FakeAdapter())
        r = client.get("/api/v1/agents/agent-a/timeline", params={"since": "yesterday"})
        assert r.status_code == 400


class TestActivityRoute:
    def _adapter(self):
        entries = [_entry(i) for i in range(6)]  # agent-a writes at T0+0..5 min
        entries += [_entry(50 + i, agent="agent-b") for i in range(3)]  # not agent-a
        entries += [_entry(90, entity_path="_system/x")]  # hidden prefix
        traces = [_trace(i) for i in range(4)]  # outcomes at T0+0..3 min
        traces.append(_trace(20, outcome_ref=""))  # no outcome_ref: never shown
        traces.append(_trace(30, agent="agent-b"))  # not agent-a
        events = [_event(i) for i in range(3)]  # T0+0..2 min +30s
        events.append(_event(40, agent="agent-b"))
        return _FakeAdapter(traces=traces, events=events, entries=entries)

    def test_filters_to_the_agent_and_merges_newest_first(self, monkeypatch):
        client = _client(monkeypatch, self._adapter())
        body = client.get("/api/v1/agents/agent-a/activity", params={"limit": 100}).json()
        assert body["agentId"] == "agent-a"
        assert body["has_more"] is False and body["next_cursor"] is None
        kinds = [(t["type"], t["timestamp"]) for t in body["timeline"]]
        # 6 writes + 4 outcomes + 3 read events; agent-b and _system/ excluded,
        # the trace without an outcome_ref dropped.
        assert len(kinds) == 13
        stamps = [k[1] for k in kinds]
        assert stamps == sorted(stamps, reverse=True)
        assert {k[0] for k in kinds} == {"write", "outcome", "read"}
        assert all(t.get("outcomeRef", "x") for t in body["timeline"])
        assert not any(t.get("entityPath", "").startswith("_system/") for t in body["timeline"])

    def test_pages_cover_the_merged_feed_exactly_once(self, monkeypatch):
        client = _client(monkeypatch, self._adapter())
        full = client.get("/api/v1/agents/agent-a/activity", params={"limit": 100}).json()
        expected = [(t["type"], t["timestamp"]) for t in full["timeline"]]

        seen: list[tuple[str, str]] = []
        cursor = None
        for _ in range(20):
            params = {"limit": 4}
            if cursor:
                params["cursor"] = cursor
            body = client.get("/api/v1/agents/agent-a/activity", params=params).json()
            seen.extend((t["type"], t["timestamp"]) for t in body["timeline"])
            if not body["has_more"]:
                assert body["next_cursor"] is None
                break
            cursor = body["next_cursor"]
        assert seen == expected

    def test_since_until_bound_every_source(self, monkeypatch):
        client = _client(monkeypatch, self._adapter())
        body = client.get(
            "/api/v1/agents/agent-a/activity",
            params={
                "since": (T0 + timedelta(minutes=1)).isoformat(),
                "until": (T0 + timedelta(minutes=3)).isoformat(),
            },
        ).json()
        stamps = [datetime.fromisoformat(t["timestamp"]) for t in body["timeline"]]
        lo, hi = T0 + timedelta(minutes=1), T0 + timedelta(minutes=3)
        assert stamps and all(lo <= s < hi for s in stamps)
        assert {t["type"] for t in body["timeline"]} == {"write", "outcome", "read"}

    def test_reads_bounded_pages_not_the_whole_namespace(self, monkeypatch):
        adapter = self._adapter()
        client = _client(monkeypatch, adapter)
        client.get("/api/v1/agents/agent-a/activity", params={"limit": 2})
        limits = {
            name: kw["limit"]
            for name, kw in adapter.calls
            if name in ("list_traces", "list_events")
        }
        assert limits == {"list_traces": 3, "list_events": 3}

    def test_foreign_cursor_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, self._adapter())
        cur = encode_cursor(T0, "a-plain-trace-cursor")
        r = client.get("/api/v1/agents/agent-a/activity", params={"cursor": cur})
        assert r.status_code == 400


class TestAdminUsageCount:
    def test_decision_trace_quota_counts_past_ten_thousand(self, monkeypatch):
        adapter = _FakeAdapter(outcome_total=25_000)
        client = _client(monkeypatch, adapter)
        r = client.get("/api/v1/admin/usage")
        assert r.status_code == 200
        quotas = {q["label"]: q["current"] for q in r.json()["quotas"]}
        assert quotas["Decision traces"] == 25_000
        assert ("count_outcomes", {}) in adapter.calls
        assert not any(name == "list_outcomes" for name, _ in adapter.calls)

    def test_base_class_count_is_bounded_by_the_scan_ceiling(self, monkeypatch):
        """Adapters without a SQL count fall back to a capped scan, so the
        ceiling is what limits them, and it is configurable."""
        adapter = _FakeAdapter(outcome_total=25_000)
        monkeypatch.setenv("AMFS_MAX_SCAN_ROWS", "500")
        assert AdapterABC.count_outcomes(adapter) == 500
        assert adapter.calls[-1] == ("list_outcomes", {"limit": 500})
