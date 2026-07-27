"""Tier 1 hybrid retrieval tests.

Covers the pieces that make /api/v1/retrieve robust to vaguely/temporally-worded
queries without a live Postgres:
  - or_tsquery: multi-term queries OR their terms (recall) instead of ANDing.
  - normalize_temporal: temporal intent -> recency signal, not embedded noise.
  - retrieve_entries: semantic UNION lexical candidate generation, visibility
    applied over the merged set, Pro-injected rerank, and the abstain floor.

The retrieve_entries tests drive the real endpoint coroutine against a fake
async adapter + fake embedder by monkeypatching the module globals, so they
exercise the exact serving-path logic.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime, timedelta, timezone

import amfs_http.server as server
import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_core.query_norm import normalize_temporal
from amfs_http.models import RetrieveRequest, SearchRequest
from amfs_postgres._fts import or_tsquery

# ──────────────────────────────────────────────────────────────────────
# or_tsquery
# ──────────────────────────────────────────────────────────────────────

class TestOrTsquery:
    def test_single_term_uses_plainto(self):
        sql, params = or_tsquery("browser")
        assert sql == "plainto_tsquery('english', %s)"
        assert params == ["browser"]

    def test_multi_term_ors_each_term(self):
        sql, params = or_tsquery("browser plugin extension")
        # One plainto per term, OR-combined with ||.
        assert sql.count("plainto_tsquery('english', %s)") == 3
        assert " || " in sql
        assert params == ["browser", "plugin", "extension"]

    def test_dedupes_and_drops_punctuation(self):
        sql, params = or_tsquery("browser, browser plugin!")
        assert params == ["browser", "plugin"]
        assert sql.count("%s") == 2

    def test_empty_query_is_safe(self):
        sql, params = or_tsquery("")
        assert "plainto_tsquery" in sql
        assert params == [""]


# ──────────────────────────────────────────────────────────────────────
# normalize_temporal
# ──────────────────────────────────────────────────────────────────────

class TestNormalizeTemporal:
    def test_strips_yesterday_and_sets_window(self):
        now = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
        r = normalize_temporal("browser plugin delivered yesterday", now=now)
        assert r.matched is True
        assert "yesterday" not in r.topical.lower()
        assert "browser plugin delivered" in r.topical
        assert r.written_after == now - timedelta(days=2)
        assert r.recency_weight_boost > 1.0

    def test_last_week(self):
        now = datetime(2026, 7, 23, tzinfo=timezone.utc)
        r = normalize_temporal("what did we ship last week", now=now)
        assert r.matched
        assert r.written_after == now - timedelta(days=14)

    def test_numeric_phrase(self):
        now = datetime(2026, 7, 23, tzinfo=timezone.utc)
        r = normalize_temporal("bug fixed 3 days ago", now=now)
        assert r.matched
        # 3 days + 1 day buffer
        assert r.written_after == now - timedelta(days=4)

    def test_no_temporal_is_noop(self):
        r = normalize_temporal("stripe checkout integration")
        assert r.matched is False
        assert r.recency_weight_boost == 1.0
        assert r.written_after is None
        assert r.topical == "stripe checkout integration"

    def test_only_temporal_keeps_original(self):
        r = normalize_temporal("yesterday")
        assert r.topical  # never empty
        assert r.matched


# ──────────────────────────────────────────────────────────────────────
# retrieve_entries hybrid serving path
# ──────────────────────────────────────────────────────────────────────

def _entry(entity_path: str, key: str, value: str, *, confidence: float = 0.9,
           days_old: float = 1.0) -> MemoryEntry:
    written = datetime.now(timezone.utc) - timedelta(days=days_old)
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        confidence=confidence,
        provenance=Provenance(agent_id="a", session_id="s", written_at=written),
    )


class _FakeAdapter:
    """Deterministic stand-in: returns preset semantic + lexical hits so the
    union/dedup/visibility logic can be asserted without a database."""

    _has_is_artifact_col = True

    def __init__(self, semantic_hits=None, lexical_hits=None):
        self._semantic = semantic_hits or []
        self._lexical = lexical_hits or []

    async def semantic_search(self, query, embedder, *, branch="main"):
        return list(self._semantic)

    async def search(self, query, *, branch="main"):
        return list(self._lexical)


class _Vis:
    def __init__(self, allowed_keys):
        self._allowed = set(allowed_keys)

    def should_filter(self):
        return True

    def filter_entries(self, entries):
        return [e for e in entries if e.entry_key in self._allowed]


def _request(vis=None):
    state = types.SimpleNamespace(visibility_filter=vis)
    return types.SimpleNamespace(state=state)


def _run(request, req):
    return asyncio.run(server.retrieve_entries(request, req, _auth=None))


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    monkeypatch.setattr(server, "_retrieval_reranker", None, raising=False)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None, raising=False)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: object())
    # Default floor for determinism unless a test overrides.
    monkeypatch.setenv("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")
    yield


class TestHybridUnion:
    def test_lexical_only_hit_surfaces(self, monkeypatch):
        """A relevant entry the vector neighbours miss must still surface via
        the lexical channel (the browser-extension incident)."""
        target = _entry("amfs/browser-extension", "task-summary",
                        "SenseLab web clipper browser extension connect flow")
        noise = _entry("app/ui", "index.css", "body { margin: 0 }", confidence=0.95)
        fake = _FakeAdapter(semantic_hits=[(noise, 0.22)], lexical_hits=[target])
        monkeypatch.setattr(server, "_async_adapter", fake)

        req = RetrieveRequest(query="browser plugin extension delivered yesterday", limit=10)
        out = _run(_request(), req)
        keys = [d["entity_path"] + "/" + d["key"] for d in out]
        assert "amfs/browser-extension/task-summary" in keys
        # keyword-matched, recent target should outrank the weak-sim css noise
        assert keys[0] == "amfs/browser-extension/task-summary"

    def test_visibility_applied_to_merged_set(self, monkeypatch):
        """A foreign entry that only matched lexically must be filtered — the
        union must not create a visibility bypass."""
        mine = _entry("amfs/browser-extension", "task-summary", "browser extension clipper")
        foreign = _entry("other/secret", "leak", "browser extension private note")
        fake = _FakeAdapter(semantic_hits=[], lexical_hits=[mine, foreign])
        monkeypatch.setattr(server, "_async_adapter", fake)

        vis = _Vis(allowed_keys={mine.entry_key})
        out = _run(_request(vis=vis), RetrieveRequest(query="browser extension", limit=10))
        keys = {d["entity_path"] + "/" + d["key"] for d in out}
        assert mine.entry_key in keys
        assert foreign.entry_key not in keys

    def test_excluded_namespaces_dropped(self, monkeypatch):
        real = _entry("amfs/browser-extension", "task-summary", "browser extension clipper")
        bench = _entry("bench-run-1/obs", "row", "browser extension benchmark row")
        system = _entry("_system/telemetry", "row", "browser extension system row")
        fake = _FakeAdapter(semantic_hits=[], lexical_hits=[real, bench, system])
        monkeypatch.setattr(server, "_async_adapter", fake)

        out = _run(_request(), RetrieveRequest(query="browser extension", limit=10))
        keys = {d["entity_path"] + "/" + d["key"] for d in out}
        assert real.entry_key in keys
        assert bench.entry_key not in keys
        assert system.entry_key not in keys

    def test_injected_reranker_reorders(self, monkeypatch):
        low = _entry("amfs/browser-extension", "target", "browser extension clipper")
        high = _entry("app/ui", "other", "browser extension unrelated", confidence=0.99)
        # Blend would favour `high` (higher confidence); reranker rescues target.
        fake = _FakeAdapter(semantic_hits=[(low, 0.4), (high, 0.4)], lexical_hits=[])
        monkeypatch.setattr(server, "_async_adapter", fake)

        class _RR:
            available = True

            def rerank(self, query, docs):
                # Score the target doc highest.
                return [0.99 if "clipper" in d else 0.01 for d in docs]

        monkeypatch.setattr(server, "_retrieval_reranker", _RR())
        out = _run(_request(), RetrieveRequest(query="browser extension", limit=10))
        assert out[0]["entity_path"] + "/" + out[0]["key"] == low.entry_key
        assert "rerank" in out[0]["_breakdown"]

    def test_query_rewriter_expands(self, monkeypatch):
        calls = {}

        class _RW:
            def expand(self, q):
                calls["q"] = q
                return [q, "add-on plugin"]

        recorded = []

        class _CapturingAdapter(_FakeAdapter):
            async def semantic_search(self, query, embedder, *, branch="main"):
                recorded.append(query.text)
                return []

        fake = _CapturingAdapter(semantic_hits=[], lexical_hits=[
            _entry("amfs/browser-extension", "t", "browser extension")
        ])
        monkeypatch.setattr(server, "_async_adapter", fake)
        monkeypatch.setattr(server, "_retrieval_query_rewriter", _RW())
        _run(_request(), RetrieveRequest(query="browser extension", limit=10))
        # Both the original topical query and the rewriter paraphrase were searched.
        assert "add-on plugin" in recorded

    def test_abstain_trims_low_sim_no_keyword_tail(self, monkeypatch):
        target = _entry("amfs/browser-extension", "task-summary", "browser extension clipper")
        junk = _entry("app/ui", "index.css", "body{}", confidence=0.95)
        # target: keyword hit (survives). junk: semantic 0.05 (< floor) + no keyword -> trimmed.
        fake = _FakeAdapter(semantic_hits=[(junk, 0.05)], lexical_hits=[target])
        monkeypatch.setattr(server, "_async_adapter", fake)
        monkeypatch.setenv("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")

        out = _run(_request(), RetrieveRequest(query="browser extension", limit=10))
        keys = {d["entity_path"] + "/" + d["key"] for d in out}
        assert target.entry_key in keys
        assert junk.entry_key not in keys


# ──────────────────────────────────────────────────────────────────────
# search_entries reuse accounting
#
# `amfs_search` is the read surface some agent profiles (e.g. the Base44
# builder profile) expose instead of /retrieve. A text-driven search is a real
# recall and must bump recall_count, or reuse metrics read 0 even when memory
# was used. Browse/filter calls (no query) must NOT bump.
# ──────────────────────────────────────────────────────────────────────

class _RecordingSearchAdapter:
    def __init__(self, hits):
        self._hits = hits
        self.bumped: list[tuple[str, str]] = []

    async def search(self, query, *, branch="main"):
        return list(self._hits)

    async def increment_recall_count(self, entity_path, key, *, branch="main"):
        self.bumped.append((entity_path, key))


def _run_search(request, req):
    return asyncio.run(server.search_entries(request, req, _auth=None))


class TestSearchReuseAccounting:
    def test_query_driven_search_bumps_only_the_top_hit(self, monkeypatch):
        hits = [
            _entry("invoicehub", "deferred-decisions", "recurring invoices deferred"),
            _entry("invoicehub", "schema", "client and invoice entities"),
            _entry("invoicehub", "layout", "sidebar clients invoices"),
            _entry("invoicehub", "extra", "fourth hit should not be credited"),
        ]
        fake = _RecordingSearchAdapter(hits)
        monkeypatch.setattr(server, "_async_adapter", fake)

        _run_search(_request(), SearchRequest(query="invoicehub decisions", limit=20))
        # Reuse counts the memory the agent took, not the list it was shown.
        # Crediting the head of the list inflated reuse ~3x — one real session
        # made 4 lookups and was credited 16 reuses, of which 1 was used.
        assert fake.bumped == [("invoicehub", "deferred-decisions")]

    def test_browse_without_query_does_not_bump(self, monkeypatch):
        hits = [_entry("invoicehub", "deferred-decisions", "recurring invoices deferred")]
        fake = _RecordingSearchAdapter(hits)
        monkeypatch.setattr(server, "_async_adapter", fake)

        # Pure filter/browse (no query text) is not a recall — must not bump.
        _run_search(_request(), SearchRequest(agent_id="base44-builder", limit=20))
        assert fake.bumped == []

    def test_system_and_bench_namespaces_skipped(self, monkeypatch):
        # Scratch rows outranking the real hit must not consume the credit —
        # with a single credit to give, skipping them has to mean "keep
        # looking", not "spend it here".
        hits = [
            _entry("_system/telemetry", "row", "system row matches query"),
            _entry("bench-run-1/obs", "row", "benchmark row matches query"),
            _entry("invoicehub", "deferred-decisions", "real recall target"),
        ]
        fake = _RecordingSearchAdapter(hits)
        monkeypatch.setattr(server, "_async_adapter", fake)

        _run_search(_request(), SearchRequest(query="matches query", limit=20))
        assert fake.bumped == [("invoicehub", "deferred-decisions")]
