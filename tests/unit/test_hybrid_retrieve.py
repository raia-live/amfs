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
import math
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

    def test_rerank_does_not_discard_confidence_on_a_near_tie(self, monkeypatch):
        """The reranker supplies the relevance term; it does not overrule the rest.

        Measured on dev before this was fixed: the cross-encoder replaced the
        whole composite, so a memory discredited by eight critical_failure
        outcomes (confidence 0.9834 -> 0.268) still came back first, ahead of a
        near-identical memory validated twelve times at confidence 1.0. The
        logits behind that were 7.2307 and 7.2206 — a distinction the model does
        not mean to draw, and the only thing deciding the order.
        """
        discredited = _entry("looptest/deploy", "migration-order",
                             "Run schema migrations before the deploy, never after.",
                             confidence=0.268)
        validated = _entry("looptest/deploy", "migration-sequence",
                           "Schema migrations must run ahead of the deploy, not after it.",
                           confidence=1.0)
        fake = _FakeAdapter(semantic_hits=[(discredited, 0.807), (validated, 0.847)],
                            lexical_hits=[])
        monkeypatch.setattr(server, "_async_adapter", fake)

        class _RR:
            available = True

            def rerank(self, query, docs):
                # The real logits, in the order the docs are handed over.
                return [7.2307 if "never after" in d else 7.2206 for d in docs]

        monkeypatch.setattr(server, "_retrieval_reranker", _RR())
        out = _run(_request(), RetrieveRequest(query="run migrations before or after deploy",
                                               limit=10))
        keys = [d["key"] for d in out]
        assert keys[0] == "migration-sequence", (
            "a memory validated by outcomes must outrank one the outcomes "
            f"discredited when the cross-encoder is all but indifferent, got {keys}"
        )
        # Logits are squashed, not min-maxed: two near-identical scores must stay
        # near-identical rather than being stretched to the ends of the range.
        norms = [d["_breakdown"]["rerank_normalised"] for d in out]
        assert abs(norms[0] - norms[1]) < 0.01, norms

    def test_rerank_keeps_confidence_decisive_for_identical_text(self, monkeypatch):
        """Same text, different confidence: the better-validated one wins.

        The cleanest form of the dev finding — two entries whose text was
        byte-identical, at confidence 0.5 and 1.0, came back with the 0.5 one
        ranked higher, on logits of 5.4855 against 5.3320. Identical text cannot
        differ in relevance, so that ordering was pure cross-encoder jitter
        deciding a question confidence had already answered.
        """
        text = "Migrations belong before the deploy, not after it."
        weak = _entry("looptest/deploy", "aaa-note", text, confidence=0.5)
        strong = _entry("looptest/deploy", "zzz-note", text, confidence=1.0)
        fake = _FakeAdapter(semantic_hits=[(weak, 0.8192), (strong, 0.8192)],
                            lexical_hits=[])
        monkeypatch.setattr(server, "_async_adapter", fake)

        class _RR:
            available = True

            def rerank(self, query, docs):
                # The observed logits: jitter favouring the weaker entry.
                return [5.4855 if "aaa-note" in d else 5.3320 for d in docs]

        monkeypatch.setattr(server, "_retrieval_reranker", _RR())
        out = _run(_request(), RetrieveRequest(query="migrations before deploy", limit=10))
        assert [d["key"] for d in out][0] == "zzz-note"

    def test_a_confident_reranker_is_not_flattened_by_where_its_logits_sit(
        self, monkeypatch
    ):
        """The saturation regression, measured on dev after the first fix.

        A logistic is steep only near zero. Where the whole batch sits far out on
        one tail — the cross-encoder confident about every candidate — a large
        genuine difference arrived as almost nothing: raw spreads of 3.04 and
        6.04 survived as 0.0002 and 0.0047. Confidence then decided a comparison
        relevance had already settled, and a tangential entry outranked one the
        reranker preferred by 2.73 logits which also carried twelve validated
        outcomes.

        Both logits here are past +6, so on the uncentred form they map to 0.9999
        and 0.9989 and the 0.3 confidence gap wins. Centred on the batch median
        the same spread is worth 0.59, and relevance decides — which is the whole
        point of handing the reranker the relevance term.
        """
        direct = _entry("looptest/deploy", "direct-answer",
                        "Roll back by pinning the previous revision and shifting traffic.",
                        confidence=0.7)
        tangential = _entry("looptest/saturation", "tangential-high-confidence",
                            "Deployment involves several coordinated services.",
                            confidence=1.0)
        fake = _FakeAdapter(semantic_hits=[(direct, 0.82), (tangential, 0.82)],
                            lexical_hits=[])
        monkeypatch.setattr(server, "_async_adapter", fake)

        class _RR:
            available = True

            def rerank(self, query, docs):
                return [9.5 if "pinning" in d else 6.8 for d in docs]

        monkeypatch.setattr(server, "_retrieval_reranker", _RR())
        out = _run(_request(), RetrieveRequest(query="how do I roll back a deploy",
                                               limit=10))
        keys = [d["key"] for d in out]
        assert keys[0] == "direct-answer", (
            "a 2.7-logit relevance gap must survive normalisation even when both "
            f"logits sit deep in the tail, got {keys}"
        )

    def test_the_same_failure_on_the_negative_tail(self, monkeypatch):
        """Saturation is symmetric, so the fix has to be position-independent.

        Two candidates the cross-encoder dislikes, one much less than the other.
        Uncentred these map to 0.0025 and 0.0001, a difference of nothing.
        """
        better = _entry("looptest/deploy", "less-bad", "Traffic shifting notes.",
                        confidence=0.6)
        worse = _entry("looptest/deploy", "more-bad", "Unrelated billing notes.",
                       confidence=1.0)
        fake = _FakeAdapter(semantic_hits=[(better, 0.5), (worse, 0.5)], lexical_hits=[])
        monkeypatch.setattr(server, "_async_adapter", fake)

        class _RR:
            available = True

            def rerank(self, query, docs):
                return [-6.0 if "Traffic" in d else -9.0 for d in docs]

        monkeypatch.setattr(server, "_retrieval_reranker", _RR())
        out = _run(_request(), RetrieveRequest(query="traffic shifting", limit=10))
        assert [d["key"] for d in out][0] == "less-bad"


class TestRerankNormalisation:
    """Properties of the normaliser itself, independent of the ranking around it.

    Pinned here because the function has now been wrong in two opposite
    directions — min-max amplified noise, an absolute logistic discarded real
    differences in the tails — and the two constraints pull against each other,
    so a future change that satisfies one can silently break the other.
    """

    def test_a_gap_still_counts_for_something_wherever_the_batch_sits(self):
        """The original defect: position all but erased what a gap was worth.

        Not equality. Clamping takes some of the difference back when the batch
        is already extreme, which is unavoidable in a bounded range and lands in
        the region where compressing toward a tie is the right answer anyway.
        The claim is that a real gap stays decisive rather than becoming noise:
        the flat logistic gave 0.0015 here, against a 0.06 confidence gap.
        """
        near_zero = server._normalise_rerank([1.0, -1.0])
        far_out = server._normalise_rerank([21.0, 19.0])
        assert abs(near_zero[0] - near_zero[1]) > 0.4
        assert abs(far_out[0] - far_out[1]) > 0.4

    def test_a_strong_batch_stays_high_for_the_tail_to_be_compared_against(self):
        """The reranker only scores the top N, and the rest keep raw similarity.

        Step 8 re-sorts the reranked head together with that tail, so the two
        have to share a scale. Centring alone put the median of any batch at
        exactly 0.5 however good it was, which would drop the reranker's own
        favourites below entries it never judged.
        """
        strong = server._normalise_rerank([9.1, 9.0, 8.9])
        assert min(strong) > 0.5, strong
        assert max(strong) > 0.95, strong

    def test_a_weak_batch_stays_low_so_the_tail_can_win(self):
        """The same property in the direction that should lose.

        If the cross-encoder rejects everything it was given, an unjudged tail
        candidate outranking them is correct, not a bug.
        """
        weak = server._normalise_rerank([-9.1, -9.0, -8.9])
        assert max(weak) < 0.5, weak

    def test_a_small_spread_stays_small(self):
        """The anti-min-max constraint, which the fix must not trade away.

        The observed jitter pair. Min-max would send these to 1.0 and 0.0 and
        make cross-encoder noise the most decisive signal in the blend.
        """
        norms = server._normalise_rerank([7.2307, 7.2206])
        assert abs(norms[0] - norms[1]) < 0.01

    def test_a_real_difference_survives(self):
        """The other side of the same constraint: 2.7 logits is not noise."""
        norms = server._normalise_rerank([9.5, 6.8])
        assert abs(norms[0] - norms[1]) > 0.5

    def test_order_is_never_changed_by_normalising(self):
        raw = [3.1, -8.0, 7.25, 7.24, 0.0]
        norms = server._normalise_rerank(raw)
        assert [n for _, n in sorted(zip(raw, norms), key=lambda p: p[0])] == sorted(norms)

    def test_scores_already_calibrated_are_left_alone(self):
        """A reranker returning probabilities is taken at its word."""
        assert server._normalise_rerank([0.99, 0.01]) == [0.99, 0.01]

    def test_degenerate_batches(self):
        """With nothing to compare against, a score is worth its absolute value.

        One candidate, or several identical ones, means the centred term is
        exactly 0.5 and the anchor is all that remains — which is the plain
        logistic, and the right answer: there is no peer comparison to make, so
        only "how good is this at all" is left to say.
        """
        assert server._normalise_rerank([]) == []
        assert server._normalise_rerank([12.0]) == [pytest.approx(1.0, abs=1e-5)]
        # Not [0.0, 0.0], which is inside 0..1 and so read as calibrated
        # probabilities and returned untouched by the branch above.
        same = server._normalise_rerank([4.0, 4.0])
        assert same[0] == same[1] == pytest.approx(1 / (1 + math.exp(-4.0)))
        assert server._normalise_rerank([-3.0, -3.0]) == [
            pytest.approx(1 / (1 + math.exp(3.0))), pytest.approx(1 / (1 + math.exp(3.0)))
        ]

    def test_extreme_logits_do_not_overflow(self):
        out = server._normalise_rerank([-2000.0, 2000.0])
        assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(1.0)

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


class TestAdaptiveKKeepsTheQuerysSubject:
    def test_the_entry_carrying_the_rare_term_survives_a_validated_leader(self, monkeypatch):
        """A validated note about checkout, written in the query's own phrasing,
        leads on record for a *pricing* request. The untested pricing runbook is
        a hair less relevant and would have been pruned behind it; it is the
        only candidate carrying "pricing", so the graded lexical term keeps it.
        A second checkout note with the leader's phrasing and none of the
        query's rare terms is pruned, as adaptive_k intends."""
        leader = _entry("acme/deploy", "checkout-deploy-order",
                        "checkout service standard deploy correct step order: migrate, warm, roll",
                        confidence=0.98)
        leader = leader.model_copy(update={"success_count": 3, "last_outcome": "success"})
        runbook = _entry("acme/deploy", "runbook-pricing-a",
                         "pricing service standard deploy: warm, migrate, roll", confidence=0.7)
        other = _entry("acme/deploy", "checkout-deploy-note",
                       "checkout service deploy correct step order confirmed again", confidence=0.9)
        # The rest of a store that has been learning: notes about other
        # services in the same phrasing. This is what makes "correct step
        # order" common and "pricing" rare — IDF is read over the pool.
        crowd = [
            _entry("acme/deploy", f"{svc}-standard-deploy-order",
                   f"{svc} service standard deploy correct step order: warm, migrate, roll",
                   confidence=0.9)
            for svc in ("fraud", "shipping", "inventory", "notifications", "loyalty")
        ]
        fake = _FakeAdapter(
            # The leader is semantically closest by a margin the bi-encoder
            # does produce for a note phrased like the query; the runbook's
            # relevance lands a hair under it once trust is left out.
            semantic_hits=[(leader, 0.98), (runbook, 0.70), (other, 0.76)]
            + [(e, 0.70 - i * 0.01) for i, e in enumerate(crowd)],
            lexical_hits=[leader, runbook, other, *crowd],
        )
        monkeypatch.setattr(server, "_async_adapter", fake)
        out = _run(_request(), RetrieveRequest(
            query="pricing service standard deploy correct step order",
            entity_path="acme/deploy", limit=10, adaptive_k=True,
        ))
        keys = [d["key"] for d in out if not d.get("_meta")]
        assert keys[0] == "checkout-deploy-order", keys
        assert "runbook-pricing-a" in keys, keys
        assert "checkout-deploy-note" not in keys, keys
        by_key = {d["key"]: d["_breakdown"] for d in out if not d.get("_meta")}
        assert by_key["runbook-pricing-a"]["relevance"] < by_key["checkout-deploy-order"]["relevance"]
        assert by_key["runbook-pricing-a"]["keyword"] > by_key["checkout-deploy-order"]["keyword"] + 0.2
