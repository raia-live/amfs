"""Unit tests for TracePatternExtractor (WS1: trace mining)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import (
    DecisionTrace,
    Digest,
    DigestType,
    ExternalContext,
    MemoryType,
    TraceEntry,
)
from amfs_cortex.trace_miner import TracePatternExtractor


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _trace(
    agent_id: str,
    *,
    entity_path: str = "svc",
    keys: list[str] | None = None,
    contexts: list[tuple[str, str]] | None = None,
    written_by_map: dict[str, str] | None = None,
) -> DecisionTrace:
    """Build a mock DecisionTrace with causal entries and external contexts."""
    causal = []
    for k in keys or []:
        wb = (written_by_map or {}).get(k, agent_id)
        causal.append(TraceEntry(
            entity_path=entity_path, key=k, version=1,
            confidence=1.0, written_by=wb,
        ))
    ext = [
        ExternalContext(label=label, summary="s", source=src)
        for label, src in (contexts or [])
    ]
    return DecisionTrace(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        session_id="s1",
        outcome_type="success",
        causal_entries=causal,
        external_contexts=ext,
    )


def _mock_adapter(traces: list[DecisionTrace] | None = None):
    adapter = MagicMock()
    adapter.list_traces.return_value = traces or []
    return adapter


# ──────────────────────────────────────────────────────────────
# extract_patterns
# ──────────────────────────────────────────────────────────────


class TestExtractPatterns:
    def test_returns_entries_with_enough_traces(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1", "k2"]),
            _trace("a3", keys=["k1", "k2"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")

        assert len(patterns) >= 1

    def test_too_few_traces_returns_empty(self) -> None:
        traces = [_trace("a1", keys=["k1", "k2"])]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")

        assert patterns == []

    def test_patterns_are_experience_type(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1", "k2"]),
            _trace("a3", keys=["k1", "k2"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")

        for p in patterns:
            assert p.memory_type == MemoryType.EXPERIENCE

    def test_patterns_capped_at_ten(self) -> None:
        keys = [f"k{i}" for i in range(20)]
        traces = [_trace(f"a{i}", keys=keys) for i in range(5)]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")

        assert len(patterns) <= 10

    def test_patterns_have_correct_entity_path(self) -> None:
        traces = [
            _trace("a1", entity_path="myapp/auth", keys=["k1", "k2"]),
            _trace("a2", entity_path="myapp/auth", keys=["k1", "k2"]),
            _trace("a3", entity_path="myapp/auth", keys=["k1", "k2"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter)

        patterns = extractor.extract_patterns("myapp/auth")

        for p in patterns:
            assert p.entity_path == "myapp/auth"


# ──────────────────────────────────────────────────────────────
# Read co-occurrence
# ──────────────────────────────────────────────────────────────


class TestReadCooccurrence:
    def test_cooccurring_keys_produce_pattern(self) -> None:
        traces = [
            _trace("a1", keys=["config", "secrets"]),
            _trace("a2", keys=["config", "secrets"]),
            _trace("a3", keys=["config", "secrets"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        cooccurrence = [p for p in patterns if "read_cooccurrence" in str(p.value)]
        assert len(cooccurrence) >= 1
        assert "config" in str(cooccurrence[0].value)
        assert "secrets" in str(cooccurrence[0].value)

    def test_keys_in_only_one_trace_no_pattern(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k3", "k4"]),
            _trace("a3", keys=["k5", "k6"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        cooccurrence = [p for p in patterns if "read_cooccurrence" in str(p.value)]
        assert len(cooccurrence) == 0

    def test_single_key_traces_skipped(self) -> None:
        traces = [
            _trace("a1", keys=["k1"]),
            _trace("a2", keys=["k1"]),
            _trace("a3", keys=["k1"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        cooccurrence = [p for p in patterns if "read_cooccurrence" in str(p.value)]
        assert len(cooccurrence) == 0

    def test_pattern_key_format(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1", "k2"]),
            _trace("a3", keys=["k1", "k2"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        cooccurrence = [p for p in patterns if "read_cooccurrence" in str(p.value)]
        if cooccurrence:
            assert cooccurrence[0].key.startswith("pattern-read-cooccurrence-")


# ──────────────────────────────────────────────────────────────
# Context patterns (ExternalContext handling)
# ──────────────────────────────────────────────────────────────


class TestContextPatterns:
    def test_recurring_pydantic_contexts_produce_pattern(self) -> None:
        """Regression test: ExternalContext is a Pydantic model, not a dict."""
        traces = [
            _trace("a1", keys=["k1"], contexts=[("pagerduty-check", "pagerduty")]),
            _trace("a2", keys=["k1"], contexts=[("pagerduty-check", "pagerduty")]),
            _trace("a3", keys=["k1"], contexts=[("pagerduty-check", "pagerduty")]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        ctx_patterns = [p for p in patterns if "recurring_context" in str(p.value)]
        assert len(ctx_patterns) >= 1
        assert "pagerduty-check" in str(ctx_patterns[0].value)

    def test_infrequent_context_no_pattern(self) -> None:
        traces = [
            _trace("a1", keys=["k1"], contexts=[("rare-check", "tool")]),
            _trace("a2", keys=["k1"], contexts=[("other-check", "tool")]),
            _trace("a3", keys=["k1"], contexts=[("another-check", "tool")]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        ctx_patterns = [p for p in patterns if "recurring_context" in str(p.value)]
        assert len(ctx_patterns) == 0

    def test_compile_digest_handles_pydantic_external_context(self) -> None:
        """Regression test: compile_trace_digest must use attribute access on ExternalContext."""
        traces = [
            _trace("a1", keys=["k1"], contexts=[("deploy-check", "ci")]),
            _trace("a2", keys=["k1"], contexts=[("deploy-check", "ci")]),
            _trace("a3", keys=["k1"], contexts=[("deploy-check", "ci")]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is not None
        assert "deploy-check" in str(digest.summary.get("recurring_contexts", {}))


# ──────────────────────────────────────────────────────────────
# Agent sequences
# ──────────────────────────────────────────────────────────────


class TestAgentSequences:
    def test_recurring_sequence_produces_pattern(self) -> None:
        wb_map = {"k1": "agent-infra", "k2": "agent-deploy"}
        traces = [
            _trace("a1", keys=["k1", "k2"], written_by_map=wb_map),
            _trace("a2", keys=["k1", "k2"], written_by_map=wb_map),
            _trace("a3", keys=["k1", "k2"], written_by_map=wb_map),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        seq_patterns = [p for p in patterns if "agent_sequence" in str(p.value)]
        assert len(seq_patterns) >= 1

    def test_single_agent_traces_no_sequence(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1", "k2"]),
            _trace("a3", keys=["k1", "k2"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        seq_patterns = [p for p in patterns if "agent_sequence" in str(p.value)]
        assert len(seq_patterns) == 0

    def test_sequence_preserves_order(self) -> None:
        wb_map = {"k1": "step-1", "k2": "step-2", "k3": "step-3"}
        traces = [
            _trace(f"a{i}", keys=["k1", "k2", "k3"], written_by_map=wb_map)
            for i in range(3)
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        patterns = extractor.extract_patterns("svc")
        seq_patterns = [p for p in patterns if "agent_sequence" in str(p.value)]
        if seq_patterns:
            seq = seq_patterns[0].value.get("sequence", [])
            assert seq == ["step-1", "step-2", "step-3"]


# ──────────────────────────────────────────────────────────────
# compile_trace_digest
# ──────────────────────────────────────────────────────────────


class TestCompileTraceDigest:
    def test_returns_digest_with_correct_type(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1"]),
            _trace("a3", keys=["k1", "k3"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is not None
        assert digest.digest_type == DigestType.TRACE_PATTERN
        assert digest.scope == "svc"

    def test_returns_none_for_too_few_traces(self) -> None:
        traces = [_trace("a1", keys=["k1"])]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is None

    def test_summary_contains_expected_fields(self) -> None:
        traces = [
            _trace("a1", keys=["k1", "k2"]),
            _trace("a2", keys=["k1"]),
            _trace("a3", keys=["k2", "k3"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is not None
        summary = digest.summary
        assert "narrative" in summary
        assert "trace_count" in summary
        assert summary["trace_count"] == 3
        assert "agents" in summary
        assert set(summary["agents"]) == {"a1", "a2", "a3"}
        assert "top_read_keys" in summary
        assert "total_reads" in summary
        assert summary["total_reads"] == 5

    def test_digest_entry_count_is_trace_count(self) -> None:
        traces = [_trace(f"a{i}", keys=["k1"]) for i in range(5)]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is not None
        assert digest.entry_count == 5

    def test_digest_source_agents_populated(self) -> None:
        traces = [
            _trace("alpha", keys=["k1"]),
            _trace("beta", keys=["k1"]),
            _trace("gamma", keys=["k1"]),
        ]
        adapter = _mock_adapter(traces)
        extractor = TracePatternExtractor(adapter, min_traces=3)

        digest = extractor.compile_trace_digest("svc")

        assert digest is not None
        assert sorted(digest.source_agents) == ["alpha", "beta", "gamma"]
