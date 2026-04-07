"""Unit tests for the Cortex drift gate (HMO Track 3)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from amfs_core.models import Digest, DigestType, MemoryEntry, Provenance


def _make_entry(key: str, confidence: float = 1.0) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc",
        key=key,
        value="v",
        provenance=Provenance(
            agent_id="a", session_id="s", written_at=datetime.now(timezone.utc)
        ),
        confidence=confidence,
    )


class TestScopeFingerprint:
    """Test DigestCompiler.compute_scope_fingerprint()."""

    def test_empty_scope_returns_stable_hash(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        adapter = MagicMock()
        adapter.list.return_value = []
        compiler = DigestCompiler(adapter=adapter)

        fp = compiler.compute_scope_fingerprint("entity:svc")
        assert fp is not None
        assert fp == hashlib.sha256(b"empty").hexdigest()[:16]

    def test_deterministic_for_same_state(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        entries = [_make_entry("k1", 0.9), _make_entry("k2", 0.5)]
        adapter = MagicMock()
        adapter.list.return_value = entries
        compiler = DigestCompiler(adapter=adapter)

        fp1 = compiler.compute_scope_fingerprint("entity:svc")
        fp2 = compiler.compute_scope_fingerprint("entity:svc")
        assert fp1 == fp2

    def test_changes_when_entry_added(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        entries = [_make_entry("k1")]
        adapter = MagicMock()
        adapter.list.return_value = entries
        compiler = DigestCompiler(adapter=adapter)

        fp1 = compiler.compute_scope_fingerprint("entity:svc")

        entries.append(_make_entry("k2"))
        fp2 = compiler.compute_scope_fingerprint("entity:svc")
        assert fp1 != fp2

    def test_changes_when_confidence_changes(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        adapter = MagicMock()
        compiler = DigestCompiler(adapter=adapter)

        adapter.list.return_value = [_make_entry("k1", 1.0)]
        fp1 = compiler.compute_scope_fingerprint("entity:svc")

        adapter.list.return_value = [_make_entry("k1", 0.5)]
        fp2 = compiler.compute_scope_fingerprint("entity:svc")
        assert fp1 != fp2

    def test_unknown_scope_returns_none(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        adapter = MagicMock()
        compiler = DigestCompiler(adapter=adapter)

        fp = compiler.compute_scope_fingerprint("unknown:svc")
        assert fp is None

    def test_no_prefix_returns_none(self) -> None:
        from amfs_cortex.compiler import DigestCompiler

        adapter = MagicMock()
        compiler = DigestCompiler(adapter=adapter)

        fp = compiler.compute_scope_fingerprint("no-prefix")
        assert fp is None


class TestDriftGate:
    """Test CortexWorker drift gate behavior."""

    def _make_worker(self, compiler, drift_threshold: float = 0.10):
        from amfs_cortex.worker import CortexWorker
        return CortexWorker(
            dsn="postgresql://fake",
            compiler=compiler,
            debounce_ms=0,
            use_advisory_lock=False,
            drift_threshold=drift_threshold,
        )

    def test_first_event_always_compiles(self) -> None:
        """No anchor exists, so the first compilation always proceeds."""
        compiler = MagicMock()
        digest = Digest(
            digest_type=DigestType.ENTITY,
            scope="svc",
            summary={"k": "v"},
            entry_count=1,
        )
        compiler.compile.return_value = digest
        compiler.compute_scope_fingerprint.return_value = "abc123"

        worker = self._make_worker(compiler, drift_threshold=0.10)
        worker._pending = {"entity:svc@main": 0}
        worker._recompile_pending()

        compiler.compile.assert_called_once_with("entity:svc", branch="main")
        assert worker._digests_compiled == 1

    def test_identical_fingerprint_skips_recompile(self) -> None:
        """When fingerprint hasn't changed, compilation is skipped."""
        compiler = MagicMock()
        compiler.compute_scope_fingerprint.return_value = "same_fp"

        worker = self._make_worker(compiler, drift_threshold=0.10)
        worker._anchor_fingerprints["entity:svc@main"] = "same_fp"
        worker._pending = {"entity:svc@main": 0}
        worker._recompile_pending()

        compiler.compile.assert_not_called()
        assert worker._drift_skipped == 1

    def test_changed_fingerprint_triggers_recompile(self) -> None:
        """When fingerprint differs, compilation proceeds."""
        compiler = MagicMock()
        digest = Digest(
            digest_type=DigestType.ENTITY,
            scope="svc",
            summary={"k": "v"},
            entry_count=2,
        )
        compiler.compile.return_value = digest
        compiler.compute_scope_fingerprint.return_value = "new_fp"

        worker = self._make_worker(compiler, drift_threshold=0.10)
        worker._anchor_fingerprints["entity:svc@main"] = "old_fp"
        worker._pending = {"entity:svc@main": 0}
        worker._recompile_pending()

        compiler.compile.assert_called_once()
        assert worker._digests_compiled == 1
        assert worker._anchor_fingerprints["entity:svc@main"] == "new_fp"

    def test_drift_disabled_always_compiles(self) -> None:
        """When drift_threshold=0, the gate is disabled."""
        compiler = MagicMock()
        digest = Digest(
            digest_type=DigestType.ENTITY,
            scope="svc",
            summary={},
            entry_count=1,
        )
        compiler.compile.return_value = digest
        compiler.compute_scope_fingerprint.return_value = "fp"

        worker = self._make_worker(compiler, drift_threshold=0.0)
        worker._anchor_fingerprints["entity:svc@main"] = "fp"
        worker._pending = {"entity:svc@main": 0}
        worker._recompile_pending()

        compiler.compile.assert_called_once()

    def test_stats_include_drift_info(self) -> None:
        compiler = MagicMock()
        worker = self._make_worker(compiler, drift_threshold=0.15)
        worker._started_at = 0
        stats = worker.stats
        assert stats["drift_threshold"] == 0.15
        assert stats["drift_skipped"] == 0
        assert "anchored_scopes" in stats
