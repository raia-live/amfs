"""Abstract adapter contract tests — ALL adapters must pass these.

Subclass AdapterContractTests, implement the `adapter` fixture, and every
test method here will be inherited and run against your adapter.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, OutcomeRecord, OutcomeType, Provenance


def _make_entry(
    entity_path: str = "checkout-service",
    key: str = "retry-pattern",
    version: int = 1,
    confidence: float = 1.0,
    value: dict | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        version=version,
        value=value or {"pattern": "exponential-backoff", "max_retries": 3},
        confidence=confidence,
        provenance=Provenance(
            agent_id="review-agent",
            session_id="sess-001",
            written_at=datetime.now(timezone.utc),
        ),
    )


class AdapterContractTests:
    """Mixin providing contract tests for any AdapterABC implementation."""

    # Subclasses must provide an `adapter` fixture returning an AdapterABC.

    # ------------------------------------------------------------------
    # read / write basics
    # ------------------------------------------------------------------

    def test_write_and_read(self, adapter: AdapterABC) -> None:
        entry = _make_entry()
        written = adapter.write(entry)
        assert written.version == 1

        result = adapter.read("checkout-service", "retry-pattern")
        assert result is not None
        assert result.version == 1
        assert result.value == entry.value
        assert result.provenance.agent_id == "review-agent"

    def test_read_nonexistent_returns_none(self, adapter: AdapterABC) -> None:
        assert adapter.read("no-such-entity", "no-such-key") is None

    def test_write_increments_version(self, adapter: AdapterABC) -> None:
        e1 = _make_entry()
        w1 = adapter.write(e1)
        assert w1.version == 1

        e2 = _make_entry(value={"updated": True})
        w2 = adapter.write(e2)
        assert w2.version == 2

        current = adapter.read("checkout-service", "retry-pattern")
        assert current is not None
        assert current.version == 2
        assert current.value == {"updated": True}

    # ------------------------------------------------------------------
    # min_confidence filter
    # ------------------------------------------------------------------

    def test_read_min_confidence_filters(self, adapter: AdapterABC) -> None:
        entry = _make_entry(confidence=0.5)
        adapter.write(entry)

        assert adapter.read("checkout-service", "retry-pattern", min_confidence=0.3) is not None
        assert adapter.read("checkout-service", "retry-pattern", min_confidence=0.8) is None

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def test_list_current_only(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(key="key-a"))
        adapter.write(_make_entry(key="key-b"))
        # Write a second version of key-a
        adapter.write(_make_entry(key="key-a", value={"v": 2}))

        entries = adapter.list("checkout-service")
        keys = {e.key for e in entries}
        assert keys == {"key-a", "key-b"}
        # key-a should be version 2
        key_a = next(e for e in entries if e.key == "key-a")
        assert key_a.version == 2

    def test_list_include_superseded(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(key="key-x"))
        adapter.write(_make_entry(key="key-x", value={"v": 2}))

        entries = adapter.list("checkout-service", include_superseded=True)
        versions = sorted(e.version for e in entries if e.key == "key-x")
        assert versions == [1, 2]

    def test_list_all_entities(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(entity_path="svc-a", key="k1"))
        adapter.write(_make_entry(entity_path="svc-b", key="k2"))

        entries = adapter.list()
        entity_paths = {e.entity_path for e in entries}
        assert entity_paths == {"svc-a", "svc-b"}

    # ------------------------------------------------------------------
    # commit_outcome
    # ------------------------------------------------------------------

    def test_commit_outcome_p1_incident(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(confidence=1.0))

        record = OutcomeRecord(
            outcome_ref="INC-001",
            outcome_type=OutcomeType.CRITICAL_FAILURE,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["checkout-service/retry-pattern"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert len(updated) == 1
        # CRITICAL_FAILURE erodes confidence: 1.0 * 0.85 = 0.85
        assert abs(updated[0].confidence - 0.85) < 1e-6
        assert updated[0].outcome_count == 1

    def test_commit_outcome_clean_deploy(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(confidence=0.9))

        record = OutcomeRecord(
            outcome_ref="DEP-001",
            outcome_type=OutcomeType.SUCCESS,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["checkout-service/retry-pattern"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert len(updated) == 1
        # SUCCESS reinforces confidence: 0.9 * 1.03 = 0.927
        assert abs(updated[0].confidence - 0.927) < 1e-6

    def test_commit_outcome_missing_entry_skipped(self, adapter: AdapterABC) -> None:
        record = OutcomeRecord(
            outcome_ref="INC-002",
            outcome_type=OutcomeType.FAILURE,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["nonexistent/key"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert updated == []

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    def test_watch_receives_new_writes(self, adapter: AdapterABC) -> None:
        received: list[MemoryEntry] = []
        handle = adapter.watch("checkout-service", received.append)

        try:
            # Give watcher time to start (FSEvents on macOS can be slow)
            time.sleep(0.5)
            adapter.write(_make_entry())
            # Poll for the event with a generous timeout
            deadline = time.monotonic() + 3.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.1)
            assert len(received) >= 1
            assert received[0].key == "retry-pattern"
        finally:
            handle.cancel()
            assert handle.cancelled

    def test_watch_cancel_stops_notifications(self, adapter: AdapterABC) -> None:
        received: list[MemoryEntry] = []
        handle = adapter.watch("checkout-service", received.append)
        time.sleep(0.2)
        handle.cancel()

        adapter.write(_make_entry())
        time.sleep(0.5)
        # After cancel, should not receive new entries
        # (may have received 0 or some depending on timing, but handle is cancelled)
        assert handle.cancelled

    # ------------------------------------------------------------------
    # commits
    # ------------------------------------------------------------------
    #
    # Added after the connector functional sweep found there was no way to
    # obtain two commit ids to give amfs_merge_base. The cause was that
    # save_commit, get_commit and list_commits are concrete on AdapterABC with
    # placeholder bodies — a no-op, None, and [] — so an adapter that never
    # implemented them still satisfied the interface. Postgres never did, which
    # means every hosted account's commit log had been empty since the
    # beginning while reading as an honest answer about a history nobody had
    # written yet.
    #
    # Nothing in this suite asked. That is the actual defect: a contract that
    # tests only the methods an adapter chose to write cannot tell "not
    # implemented" from "nothing there". These four ask.

    @staticmethod
    def _make_commit(commit_id: str, *parents: str, message: str = "batch"):
        from amfs_core.models import Commit, CommitEntry

        return Commit(
            id=commit_id,
            message=message,
            author_agent_id="review-agent",
            session_id="sess-001",
            entries=[
                CommitEntry(entity_path="checkout-service", key="retry-pattern",
                            version=1, content_hash="h1"),
            ],
            tree_hash=f"tree-{commit_id}",
            parent_ids=list(parents),
        )

    def test_a_saved_commit_can_be_read_back(self, adapter: AdapterABC) -> None:
        adapter.save_commit(self._make_commit("c-round-trip"))

        got = adapter.get_commit("c-round-trip")

        assert got is not None, (
            "save_commit accepted the commit and get_commit cannot find it — "
            "the placeholder no-op is still in place"
        )
        assert got.id == "c-round-trip"
        assert got.author_agent_id == "review-agent"
        assert got.tree_hash == "tree-c-round-trip"

    def test_the_entries_survive_the_round_trip(self, adapter: AdapterABC) -> None:
        """A commit whose entries are lost cannot say what it committed."""
        adapter.save_commit(self._make_commit("c-entries"))

        got = adapter.get_commit("c-entries")

        assert got is not None
        assert len(got.entries) == 1
        assert got.entries[0].entity_path == "checkout-service"
        assert got.entries[0].key == "retry-pattern"

    def test_parents_survive_so_the_history_is_a_chain(self, adapter: AdapterABC) -> None:
        """Without parents every commit is a root and no pair has an ancestor.

        This is the property common_ancestor walks, and losing it is invisible:
        the answer "these two share no ancestor" is a perfectly plausible thing
        for a repository to say.
        """
        adapter.save_commit(self._make_commit("c-parent"))
        adapter.save_commit(self._make_commit("c-child", "c-parent"))

        got = adapter.get_commit("c-child")

        assert got is not None
        assert got.parent_ids == ["c-parent"]

    def test_list_commits_returns_what_was_saved_newest_first(
        self, adapter: AdapterABC
    ) -> None:
        adapter.save_commit(self._make_commit("c-one"))
        adapter.save_commit(self._make_commit("c-two", "c-one"))

        listed = adapter.list_commits(limit=10)

        ids = [c.id for c in listed]
        assert "c-one" in ids and "c-two" in ids, (
            f"list_commits returned {ids} after two saves"
        )
        assert ids.index("c-two") < ids.index("c-one"), "newest first"

    def test_an_unknown_commit_is_none_rather_than_an_error(
        self, adapter: AdapterABC
    ) -> None:
        assert adapter.get_commit("c-never-written") is None

    def test_saving_the_same_commit_twice_does_not_duplicate_it(
        self, adapter: AdapterABC
    ) -> None:
        """The id is a content hash, so a retried flush is the same commit.

        Without this an interrupted transaction that is retried leaves two rows
        claiming to be one commit, and the log shows work that happened once as
        having happened twice.
        """
        adapter.save_commit(self._make_commit("c-idempotent"))
        adapter.save_commit(self._make_commit("c-idempotent"))

        listed = [c for c in adapter.list_commits(limit=50) if c.id == "c-idempotent"]

        assert len(listed) == 1
