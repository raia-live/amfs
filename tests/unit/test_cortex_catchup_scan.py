"""The Cortex catch-up scan asks the store which scopes exist; it does not load the store.

``_catchup_for_current_tenant`` used to call ``adapter.list(branch=...)`` — every current
entry of the tenant, deserialised into MemoryEntry objects — once per instance every
``catchup_interval_s``, then ``list_digests`` for every digest with its summary. On an
80k-entry tenant that was 30-300 s of work per scan, held a transaction the whole time,
and ran fleet-wide every few seconds. The scan now takes the distinct scopes from
``list_scopes`` / ``list_digest_scopes`` when the adapter has them and falls back to the
old reads when it does not, queueing exactly the same missing scopes either way.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from amfs_core.models import Digest, DigestType, MemoryEntry, Provenance


def _worker(adapter):
    from amfs_cortex.compiler import DigestCompiler
    from amfs_cortex.worker import CortexWorker

    compiler = DigestCompiler(adapter=adapter, namespace="ns")
    return CortexWorker(dsn="postgresql://fake", compiler=compiler, use_advisory_lock=False)


def _entry(path: str, agent: str) -> MemoryEntry:
    return MemoryEntry(entity_path=path, key="k", value="v", provenance=Provenance(agent_id=agent, session_id="s", written_at=datetime.now(UTC)))


def test_scan_uses_the_aggregate_reads_and_never_lists_entries() -> None:
    adapter = MagicMock()
    adapter.list_scopes.return_value = ({"acme/support", "acme/billing"},
                                        {"support-agent", "webhook/github", "external/zapier"})
    adapter.list_digest_scopes.return_value = {"entity:acme/support", "agent_brief:support-agent"}

    worker = _worker(adapter)
    queued, n_agents, n_entities = worker._catchup_for_current_tenant()

    adapter.list_scopes.assert_called_once_with(branch="main")
    adapter.list_digest_scopes.assert_called_once_with(namespace="ns", branch="main")
    adapter.list.assert_not_called()
    adapter.list_digests.assert_not_called()
    # Only the scope without a digest is queued; webhook/external authors get no agent brief.
    assert set(worker._pending) == {"entity:acme/billing@main"}
    assert (queued, n_agents, n_entities) == (1, 1, 2)


def test_scan_falls_back_to_listing_on_an_adapter_without_the_aggregates() -> None:
    class Plain:
        def list(self, branch="main"):
            return [_entry("acme/support", "support-agent"), _entry("acme/billing", "webhook/github")]

        def list_digests(self, namespace="default"):
            return [Digest(digest_type=DigestType.ENTITY, scope="acme/support", summary={}, entry_count=1)]

    worker = _worker(Plain())
    queued, n_agents, n_entities = worker._catchup_for_current_tenant()

    assert set(worker._pending) == {"entity:acme/billing@main", "agent:support-agent@main"}
    assert (queued, n_agents, n_entities) == (2, 1, 2)


def test_both_paths_queue_the_same_scopes() -> None:
    """The aggregate is a faster answer to the same question, so the two paths agree."""
    entries = [_entry("a/x", "ag1"), _entry("a/x", "ag2"), _entry("a/y", "ag1"), _entry("b/z", "webhook/w")]
    digests = [Digest(digest_type=DigestType.AGENT_BRIEF, scope="ag1", summary={}, entry_count=1)]

    class Plain:
        def list(self, branch="main"):
            return entries

        def list_digests(self, namespace="default"):
            return digests

    class Aggregating(Plain):
        def list_scopes(self, *, branch="main"):
            return {e.entity_path for e in entries}, {e.provenance.agent_id for e in entries}

        def list_digest_scopes(self, namespace="default", branch="main"):
            return {f"{d.digest_type.value}:{d.scope}" for d in digests}

    slow, fast = _worker(Plain()), _worker(Aggregating())
    assert slow._catchup_for_current_tenant() == fast._catchup_for_current_tenant()
    assert set(slow._pending) == set(fast._pending) == {
        "entity:a/x@main", "entity:a/y@main", "entity:b/z@main", "agent:ag2@main",
    }
