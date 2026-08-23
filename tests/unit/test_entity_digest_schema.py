"""The entity digest carries a schema profile + materialized aggregates for
record-shaped (room/tabular) entities, and leaves plain-text entities alone.

This is what makes a room briefing a one-call orientation: the digest served by
/api/v1/briefing already contains the shape of the data and precomputed numeric
rollups, so an agent need not list every record.
"""

from __future__ import annotations

import types
from datetime import datetime, timezone

from amfs_cortex.strategies import RuleBasedStrategy, _structured_profile


def _e(value, key, conf=0.9):
    return types.SimpleNamespace(
        value=value,
        key=key,
        confidence=conf,
        outcome_count=0,
        provenance=types.SimpleNamespace(
            agent_id="a1", written_at=datetime.now(timezone.utc)
        ),
    )


def test_plain_text_entities_get_no_profile():
    prof, agg = _structured_profile([_e("just a note", "n1"), _e("another", "n2")])
    assert prof is None
    assert agg is None


def test_record_entities_get_profile_and_aggregates():
    entries = [_e({"listings": [{"price": 100, "city": "A"}, {"price": 300, "city": "B"}]}, "b1")]
    prof, agg = _structured_profile(entries)
    assert prof is not None
    assert agg["total_rows"] == 2
    assert agg["row_path"] == "listings"
    assert "price" in agg["numeric_fields"]
    assert agg["numeric_fields"]["price"]["mean"] == 200


class _Adapter:
    def __init__(self, entries):
        self._entries = entries

    def search(self, query, branch="main"):
        return list(self._entries)


def test_compile_entity_carries_schema_profile():
    entries = [_e({"listings": [{"price": 100, "city": "A"}]}, "b1")]
    digest = RuleBasedStrategy().compile_entity("@room/listings", _Adapter(entries), "default")
    assert digest is not None
    assert "schema_profile" in digest.summary
    assert "materialized_aggregates" in digest.summary


def test_compile_entity_plain_text_has_no_schema_profile():
    entries = [_e("a runbook note", "runbook")]
    digest = RuleBasedStrategy().compile_entity("ops/notes", _Adapter(entries), "default")
    assert digest is not None
    assert "schema_profile" not in digest.summary


def test_small_entity_profile_is_not_flagged_sampled():
    entries = [_e({"listings": [{"price": 100, "city": "A"}]}, "b1")]
    digest = RuleBasedStrategy().compile_entity("@room/listings", _Adapter(entries), "default")
    assert "sampled" not in digest.summary["schema_profile"]
    assert "sampled" not in digest.summary["materialized_aggregates"]


def test_large_entity_profile_is_flagged_as_a_sample(monkeypatch):
    """When the entity has as many keys as the digest cap, the rollups are a
    top-confidence sample and must say so — otherwise they read as the totals
    and disagree with a full amfs_aggregate."""
    import amfs_cortex.strategies as strategies

    monkeypatch.setattr(strategies, "_ENTITY_DIGEST_LIMIT", 3)
    entries = [
        _e({"price": 100}, "k1"),
        _e({"price": 200}, "k2"),
        _e({"price": 300}, "k3"),
    ]
    digest = strategies.RuleBasedStrategy().compile_entity(
        "@room/listings", _Adapter(entries), "default",
    )
    prof = digest.summary["schema_profile"]
    agg = digest.summary["materialized_aggregates"]
    assert prof["sampled"] is True
    assert prof["sample_size"] == 3
    assert "amfs_aggregate" in prof["note"]
    assert agg["sampled"] is True
