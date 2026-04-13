"""Tests for memory contract validation."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.contracts import (
    find_matching_contracts,
    validate_against_contract,
    validate_entry,
)
from amfs_core.models import MemoryContract, MemoryEntry, Provenance


def _make_entry(
    entity_path: str = "repo/svc",
    key: str = "k1",
    value=None,
    confidence: float = 1.0,
    ttl_at=None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value if value is not None else {"data": 1},
        provenance=Provenance(
            agent_id="test", session_id="s",
            written_at=datetime.now(timezone.utc),
        ),
        confidence=confidence,
        ttl_at=ttl_at,
    )


class TestFindMatchingContracts:
    def test_exact_match(self):
        c = MemoryContract(entity_path="repo/svc", key_pattern="config-*")
        matches = find_matching_contracts([c], "repo/svc", "config-db")
        assert len(matches) == 1

    def test_wildcard_key(self):
        c = MemoryContract(entity_path="repo/svc", key_pattern="*")
        matches = find_matching_contracts([c], "repo/svc", "anything")
        assert len(matches) == 1

    def test_no_match(self):
        c = MemoryContract(entity_path="repo/other", key_pattern="*")
        matches = find_matching_contracts([c], "repo/svc", "k1")
        assert len(matches) == 0

    def test_prefix_match(self):
        c = MemoryContract(entity_path="repo/svc", key_pattern="risk-*")
        matches = find_matching_contracts([c], "repo/svc/sub", "risk-high")
        assert len(matches) == 1

    def test_key_pattern_no_match(self):
        c = MemoryContract(entity_path="repo/svc", key_pattern="config-*")
        matches = find_matching_contracts([c], "repo/svc", "risk-high")
        assert len(matches) == 0


class TestValidateAgainstContract:
    def test_confidence_too_low(self):
        c = MemoryContract(entity_path="repo/svc", min_confidence=0.8)
        entry = _make_entry(confidence=0.5)
        violations = validate_against_contract(entry, c)
        assert len(violations) == 1
        assert "below minimum" in violations[0].message

    def test_confidence_too_high(self):
        c = MemoryContract(entity_path="repo/svc", max_confidence=0.9)
        entry = _make_entry(confidence=1.0)
        violations = validate_against_contract(entry, c)
        assert len(violations) == 1
        assert "above maximum" in violations[0].message

    def test_ttl_required_missing(self):
        c = MemoryContract(entity_path="repo/svc", ttl_required=True)
        entry = _make_entry()
        violations = validate_against_contract(entry, c)
        assert len(violations) == 1
        assert "TTL" in violations[0].message

    def test_ttl_required_present(self):
        c = MemoryContract(entity_path="repo/svc", ttl_required=True)
        entry = _make_entry(ttl_at=datetime(2099, 1, 1, tzinfo=timezone.utc))
        violations = validate_against_contract(entry, c)
        assert len(violations) == 0

    def test_required_fields_missing(self):
        c = MemoryContract(entity_path="repo/svc", required_fields=["name", "version"])
        entry = _make_entry(value={"name": "test"})
        violations = validate_against_contract(entry, c)
        assert len(violations) == 1
        assert "version" in violations[0].message

    def test_required_fields_present(self):
        c = MemoryContract(entity_path="repo/svc", required_fields=["name"])
        entry = _make_entry(value={"name": "test"})
        violations = validate_against_contract(entry, c)
        assert len(violations) == 0

    def test_valid_entry(self):
        c = MemoryContract(entity_path="repo/svc", min_confidence=0.5)
        entry = _make_entry(confidence=0.9)
        violations = validate_against_contract(entry, c)
        assert len(violations) == 0


class TestValidateEntry:
    def test_no_matching_contracts(self):
        contracts = [MemoryContract(entity_path="other/svc")]
        entry = _make_entry(entity_path="repo/svc")
        violations = validate_entry(entry, contracts)
        assert len(violations) == 0

    def test_multiple_violations(self):
        contracts = [
            MemoryContract(entity_path="repo/svc", min_confidence=0.9, ttl_required=True),
        ]
        entry = _make_entry(confidence=0.5)
        violations = validate_entry(entry, contracts)
        assert len(violations) == 2
