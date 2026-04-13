"""Tests for amfs_core.hashing — content hashing, integrity chains, verification."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from amfs_core.hashing import (
    IntegrityReport,
    content_hash,
    integrity_chain_hash,
    tree_hash,
    verify_entries,
    verify_entry,
)
from amfs_core.models import MemoryEntry, MemoryType, Provenance


def _make_entry(
    entity_path: str = "test/svc",
    key: str = "k1",
    version: int = 1,
    value: dict | str = "hello",
    content_hash_val: str | None = None,
    integrity_chain_val: str | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        version=version,
        value=value,
        provenance=Provenance(
            agent_id="test-agent",
            session_id="sess-1",
            written_at=datetime.now(timezone.utc),
        ),
        content_hash=content_hash_val,
        integrity_chain=integrity_chain_val,
    )


class TestContentHash:
    def test_deterministic(self):
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_different_values_different_hashes(self):
        assert content_hash("hello") != content_hash("world")

    def test_string_value(self):
        h = content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_nested_dict(self):
        val = {"nested": {"deep": [1, 2, 3]}}
        h = content_hash(val)
        assert len(h) == 64


class TestTreeHash:
    def test_sorted_order(self):
        hashes = ["aaa", "zzz", "bbb"]
        result = tree_hash(hashes)
        assert result == tree_hash(["zzz", "aaa", "bbb"])

    def test_different_sets_different_hashes(self):
        assert tree_hash(["a", "b"]) != tree_hash(["a", "c"])


class TestIntegrityChainHash:
    def test_no_previous(self):
        h = integrity_chain_hash("abc123", None)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_with_previous(self):
        h1 = integrity_chain_hash("abc", None)
        h2 = integrity_chain_hash("def", h1)
        assert h1 != h2

    def test_deterministic(self):
        a = integrity_chain_hash("abc", "prev")
        b = integrity_chain_hash("abc", "prev")
        assert a == b


class TestVerifyEntry:
    def test_legacy_entry_no_hash(self):
        entry = _make_entry()
        assert verify_entry(entry) is True

    def test_valid_hash(self):
        value = {"key": "value"}
        h = content_hash(value)
        entry = _make_entry(value=value, content_hash_val=h)
        assert verify_entry(entry) is True

    def test_corrupted_hash(self):
        entry = _make_entry(value="hello", content_hash_val="bad_hash")
        assert verify_entry(entry) is False


class TestVerifyEntries:
    def test_empty_list(self):
        report = verify_entries([])
        assert report.total_checked == 0
        assert report.valid == 0
        assert report.is_clean

    def test_legacy_entries(self):
        entries = [_make_entry(version=1), _make_entry(version=2)]
        report = verify_entries(entries)
        assert report.total_checked == 2
        assert report.valid == 2
        assert report.is_clean

    def test_valid_chain(self):
        val1 = "first"
        h1 = content_hash(val1)
        c1 = integrity_chain_hash(h1, None)

        val2 = "second"
        h2 = content_hash(val2)
        c2 = integrity_chain_hash(h2, c1)

        entries = [
            _make_entry(version=1, value=val1, content_hash_val=h1, integrity_chain_val=c1),
            _make_entry(version=2, value=val2, content_hash_val=h2, integrity_chain_val=c2),
        ]
        report = verify_entries(entries)
        assert report.total_checked == 2
        assert report.valid == 2
        assert report.is_clean

    def test_corrupted_entry(self):
        entries = [
            _make_entry(version=1, value="hello", content_hash_val="wrong_hash"),
        ]
        report = verify_entries(entries)
        assert len(report.corrupted) == 1
        assert report.corrupted[0]["version"] == 1
        assert not report.is_clean

    def test_broken_chain(self):
        val1 = "first"
        h1 = content_hash(val1)
        c1 = integrity_chain_hash(h1, None)

        val2 = "second"
        h2 = content_hash(val2)
        bad_chain = "not_a_valid_chain"

        entries = [
            _make_entry(version=1, value=val1, content_hash_val=h1, integrity_chain_val=c1),
            _make_entry(version=2, value=val2, content_hash_val=h2, integrity_chain_val=bad_chain),
        ]
        report = verify_entries(entries)
        assert len(report.chain_breaks) == 1
        assert not report.is_clean
