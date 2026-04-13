"""Content hashing, integrity chains, and verification for AMFS entries."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from amfs_core.models import MemoryEntry


def _canonical_json(value: Any) -> str:
    """Produce a deterministic JSON string for hashing.

    Uses sorted keys, no whitespace, and ensures_ascii for byte-level
    reproducibility across platforms and Python versions.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def content_hash(value: Any) -> str:
    """SHA-256 hash of the canonical JSON representation of a value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def tree_hash(entry_hashes: list[str]) -> str:
    """Hash of sorted entry content hashes — represents the state of a commit."""
    combined = "\n".join(sorted(entry_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def integrity_chain_hash(current_content_hash: str, previous_chain: str | None) -> str:
    """Chain link connecting the current entry to its predecessor.

    Each version's integrity_chain = SHA-256(content_hash + previous_integrity_chain).
    For the first version (no predecessor), previous_chain is treated as empty string.
    """
    payload = current_content_hash + (previous_chain or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_entry(entry: MemoryEntry) -> bool:
    """Verify that an entry's content_hash matches its value.

    Returns True if the hash is valid or if no hash was stored (legacy entry).
    Returns False only if a hash exists and does not match.
    """
    if entry.content_hash is None:
        return True
    expected = content_hash(entry.value)
    return expected == entry.content_hash


class IntegrityReport:
    """Result of verifying entries in a memory store."""

    __slots__ = ("total_checked", "valid", "corrupted", "orphaned", "chain_breaks")

    def __init__(self) -> None:
        self.total_checked: int = 0
        self.valid: int = 0
        self.corrupted: list[dict] = []
        self.orphaned: list[dict] = []
        self.chain_breaks: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "valid": self.valid,
            "corrupted": self.corrupted,
            "orphaned": self.orphaned,
            "chain_breaks": self.chain_breaks,
        }

    @property
    def is_clean(self) -> bool:
        return not self.corrupted and not self.chain_breaks


def verify_entries(entries: list[MemoryEntry]) -> IntegrityReport:
    """Verify a batch of entries for content integrity and chain continuity.

    Groups entries by (entity_path, key), sorts by version, and checks:
    1. content_hash matches recomputed hash from value
    2. integrity_chain links correctly to the previous version's content_hash
    """
    report = IntegrityReport()

    lineages: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        lineage_key = f"{entry.entity_path}/{entry.key}"
        lineages.setdefault(lineage_key, []).append(entry)

    for lineage_key, versions in lineages.items():
        versions.sort(key=lambda e: e.version)
        previous_chain: str | None = None

        for entry in versions:
            report.total_checked += 1

            if entry.content_hash is None:
                report.valid += 1
                previous_chain = None
                continue

            expected_hash = content_hash(entry.value)
            if expected_hash != entry.content_hash:
                report.corrupted.append({
                    "entity_path": entry.entity_path,
                    "key": entry.key,
                    "version": entry.version,
                    "expected_hash": expected_hash,
                    "actual_hash": entry.content_hash,
                })
                previous_chain = entry.integrity_chain
                continue

            if entry.integrity_chain is not None and previous_chain is not None:
                expected_chain = integrity_chain_hash(entry.content_hash, previous_chain)
                if expected_chain != entry.integrity_chain:
                    report.chain_breaks.append({
                        "entity_path": entry.entity_path,
                        "key": entry.key,
                        "version": entry.version,
                        "expected_chain": expected_chain,
                        "actual_chain": entry.integrity_chain,
                    })

            report.valid += 1
            previous_chain = entry.integrity_chain

    return report
