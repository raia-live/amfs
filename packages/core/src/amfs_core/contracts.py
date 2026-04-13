"""Memory contract validation for agent-memory binding."""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from amfs_core.models import MemoryContract, MemoryEntry

logger = logging.getLogger(__name__)


class ContractViolation:
    """A single contract validation failure."""

    __slots__ = ("contract", "field", "message")

    def __init__(self, contract: MemoryContract, field: str, message: str) -> None:
        self.contract = contract
        self.field = field
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_path": self.contract.entity_path,
            "key_pattern": self.contract.key_pattern,
            "field": self.field,
            "message": self.message,
        }


def find_matching_contracts(
    contracts: list[MemoryContract],
    entity_path: str,
    key: str,
) -> list[MemoryContract]:
    """Return all contracts whose entity_path and key_pattern match."""
    matches = []
    for c in contracts:
        if c.entity_path != entity_path and c.entity_path != "*":
            if not entity_path.startswith(c.entity_path + "/"):
                continue
        if c.key_pattern != "*" and not fnmatch.fnmatch(key, c.key_pattern):
            continue
        matches.append(c)
    return matches


def validate_against_contract(
    entry: MemoryEntry,
    contract: MemoryContract,
) -> list[ContractViolation]:
    """Validate a memory entry against a single contract."""
    violations: list[ContractViolation] = []

    if entry.confidence < contract.min_confidence:
        violations.append(ContractViolation(
            contract, "confidence",
            f"Confidence {entry.confidence} below minimum {contract.min_confidence}",
        ))

    if entry.confidence > contract.max_confidence:
        violations.append(ContractViolation(
            contract, "confidence",
            f"Confidence {entry.confidence} above maximum {contract.max_confidence}",
        ))

    if contract.ttl_required and entry.ttl_at is None:
        violations.append(ContractViolation(
            contract, "ttl_at",
            "TTL is required by contract but not set",
        ))

    if contract.required_fields and isinstance(entry.value, dict):
        for field in contract.required_fields:
            if field not in entry.value:
                violations.append(ContractViolation(
                    contract, f"value.{field}",
                    f"Required field '{field}' missing from value",
                ))

    return violations


def validate_entry(
    entry: MemoryEntry,
    contracts: list[MemoryContract],
) -> list[ContractViolation]:
    """Validate entry against all matching contracts."""
    matching = find_matching_contracts(contracts, entry.entity_path, entry.key)
    violations: list[ContractViolation] = []
    for contract in matching:
        violations.extend(validate_against_contract(entry, contract))
    return violations
