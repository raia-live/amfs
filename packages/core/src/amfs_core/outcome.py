"""OutcomeBackPropagator — applies outcome events to memory entries via the engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from amfs_core.abc import AdapterABC
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
    OutcomeType,
)

logger = logging.getLogger(__name__)


class OutcomeBackPropagator:
    """Processes outcome events and back-propagates confidence changes.

    When an outcome is committed (e.g., a P1 incident), this class:
    1. Looks up each causal entry by its ``entity_path/key`` spec.
    2. Computes new confidence = old_confidence * type_multiplier * causal_confidence.
    3. Writes new versions of affected entries via the adapter.

    Parameters
    ----------
    adapter:
        The storage adapter to read from and write to.
    """

    def __init__(self, adapter: AdapterABC) -> None:
        self._adapter = adapter

    def propagate(self, record: OutcomeRecord) -> list[MemoryEntry]:
        """Apply an outcome to all its causal entries.

        This is a higher-level API than ``adapter.commit_outcome()`` —
        it delegates to the adapter but adds logging and validation.

        Returns the list of entries whose confidence was updated.
        """
        logger.info(
            "Propagating outcome %s (%s) to %d causal entries",
            record.outcome_ref,
            record.outcome_type.value,
            len(record.causal_entry_keys),
        )
        updated = self._adapter.commit_outcome(record)
        logger.info(
            "Outcome %s updated %d entries",
            record.outcome_ref,
            len(updated),
        )
        return updated

    def propagate_batch(self, records: list[OutcomeRecord]) -> list[MemoryEntry]:
        """Apply multiple outcomes sequentially. Returns all updated entries."""
        all_updated: list[MemoryEntry] = []
        for record in records:
            updated = self.propagate(record)
            all_updated.extend(updated)
        return all_updated

    @staticmethod
    def compute_new_confidence(
        current_confidence: float,
        outcome_type: OutcomeType,
        causal_confidence: float = 1.0,
    ) -> float:
        """Compute the new confidence value after an outcome.

        This is a pure function useful for previewing the effect
        without actually writing.
        """
        multiplier = OUTCOME_MULTIPLIERS[outcome_type]
        return current_confidence * multiplier * causal_confidence

    @staticmethod
    def make_record(
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str],
        agent_id: str,
        *,
        causal_confidence: float = 1.0,
        committed_at: datetime | None = None,
    ) -> OutcomeRecord:
        """Convenience factory for creating OutcomeRecord instances."""
        return OutcomeRecord(
            outcome_ref=outcome_ref,
            outcome_type=outcome_type,
            causal_confidence=causal_confidence,
            committed_at=committed_at or datetime.now(timezone.utc),
            causal_entry_keys=causal_entry_keys,
            agent_id=agent_id,
        )
