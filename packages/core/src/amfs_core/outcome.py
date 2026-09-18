"""OutcomeBackPropagator — applies outcome events to memory entries via the engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from amfs_core.abc import AdapterABC
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
    OutcomeType,
    clamp_confidence,
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
        *,
        n_causal: int = 1,
        evidence_success: float = 0.0,
        evidence_failure: float = 0.0,
        prior_confidence: float | None = None,
    ) -> float:
        """Preview the confidence an outcome would leave, without writing.

        Under the default evidence model (see ``amfs_core.evidence``) the
        result depends on the entry's accumulated evidence, so callers that
        have it should pass it; with none, this is the update a fresh entry at
        ``current_confidence`` receives. ``AMFS_OUTCOME_MODEL=multiplicative``
        restores ``current * multiplier * causal_confidence``.
        """
        from amfs_core import evidence as ev

        if ev.outcome_model() == "multiplicative":
            multiplier = OUTCOME_MULTIPLIERS[outcome_type]
            return clamp_confidence(current_confidence * multiplier * causal_confidence)
        prior = current_confidence if prior_confidence is None else prior_confidence
        w = ev.evidence_weight(
            outcome_type,
            current_confidence=current_confidence,
            causal_confidence=causal_confidence,
            n_causal=n_causal,
        )
        if ev.is_success(outcome_type):
            e_s, e_f = evidence_success * ev.EVIDENCE_DECAY + w, evidence_failure * ev.EVIDENCE_DECAY
        else:
            e_s, e_f = evidence_success * ev.EVIDENCE_DECAY, evidence_failure * ev.EVIDENCE_DECAY + w
        return ev.posterior(prior, e_s, e_f)

    @staticmethod
    def make_record(
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str],
        agent_id: str,
        *,
        causal_confidence: float = 1.0,
        committed_at: datetime | None = None,
        task_input: str | None = None,
        response_text: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        session_metadata: dict[str, Any] | None = None,
        trace_follows: bool = False,
        attempts: list[Any] | None = None,
        final_action_index: int | None = None,
        causal_entry_versions: dict[str, int] | None = None,
        actions_taken: list[dict[str, Any]] | None = None,
        entity_paths: list[str] | None = None,
        situation: str | None = None,
    ) -> OutcomeRecord:
        """Convenience factory for creating OutcomeRecord instances."""
        return OutcomeRecord(
            attempts=attempts or [],
            final_action_index=final_action_index,
            causal_entry_versions=dict(causal_entry_versions or {}),
            actions_taken=list(actions_taken or []),
            entity_paths=list(entity_paths or []),
            situation=situation,
            outcome_ref=outcome_ref,
            outcome_type=outcome_type,
            causal_confidence=causal_confidence,
            committed_at=committed_at or datetime.now(timezone.utc),
            causal_entry_keys=causal_entry_keys,
            agent_id=agent_id,
            task_input=task_input,
            response_text=response_text,
            tool_calls=tool_calls or [],
            session_metadata=session_metadata or None,
            trace_follows=trace_follows,
        )
