"""Importance evaluator — optional hook for multi-dimensional memory scoring.

Provides an extension point for evaluating the importance of a memory
entry at write time.  The OSS default is a no-op; the Pro layer can
plug in an LLM-based evaluator that scores entries across dimensions
like behavioral alignment, reasoning utility, and contextual persistence
(inspired by the HMO paper's scoring framework).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ImportanceEvaluator(ABC):
    """Interface for scoring memory importance at write time."""

    @abstractmethod
    def evaluate(
        self,
        value: Any,
        *,
        entity_path: str,
        key: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[float | None, dict[str, float]]:
        """Evaluate the importance of a memory value.

        Returns:
            A tuple of (overall_score, dimension_breakdown).
            overall_score is 0.0-1.0 or None to skip scoring.
            dimension_breakdown maps dimension names to individual scores.
        """


class NoOpEvaluator(ImportanceEvaluator):
    """Default evaluator that performs no scoring (zero overhead)."""

    def evaluate(
        self,
        value: Any,
        *,
        entity_path: str,
        key: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[float | None, dict[str, float]]:
        return None, {}
