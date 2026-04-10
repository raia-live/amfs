"""Write-time quality evaluator — feedback to improve memory quality.

Extension point for evaluating memory quality at write time.  The default
``HeuristicQualityEvaluator`` applies lightweight rule-based checks and
produces a :class:`QualityReport` with actionable suggestions.  The Pro
layer can substitute an LLM-based evaluator for deeper analysis.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from amfs_core.models import QualityIssue, QualityReport

_RATIONALE_WORDS = frozenset({
    "because", "reason", "hypothesis", "rationale", "evidence",
    "theory", "assumption", "likely", "suspect", "believe",
    "since", "due to", "caused by", "suggests", "indicates",
})


class MemoryQualityEvaluator(ABC):
    """Interface for evaluating memory quality at write time."""

    @abstractmethod
    def evaluate(
        self,
        value: Any,
        *,
        entity_path: str,
        key: str,
        confidence: float = 1.0,
        memory_type: str = "fact",
        pattern_refs: list[str] | None = None,
        existing_keys: list[str] | None = None,
    ) -> QualityReport:
        """Evaluate quality and return a report with score and issues."""


class NoOpQualityEvaluator(MemoryQualityEvaluator):
    """Returns a perfect score with no issues (zero overhead)."""

    def evaluate(
        self,
        value: Any,
        *,
        entity_path: str,
        key: str,
        confidence: float = 1.0,
        memory_type: str = "fact",
        pattern_refs: list[str] | None = None,
        existing_keys: list[str] | None = None,
    ) -> QualityReport:
        return QualityReport(score=1.0, action="stored_ok", issues=[])


class HeuristicQualityEvaluator(MemoryQualityEvaluator):
    """Rule-based quality checks that run at write time.

    Each check can subtract from a starting score of 1.0 and append a
    :class:`QualityIssue`.  The evaluator is stateless — all context
    (existing sibling keys, etc.) is passed in by the caller.
    """

    def evaluate(
        self,
        value: Any,
        *,
        entity_path: str,
        key: str,
        confidence: float = 1.0,
        memory_type: str = "fact",
        pattern_refs: list[str] | None = None,
        existing_keys: list[str] | None = None,
    ) -> QualityReport:
        issues: list[QualityIssue] = []
        score = 1.0

        str_value = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
        is_structured = isinstance(value, (dict, list))
        refs = pattern_refs or []
        mt = memory_type.lower() if isinstance(memory_type, str) else str(memory_type).lower()
        siblings = existing_keys or []

        score -= self._check_too_short(str_value, issues)
        score -= self._check_unstructured(str_value, is_structured, issues)
        score -= self._check_missing_pattern_refs(refs, siblings, issues)
        score -= self._check_belief_no_rationale(mt, str_value, issues)
        score -= self._check_overconfident_belief(mt, confidence, issues)

        score = max(score, 0.0)
        action = "stored_ok" if score >= 0.8 else "stored_with_suggestions"
        return QualityReport(score=round(score, 2), action=action, issues=issues)

    # ------------------------------------------------------------------
    # Individual checks — each returns the penalty (0.0 if not triggered)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_too_short(str_value: str, issues: list[QualityIssue]) -> float:
        if len(str_value) < 50:
            issues.append(QualityIssue(
                type="too_short",
                message=(
                    f"Value is only {len(str_value)} characters. Memories under "
                    "~50 chars rarely contain enough detail to be useful to "
                    "future agents."
                ),
                suggestion=(
                    "Rewrite with specific details: what was done, why, and "
                    "key parameters or decisions."
                ),
            ))
            return 0.3
        return 0.0

    @staticmethod
    def _check_unstructured(
        str_value: str,
        is_structured: bool,
        issues: list[QualityIssue],
    ) -> float:
        if not is_structured and len(str_value) > 50:
            issues.append(QualityIssue(
                type="unstructured",
                message=(
                    "Value is a plain string. Structured data (JSON objects) "
                    "is easier for other agents to parse and act on."
                ),
                suggestion=(
                    "Consider using a JSON object with named fields instead "
                    "of a free-text string."
                ),
            ))
            return 0.1
        return 0.0

    @staticmethod
    def _check_missing_pattern_refs(
        refs: list[str],
        siblings: list[str],
        issues: list[QualityIssue],
    ) -> float:
        if not refs and siblings:
            shown = siblings[:5]
            issues.append(QualityIssue(
                type="missing_pattern_refs",
                message=(
                    f"Found {len(siblings)} related entries under this entity: "
                    f"{', '.join(repr(k) for k in shown)}"
                    + ("..." if len(siblings) > 5 else "")
                    + "."
                ),
                suggestion=(
                    "Consider calling amfs_write again with pattern_refs to "
                    "link related knowledge."
                ),
            ))
            return 0.15
        return 0.0

    @staticmethod
    def _check_belief_no_rationale(
        mt: str,
        str_value: str,
        issues: list[QualityIssue],
    ) -> float:
        if mt != "belief":
            return 0.0
        lower_value = str_value.lower()
        if any(word in lower_value for word in _RATIONALE_WORDS):
            return 0.0
        issues.append(QualityIssue(
            type="belief_no_rationale",
            message=(
                "This belief does not include a rationale. Explaining why "
                "you believe something helps future agents evaluate it."
            ),
            suggestion=(
                "Add reasoning — e.g. 'because ...', 'hypothesis: ...', "
                "'evidence suggests ...'."
            ),
        ))
        return 0.1

    @staticmethod
    def _check_overconfident_belief(
        mt: str,
        confidence: float,
        issues: list[QualityIssue],
    ) -> float:
        if mt != "belief" or confidence <= 0.9:
            return 0.0
        issues.append(QualityIssue(
            type="overconfident_belief",
            message=(
                f"Confidence is {confidence} for a belief. Beliefs are "
                "subjective and should usually have confidence < 0.9."
            ),
            suggestion="Lower the confidence to reflect the subjective nature of this belief.",
        ))
        return 0.1
