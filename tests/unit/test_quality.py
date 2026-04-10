"""Unit tests for the write-time quality evaluator."""

from __future__ import annotations

from amfs_core.models import QualityIssue, QualityReport
from amfs_core.quality import (
    HeuristicQualityEvaluator,
    MemoryQualityEvaluator,
    NoOpQualityEvaluator,
)


class TestNoOpQualityEvaluator:
    def test_returns_perfect_score(self) -> None:
        evaluator = NoOpQualityEvaluator()
        report = evaluator.evaluate("anything", entity_path="svc", key="k")
        assert report.score == 1.0
        assert report.action == "stored_ok"
        assert report.issues == []

    def test_is_quality_evaluator(self) -> None:
        evaluator = NoOpQualityEvaluator()
        assert isinstance(evaluator, MemoryQualityEvaluator)


class TestHeuristicQualityEvaluator:
    def setup_method(self) -> None:
        self.evaluator = HeuristicQualityEvaluator()

    def test_short_value_triggers_too_short(self) -> None:
        report = self.evaluator.evaluate("short", entity_path="svc", key="k")
        types = [i.type for i in report.issues]
        assert "too_short" in types
        assert report.score < 0.8

    def test_unstructured_long_string_triggers_unstructured(self) -> None:
        long_text = "This is a fairly long plain text value that exceeds fifty characters and has no structure"
        report = self.evaluator.evaluate(long_text, entity_path="svc", key="k")
        types = [i.type for i in report.issues]
        assert "unstructured" in types

    def test_structured_value_no_unstructured_issue(self) -> None:
        structured = {"max_retries": 3, "backoff": "exponential"}
        report = self.evaluator.evaluate(structured, entity_path="svc", key="k")
        types = [i.type for i in report.issues]
        assert "unstructured" not in types

    def test_missing_pattern_refs_with_siblings(self) -> None:
        report = self.evaluator.evaluate(
            {"detail": "some value with enough content to be useful here"},
            entity_path="svc",
            key="k",
            existing_keys=["other-key", "another-key"],
        )
        types = [i.type for i in report.issues]
        assert "missing_pattern_refs" in types

    def test_no_missing_pattern_refs_when_refs_provided(self) -> None:
        report = self.evaluator.evaluate(
            {"detail": "some value with enough content to be useful here"},
            entity_path="svc",
            key="k",
            pattern_refs=["other-key"],
            existing_keys=["other-key"],
        )
        types = [i.type for i in report.issues]
        assert "missing_pattern_refs" not in types

    def test_no_missing_pattern_refs_when_no_siblings(self) -> None:
        report = self.evaluator.evaluate(
            {"detail": "some value with enough content to be useful here"},
            entity_path="svc",
            key="k",
        )
        types = [i.type for i in report.issues]
        assert "missing_pattern_refs" not in types

    def test_belief_without_rationale(self) -> None:
        report = self.evaluator.evaluate(
            "The service will probably fail under load and we should fix it soon",
            entity_path="svc",
            key="k",
            memory_type="belief",
        )
        types = [i.type for i in report.issues]
        assert "belief_no_rationale" in types

    def test_belief_with_rationale(self) -> None:
        report = self.evaluator.evaluate(
            "The service will fail under load because the connection pool is too small",
            entity_path="svc",
            key="k",
            memory_type="belief",
        )
        types = [i.type for i in report.issues]
        assert "belief_no_rationale" not in types

    def test_overconfident_belief(self) -> None:
        report = self.evaluator.evaluate(
            "I suspect this because the logs showed errors frequently under high concurrency",
            entity_path="svc",
            key="k",
            memory_type="belief",
            confidence=0.95,
        )
        types = [i.type for i in report.issues]
        assert "overconfident_belief" in types

    def test_fact_high_confidence_no_issue(self) -> None:
        report = self.evaluator.evaluate(
            {"max_retries": 3, "backoff": "exponential", "timeout_ms": 5000},
            entity_path="svc",
            key="k",
            memory_type="fact",
            confidence=1.0,
        )
        types = [i.type for i in report.issues]
        assert "overconfident_belief" not in types

    def test_good_structured_value_high_score(self) -> None:
        report = self.evaluator.evaluate(
            {"max_retries": 3, "backoff": "exponential", "timeout_ms": 5000},
            entity_path="svc",
            key="retry-config",
            pattern_refs=["error-handling"],
            existing_keys=["error-handling"],
        )
        assert report.score >= 0.8
        assert report.action == "stored_ok"

    def test_score_never_below_zero(self) -> None:
        report = self.evaluator.evaluate(
            "x",
            entity_path="svc",
            key="k",
            memory_type="belief",
            confidence=0.99,
            existing_keys=["a", "b"],
        )
        assert report.score >= 0.0


class TestQualityReportModel:
    def test_serialization_roundtrip(self) -> None:
        report = QualityReport(
            score=0.6,
            action="stored_with_suggestions",
            issues=[
                QualityIssue(
                    type="too_short",
                    message="Value is too short.",
                    suggestion="Add more detail.",
                ),
            ],
        )
        data = report.model_dump(mode="json")
        restored = QualityReport.model_validate(data)
        assert restored.score == 0.6
        assert len(restored.issues) == 1
        assert restored.issues[0].type == "too_short"

    def test_default_values(self) -> None:
        report = QualityReport(score=1.0)
        assert report.action == "stored_ok"
        assert report.issues == []

    def test_score_validation(self) -> None:
        import pytest

        with pytest.raises(Exception):
            QualityReport(score=1.5)
        with pytest.raises(Exception):
            QualityReport(score=-0.1)
