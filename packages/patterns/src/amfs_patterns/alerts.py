"""AlertManager — configurable rules that fire when matching patterns are detected.

Features:
  - Severity filtering per rule
  - Entity-path scoping (glob-style)
  - Cooldown-based suppression to prevent alert fatigue
  - Callback registration for routing to Slack, PagerDuty, email, or custom systems
"""

from __future__ import annotations

import fnmatch
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from amfs_patterns.detector import DetectedPattern, PatternReport

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


class AlertRule(BaseModel):
    """A single alert rule that matches against detected patterns."""

    name: str
    pattern_type: str | None = None
    min_severity: str = "info"
    entity_path_glob: str | None = None
    cooldown_minutes: int = 0
    enabled: bool = True


class AlertEvaluation(BaseModel):
    """Result of evaluating a rule against a detected pattern."""

    rule_name: str
    pattern: DetectedPattern
    fired: bool = True
    suppressed: bool = False
    suppression_reason: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertManager:
    """Evaluates detected patterns against configurable alert rules.

    Usage::

        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="Critical recurring failures",
            pattern_type="recurring_failure",
            min_severity="critical",
            cooldown_minutes=60,
        ))
        manager.on_alert(lambda eval: send_slack_notification(eval))

        evaluations = manager.evaluate(report)
    """

    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._callbacks: list[Callable[[AlertEvaluation], Any]] = []
        self._last_fired: dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def on_alert(self, callback: Callable[[AlertEvaluation], Any]) -> None:
        """Register a callback invoked for every fired (non-suppressed) alert."""
        self._callbacks.append(callback)

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules)

    def evaluate(self, report: PatternReport) -> list[AlertEvaluation]:
        """Evaluate all rules against a pattern report.

        Returns AlertEvaluation objects for every rule/pattern match.
        Fires callbacks for non-suppressed alerts.
        """
        evaluations: list[AlertEvaluation] = []
        now = datetime.now(timezone.utc)

        for pattern in report.patterns:
            for rule in self._rules:
                if not rule.enabled:
                    continue

                if not self._rule_matches(rule, pattern):
                    continue

                cooldown_key = f"{rule.name}:{pattern.entity_path}:{pattern.pattern_type}"
                suppressed = False
                suppression_reason = None

                if rule.cooldown_minutes > 0:
                    last = self._last_fired.get(cooldown_key)
                    if last and (now - last) < timedelta(minutes=rule.cooldown_minutes):
                        suppressed = True
                        remaining = rule.cooldown_minutes - (now - last).total_seconds() / 60
                        suppression_reason = (
                            f"Cooldown active: {remaining:.0f}min remaining"
                        )

                evaluation = AlertEvaluation(
                    rule_name=rule.name,
                    pattern=pattern,
                    fired=not suppressed,
                    suppressed=suppressed,
                    suppression_reason=suppression_reason,
                )
                evaluations.append(evaluation)

                if not suppressed:
                    self._last_fired[cooldown_key] = now
                    for cb in self._callbacks:
                        try:
                            cb(evaluation)
                        except Exception:
                            logger.warning(
                                "Alert callback failed for rule %s",
                                rule.name,
                                exc_info=True,
                            )

        return evaluations

    def _rule_matches(self, rule: AlertRule, pattern: DetectedPattern) -> bool:
        """Check if a rule matches a detected pattern."""
        if rule.pattern_type and rule.pattern_type != pattern.pattern_type:
            return False

        rule_sev = _SEVERITY_ORDER.get(rule.min_severity, 0)
        pattern_sev = _SEVERITY_ORDER.get(pattern.severity, 0)
        if pattern_sev < rule_sev:
            return False

        if rule.entity_path_glob:
            if not fnmatch.fnmatch(pattern.entity_path, rule.entity_path_glob):
                return False

        return True
