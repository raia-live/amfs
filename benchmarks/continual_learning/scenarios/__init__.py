from __future__ import annotations

from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec

_REGISTRY: dict[str, type[Scenario]] | None = None


def registry() -> dict[str, type[Scenario]]:
    global _REGISTRY
    if _REGISTRY is None:
        from .compose import DisjointFleetScenario
        from .diagnose import DiagnoseScenario, SizingScenario
        from .drift import DriftFactScenario, DriftToolScenario
        from .handoff import HandoffScenario
        from .realworld import (AnalyticsScenario, CiFixScenario, ConciergeScenario, OrderOpsScenario,
                                RetentionScenario, SupportScenario)
        from .runbook import FleetScenario, RunbookScenario, UnknownsScenario
        from .triage import TriageScenario
        # Grid v4 (2026-09-19): RAG on its home turf, multi-user personalisation, routing.
        from .accounts import AccountsScenario
        from .dispatch import DispatchScenario
        from .policyqa import PolicyQaScenario

        _REGISTRY = {c.name: c for c in (
            RunbookScenario, UnknownsScenario, FleetScenario, DriftFactScenario, DriftToolScenario,
            TriageScenario, HandoffScenario, DiagnoseScenario, SizingScenario,
            SupportScenario, ConciergeScenario, RetentionScenario, OrderOpsScenario, CiFixScenario, AnalyticsScenario,
            DisjointFleetScenario,
            PolicyQaScenario, AccountsScenario, DispatchScenario,
        )}
    return _REGISTRY


def make_scenario(name: str, seed: int, episodes: int, **sweep: Any) -> Scenario:
    return registry()[name](seed, episodes, **sweep)


__all__ = ["Scenario", "StepResult", "Task", "ToolSpec", "make_scenario", "registry"]
