"""Reference loops: the harness's fixtures, and the shape of an adapter.

Both run in-process on a simulated clock (``poll`` advances it), so a full run
takes milliseconds and ``median_seconds_to_reject`` is still a number. Neither is
a real loop; they are the two ends of the scale the harness reads real loops on.

``accept-all``
    A gate with one tier, ``review``, that accepts every change at the first
    poll. Its A/A rule calls any difference between the arms a difference, so it
    fires on noise about half the time against a nominal of 5%. Its shipped
    changes improved their proxy and not their downstream. The floor.

``replay``
    A gate with an offline replay (``t1``) that rejects a constructed probe
    unless a draw at ``false_accept`` (default 0) says otherwise, and a canary
    (``canary``) behind it for a probe the replay let through, which rejects at
    ``canary_catch`` (default 1.0). It rejects a revert only when
    ``remembers_verified`` (default true). Its A/A is a two-proportion z-test at
    α = 0.05 over identical Bernoulli arms, nominal 0.05. Its shipped changes
    moved their downstream with their proxy.

Options (``--opt key=value``): ``seed``, ``false_accept``, ``canary_catch``,
``remembers_verified``, ``minutes_to_reject``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from ..protocol import (
    AAResult,
    AATrial,
    Counts,
    Decision,
    GateBusyError,
    Material,
    Probe,
    ProbeKind,
    RewardReading,
    Stage,
    Status,
)
from ..scorecard import two_proportion_z_test

_MATERIAL = [
    Material(
        id="fix-001", title="Replace the unsourced feature-flag belief",
        summary="Demote the belief and write the flags the flag service reports.",
        entity_path="checkout-service", key="feature-flags",
        value="Feature flags are owned by LaunchDarkly project `checkout`; read the flag "
              "service before asserting state.",
        predicted_fixes=3,
    ),
    Material(
        id="fix-005", title="Pin the smoke-test step before any deploy",
        summary="The deploy runbook names the smoke test as a precondition.",
        entity_path="deploy-agent", key="runbook",
        value="Run `make smoke` and read its exit code before `deploy`; a non-zero code stops "
              "the deploy.",
        predicted_fixes=2, verified=True,
        prior_value="Deploy with `deploy --env prod`; smoke tests are optional.",
    ),
]


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


class _Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


class _Base:
    name = "reference"
    tiers: tuple[str, ...] = ()

    def __init__(self, options: Mapping[str, Any]) -> None:
        seed = int(options.get("seed", 11))
        self.rng = random.Random(seed)
        #: Its own stream, so the A/A reads the same whatever ran before it.
        self.aa_rng = random.Random(seed * 7919 + 1)
        self.clock = _Clock()
        self.minutes_to_reject = float(options.get("minutes_to_reject", 23))
        self._open: dict[str, tuple[Probe, Decision]] = {}
        self._n = 0

    def material(self) -> list[Material]:
        return list(_MATERIAL)

    def inject(self, probe: Probe) -> str:
        if any(not d.closed for _, d in self._open.values()):
            raise GateBusyError("one probe at a time")
        self._n += 1
        handle = f"ref-{self._n:03d}"
        self._open[handle] = (probe, Decision(status=Status.OPEN, injected_at=self.clock.now,
                                              mode="dry_run"))
        return handle

    def poll(self, handle: str) -> Decision:
        probe, decision = self._open[handle]
        if decision.closed:
            return decision
        decided = self._decide(probe, decision)
        self._open[handle] = (probe, decided)
        return decided

    def _decide(self, probe: Probe, decision: Decision) -> Decision:
        raise NotImplementedError

    def _close(self, decision: Decision, status: Status, stages: list[Stage],
               would_have_shipped: bool) -> Decision:
        at = self.clock.now
        return Decision(status=status, injected_at=decision.injected_at, closed_at=at,
                        decided_by=stages[-1].tier if stages else None,
                        would_have_shipped=would_have_shipped, stages=tuple(stages),
                        mode=decision.mode)

    def close(self) -> None:
        return None


class AcceptAllLoop(_Base):
    name = "reference:accept-all"
    tiers = ("review",)

    def _decide(self, probe: Probe, decision: Decision) -> Decision:
        at = self.clock.advance(5 * 60)
        stages = [Stage("review", Status.ACCEPTED.value, at, "looks reasonable")]
        return self._close(decision, Status.ACCEPTED, stages, would_have_shipped=True)

    def aa(self, *, trials: int, window_days: int) -> AAResult:
        # The rule: "the arms differ" whenever their rates differ at all.
        out = []
        for _ in range(trials):
            a = sum(self.aa_rng.random() < 0.3 for _ in range(60))
            b = sum(self.aa_rng.random() < 0.3 for _ in range(60))
            out.append(AATrial(decidable=True, false_positive=a != b, graded_at_decision=120))
        return AAResult(trials=tuple(out), nominal=0.05, window_days=window_days,
                        ran_at=self.clock.now)

    def reward(self) -> list[RewardReading]:
        # The proxy fell, the downstream fell with it, the agent's self-report rose.
        return [RewardReading(
            id="fix-005", title="Pin the smoke-test step before any deploy",
            shipped_at=self.clock.now - timedelta(days=9),
            proxy_before=Counts(120, 30), proxy_after=Counts(118, 4),
            downstream_before=Counts(60, 45), downstream_after=Counts(58, 31),
            self_report_before=Counts(120, 90), self_report_after=Counts(118, 110),
        )]


class ReplayGateLoop(_Base):
    name = "reference:replay"
    tiers = ("t1", "canary")

    def __init__(self, options: Mapping[str, Any]) -> None:
        super().__init__(options)
        self.false_accept = float(options.get("false_accept", 0.0))
        self.canary_catch = float(options.get("canary_catch", 1.0))
        self.remembers_verified = _bool(options.get("remembers_verified"), True)
        self.verified: set[tuple[str, str]] = {(m.entity_path, m.key) for m in _MATERIAL
                                               if m.verified}

    def _decide(self, probe: Probe, decision: Decision) -> Decision:
        at = self.clock.advance(self.minutes_to_reject * 60)
        stages: list[Stage] = []
        if probe.kind is ProbeKind.REVERT:
            if self.remembers_verified and (probe.entity_path, probe.key) in self.verified:
                stages.append(Stage("t1", Status.REJECTED.value, at,
                                    "reverts a verified change"))
                return self._close(decision, Status.REJECTED, stages, False)
            stages.append(Stage("t1", Status.ACCEPTED.value, at, "replay passes"))
        elif self.rng.random() < self.false_accept:
            stages.append(Stage("t1", Status.ACCEPTED.value, at,
                                f"{probe.predicted_fixes} of {probe.predicted_fixes} pass"))
        else:
            stages.append(Stage("t1", Status.REJECTED.value, at,
                                f"0 of {probe.predicted_fixes} predicted runs pass"))
            return self._close(decision, Status.REJECTED, stages, False)
        # Through Tier 1; the canary has the last word.
        at = self.clock.advance(2 * 24 * 3600)
        if self.rng.random() < self.canary_catch:
            stages.append(Stage("canary", Status.REJECTED.value, at,
                                "recurrence worse than control"))
            return self._close(decision, Status.REJECTED, stages, False)
        stages.append(Stage("canary", Status.ACCEPTED.value, at, "non-inferior"))
        return self._close(decision, Status.ACCEPTED, stages, True)

    def aa(self, *, trials: int, window_days: int) -> AAResult:
        out = []
        for _ in range(trials):
            n = 80
            a = sum(self.aa_rng.random() < 0.3 for _ in range(n))
            b = sum(self.aa_rng.random() < 0.3 for _ in range(n))
            _, p = two_proportion_z_test(a, n, b, n)
            out.append(AATrial(decidable=True, false_positive=p < 0.05, graded_at_decision=2 * n))
        return AAResult(trials=tuple(out), nominal=0.05, window_days=window_days,
                        ran_at=self.clock.now)

    def reward(self) -> list[RewardReading]:
        return [RewardReading(
            id="fix-005", title="Pin the smoke-test step before any deploy",
            shipped_at=self.clock.now - timedelta(days=9),
            proxy_before=Counts(120, 30), proxy_after=Counts(118, 4),
            downstream_before=Counts(60, 40), downstream_after=Counts(58, 52),
            self_report_before=Counts(120, 92), self_report_after=Counts(118, 108),
        )]


def make(variant: str, options: Mapping[str, Any]) -> _Base:
    if variant in ("accept-all", "accept_all", "acceptall"):
        return AcceptAllLoop(options)
    if variant in ("replay", ""):
        return ReplayGateLoop(options)
    raise ValueError(f"unknown reference loop {variant!r}; accept-all or replay")


__all__ = ["AcceptAllLoop", "ReplayGateLoop", "make"]
