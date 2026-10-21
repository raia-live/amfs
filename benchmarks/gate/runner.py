"""Driving a loop through the three self-tests and writing the ledger.

``run`` injects probes one at a time — most gates take one change at a time and
a probe must be in the gate alone for its decision to be its own — polls each to
a decision or the timeout, then asks the loop for its A/A and its reward
readings, scores the three and writes the results directory (``ledger.py``).
``score`` reads a results directory a loop wrote itself and does only the last
step. Both are what ``python -m benchmarks.gate`` calls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, ledger, scorecard
from . import probes as probes_mod
from .protocol import (
    Decision,
    GateBusyError,
    GatedLoop,
    Material,
    NoMaterialError,
    Probe,
    ProbeKind,
    Stage,
    Status,
)

log = logging.getLogger("benchmarks.gate")


@dataclass
class RunConfig:
    probes: int = 8
    kinds: Sequence[ProbeKind] | None = None
    #: Seconds between polls of an open probe.
    poll_interval: float = 30.0
    #: Seconds a probe may stay open before the harness marks it ``expired``
    #: and moves on. A dry-run probe closes in minutes; a live one takes the
    #: canary's time (days). 0 = wait forever.
    timeout: float = 24 * 3600.0
    #: Seconds to wait for the gate to free when the loop takes one probe at a
    #: time and one is already in it (a real fix of the loop's, say).
    busy_wait: float = 6 * 3600.0
    aa_trials: int = scorecard.DEFAULT_AA_TRIALS
    aa_window_days: int = scorecard.DEFAULT_AA_WINDOW_DAYS
    skip_aa: bool = False
    skip_reward: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    card: dict[str, Any]
    records: list[dict[str, Any]]
    out_dir: Path | None


def _expired(d: Decision, now: datetime) -> Decision:
    return Decision(
        status=Status.EXPIRED, injected_at=d.injected_at, closed_at=now, decided_by=None,
        would_have_shipped=d.would_have_shipped,
        stages=tuple(d.stages) + (Stage("harness", "expired", now, "no tier answered in time"),),
        mode=d.mode, detail=d.detail,
    )


def drive_probe(
    loop: GatedLoop, probe: Probe, cfg: RunConfig, *,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
) -> tuple[str, Decision]:
    """Inject *probe* (waiting out a busy gate up to ``cfg.busy_wait``) and poll
    it to a closed decision or ``cfg.timeout``."""
    start = clock()
    while True:
        try:
            handle = loop.inject(probe)
            break
        except GateBusyError as e:
            if cfg.busy_wait and clock() - start > cfg.busy_wait:
                raise RuntimeError(f"the gate stayed busy for {cfg.busy_wait:.0f}s: {e}") from e
            log.info("gate busy (%s); waiting %.0fs", e, cfg.poll_interval)
            sleep(cfg.poll_interval)
    injected = clock()
    while True:
        decision = loop.poll(handle)
        if decision.closed:
            return handle, decision
        if cfg.timeout and clock() - injected > cfg.timeout:
            return handle, _expired(decision, datetime.now(UTC))
        sleep(cfg.poll_interval)


def run(
    loop: GatedLoop, cfg: RunConfig, *, out_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    started = datetime.now(UTC)
    material: list[Material] = list(loop.material())
    records: list[dict[str, Any]] = []
    kinds = probes_mod.kind_cycle(cfg.kinds, cfg.probes)
    per_kind: dict[ProbeKind, int] = {}
    for i, kind in enumerate(kinds):
        idx = per_kind.get(kind, 0)
        per_kind[kind] = idx + 1
        try:
            draft = probes_mod.draft(kind, material, index=idx) if material else _bare(kind)
        except NoMaterialError as e:
            log.warning("probe %d (%s) skipped: %s", i + 1, kind.value, e)
            continue
        try:
            handle, decision = drive_probe(loop, draft, cfg, sleep=sleep, clock=clock)
        except NoMaterialError as e:
            log.warning("probe %d (%s) refused by the loop: %s", i + 1, kind.value, e)
            continue
        log.info("probe %d/%d %s -> %s by %s", i + 1, len(kinds), kind.value,
                 decision.status.value, decision.decided_by or "—")
        records.append(ledger.probe_record(handle, kind.value, decision, probe=draft))
    aa_result = (None if cfg.skip_aa
                 else loop.aa(trials=cfg.aa_trials, window_days=cfg.aa_window_days))
    readings = None if cfg.skip_reward else loop.reward()
    card = scorecard.compile_scorecard(
        loop=loop.name,
        probes=scorecard.probe_scorecard(records),
        aa=scorecard.aa_scorecard(aa_result),
        reward=scorecard.reward_scorecard(readings),
        meta={
            "harness_version": __version__, "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(), "probes_requested": cfg.probes,
            "kinds": [k.value for k in (cfg.kinds or ProbeKind)], "timeout_s": cfg.timeout,
            "aa_trials": cfg.aa_trials, "aa_window_days": cfg.aa_window_days, **cfg.meta,
        },
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ledger.write_probes(out_dir / ledger.PROBES_FILE, records)
        ledger.write_json(out_dir / ledger.AA_FILE, ledger.aa_to_json(aa_result))
        ledger.write_json(out_dir / ledger.REWARD_FILE, ledger.reward_to_json(readings))
        _write_card(out_dir, card)
    return RunResult(card=card, records=records, out_dir=out_dir)


def _bare(kind: ProbeKind) -> Probe:
    """A draft with only the kind meaningful, for a loop that drafts its own."""
    return Probe(kind=kind, title=f"Probe ({kind.value})", summary="", entity_path="", key="",
                 value="", why_wrong="drafted by the loop")


def _write_card(out_dir: Path, card: dict[str, Any]) -> None:
    ledger.write_json(out_dir / ledger.SCORECARD_JSON, card)
    (out_dir / ledger.SCORECARD_MD).write_text(scorecard.render_markdown(card), encoding="utf-8")


def score(results_dir: Path, *, loop_name: str | None = None, write: bool = True) -> dict[str, Any]:
    """Score a results directory a loop wrote itself (``ledger.py`` schema)."""
    probes_path = results_dir / ledger.PROBES_FILE
    if not probes_path.exists():
        raise FileNotFoundError(f"{probes_path} — a ledger needs at least the probes file")
    records = ledger.read_probes(probes_path)
    aa_result = ledger.aa_from_json(ledger.read_json(results_dir / ledger.AA_FILE))
    readings = ledger.reward_from_json(ledger.read_json(results_dir / ledger.REWARD_FILE))
    run_meta = ledger.read_json(results_dir / ledger.RUN_FILE) or {}
    card = scorecard.compile_scorecard(
        loop=loop_name or str(run_meta.get("loop") or results_dir.name),
        probes=scorecard.probe_scorecard(records),
        aa=scorecard.aa_scorecard(aa_result),
        reward=scorecard.reward_scorecard(readings),
        meta={"harness_version": __version__, "scored_at": datetime.now(UTC).isoformat(),
              "source": "ledger", **{k: v for k, v in run_meta.items() if k != "loop"}},
    )
    if write:
        _write_card(results_dir, card)
    return card


__all__ = ["RunConfig", "RunResult", "drive_probe", "run", "score"]
