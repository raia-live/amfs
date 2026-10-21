"""SenseLab's repair loop, over its evaluation API (``/api/v1/eval``).

The server drafts its own probes from the agent's real fixes (it has the
material; the harness's draft is advisory), runs them through the same tiers a
real fix takes and refuses to ship them, so this adapter is thin: inject, poll,
read the A/A and the reward summary, and translate each into the harness's
shapes. Endpoints:

- ``POST /agents/{id}/gate/probes {probe, mode?}`` → ``{probe, fix}`` (201); 409
  when probes are off for the agent, one is already in the gate, or there is no
  material.
- ``GET /agents/{id}/gate/probes/{probe_id}`` → the probe row.
- ``POST /agents/{id}/canary/aa/run?trials=&window_days=`` → the A/A report
  (aggregates; the per-trial rows the harness scores are reconstructed from
  ``decidable`` / ``false_positives`` / ``median_graded_at_pass``, and the
  server's own verdict is kept in ``detail`` for comparison).
- ``GET /agents/{id}/gate/reward?window_days=`` → the reward summary; each fix's
  ``proxy`` counts and ``downstream`` windows become a
  :class:`~benchmarks.gate.protocol.RewardReading`.

Configuration: ``--url`` or ``AMFS_PRO_URL`` (default
``http://localhost:8000``), ``--api-key`` or ``AMFS_EVAL_API_KEY`` /
``AMFS_API_KEY``, ``--agent``, ``--mode dry_run|live`` (the server's default is
the agent's ``gate_probes`` setting). Standard library only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from ..protocol import (
    AAResult,
    AATrial,
    Counts,
    Decision,
    GateBusyError,
    Material,
    NoMaterialError,
    Probe,
    RewardReading,
    Stage,
    Status,
)

EVAL_BASE = "/api/v1/eval"
Transport = Callable[[str, str, dict[str, Any] | None], tuple[int, Any]]

#: A read that meets one of these is retried; see :meth:`SenseLabLoop._ok`.
TRANSIENT_STATUSES = frozenset({502, 503, 504})
READ_RETRIES = 6
READ_RETRY_BASE_S = 2.0
READ_RETRY_CAP_S = 30.0


class SenseLabApiError(RuntimeError):
    def __init__(self, status: int, detail: Any, path: str) -> None:
        super().__init__(f"{path} -> {status}: {detail}")
        self.status = status
        self.detail = detail
        self.path = path


def _urllib_transport(base_url: str, api_key: str, timeout: float = 120.0) -> Transport:
    def call(method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            base_url.rstrip("/") + path, data=data, method=method,
            headers={"X-AMFS-API-Key": api_key, "Accept": "application/json",
                     **({"Content-Type": "application/json"} if data is not None else {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — caller's URL
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {"detail": raw.decode(errors="replace")[:300]}
            return e.code, payload
    return call


def _message(payload: Any) -> str:
    if isinstance(payload, dict):
        d = payload.get("detail", payload)
        if isinstance(d, dict):
            return str(d.get("message") or d.get("code") or d)
        return str(d)
    return str(payload)


def _dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return None


class SenseLabLoop:
    name = "senselab"

    def __init__(self, agent_id: str, transport: Transport, *, mode: str | None = None,
                 reward_window_days: int = 90) -> None:
        self.agent_id = agent_id
        self._call = transport
        self.mode = mode
        self.reward_window_days = reward_window_days
        self.name = f"senselab:{agent_id}"

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> SenseLabLoop:
        agent = options.get("agent") or os.environ.get("AMFS_GATE_AGENT")
        if not agent:
            raise ValueError("the senselab loop needs --agent <agent_id>")
        url = options.get("url") or os.environ.get("AMFS_PRO_URL") or "http://localhost:8000"
        key = (options.get("api_key") or os.environ.get("AMFS_EVAL_API_KEY")
               or os.environ.get("AMFS_API_KEY"))
        if not key:
            raise ValueError("the senselab loop needs --api-key or AMFS_EVAL_API_KEY")
        return cls(str(agent), _urllib_transport(str(url), str(key)),
                   mode=options.get("mode"),
                   reward_window_days=int(options.get("reward_window_days", 90)))

    # -- plumbing ---------------------------------------------------------------

    def _agent_path(self, suffix: str, **params: Any) -> str:
        path = f"{EVAL_BASE}/agents/{urllib.parse.quote(self.agent_id, safe='')}{suffix}"
        q = {k: v for k, v in params.items() if v is not None}
        return path + ("?" + urllib.parse.urlencode(q) if q else "")

    def _ok(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        """One call, or for a read, up to :data:`READ_RETRIES` of them: a
        poll that meets a 502/503/504 — an instance recycling under the
        service, a cold start — is retried after a short, growing wait
        rather than ending a run that has hours of probes behind it. A write
        is never retried here: the server may have taken it."""
        attempts = READ_RETRIES if method == "GET" else 1
        for attempt in range(1, attempts + 1):
            status, payload = self._call(method, path, body)
            if status in TRANSIENT_STATUSES and attempt < attempts:
                self._sleep(min(READ_RETRY_CAP_S, READ_RETRY_BASE_S * 2 ** (attempt - 1)))
                continue
            if status >= 400:
                raise SenseLabApiError(status, payload, path)
            return payload
        raise AssertionError("unreachable")  # pragma: no cover

    _sleep = staticmethod(time.sleep)

    # -- the protocol -----------------------------------------------------------

    def material(self) -> list[Material]:
        return []  # the server drafts from its own fixes

    def inject(self, probe: Probe) -> str:
        body: dict[str, Any] = {"probe": probe.kind.value}
        if self.mode:
            body["mode"] = self.mode
        status, payload = self._call("POST", self._agent_path("/gate/probes"), body)
        if status == 409:
            msg = _message(payload)
            low = msg.lower()
            # "a probe is already in the gate" / "N repairs are open on branches":
            # wait; "no material for a ... probe": skip this kind.
            if "already" in low or "in the gate" in low or "open on branches" in low:
                raise GateBusyError(msg)
            if "material" in low or "needs a fix" in low or "not found" in low:
                raise NoMaterialError(msg)
            raise SenseLabApiError(status, payload, "/gate/probes")
        if status >= 400:
            raise SenseLabApiError(status, payload, "/gate/probes")
        return str(payload["probe"]["id"])

    def poll(self, handle: str) -> Decision:
        row = self._ok("GET", self._agent_path(f"/gate/probes/{urllib.parse.quote(handle)}"))
        return decision_from_row(row)

    def aa(self, *, trials: int, window_days: int) -> AAResult | None:
        try:
            report = self._ok("POST", self._agent_path("/canary/aa/run", trials=trials,
                                                       window_days=window_days))
        except SenseLabApiError as e:
            if e.status in (403, 404):
                return None
            raise
        return aa_from_report(report)

    def reward(self) -> list[RewardReading]:
        try:
            summary = self._ok("GET", self._agent_path("/gate/reward",
                                                       window_days=self.reward_window_days))
        except SenseLabApiError as e:
            if e.status in (403, 404):
                return []
            raise
        return [r for r in (reward_from_row(f) for f in summary.get("fixes") or []) if r]

    def close(self) -> None:
        return None


# -- translations -------------------------------------------------------------

def decision_from_row(row: Mapping[str, Any]) -> Decision:
    status = Status(str(row.get("status") or "open"))
    stages = tuple(
        Stage(tier=str(s.get("tier") or ""), verdict=str(s.get("verdict") or ""),
              at=_dt(s.get("at")), note=s.get("note"))
        for s in row.get("stages") or []
    )
    injected = _dt(row.get("injected_at"))
    if injected is None:
        raise ValueError("probe row without injected_at")
    return Decision(
        status=status, injected_at=injected, closed_at=_dt(row.get("closed_at")),
        decided_by=row.get("rejected_by") or row.get("accepted_by"),
        would_have_shipped=bool(row.get("would_have_shipped")), stages=stages,
        mode=row.get("mode"),
        detail={"probe_id": row.get("id"), "fix_id": row.get("fix_id"),
                "source_fix_id": row.get("source_fix_id"), "report": row.get("report") or {}},
    )


def aa_from_report(report: Mapping[str, Any]) -> AAResult:
    """The server's aggregate report as trials: ``decidable`` of them decided,
    ``false_positives`` of those falsely, the rest undecidable."""
    trials = int(report.get("trials") or 0)
    decidable = int(report.get("decidable") or 0)
    fps = int(report.get("false_positives") or 0)
    median = report.get("median_graded_at_pass")
    rows: list[AATrial] = []
    rows += [AATrial(decidable=True, false_positive=True) for _ in range(min(fps, decidable))]
    rows += [AATrial(decidable=True, false_positive=False, graded_at_decision=median)
             for _ in range(max(0, decidable - fps))]
    rows += [AATrial(decidable=False, false_positive=False)
             for _ in range(max(0, trials - decidable))]
    return AAResult(
        trials=tuple(rows), nominal=float(report.get("nominal") or 0.0),
        window_days=_days(report), ran_at=_dt(report.get("ran_at") or report.get("computed_at")),
        detail={"server_verdict": report.get("verdict"), "server_note": report.get("note"),
                "fp_rate": report.get("fp_rate"), "fp_lower": report.get("fp_lower"),
                "fp_upper": report.get("fp_upper"), "by_detail": report.get("by_detail")},
    )


def _days(report: Mapping[str, Any]) -> int | None:
    if report.get("window_days") is not None:
        return int(report["window_days"])
    a, b = _dt(report.get("window_start")), _dt(report.get("window_end"))
    return round((b - a).total_seconds() / 86400) if a and b else None


def _counts(d: Mapping[str, Any] | None, n_key: str, hit_key: str) -> Counts | None:
    if not d or d.get(n_key) is None:
        return None
    return Counts(n=int(d.get(n_key) or 0), hits=int(d.get(hit_key) or 0))


def reward_from_row(row: Mapping[str, Any]) -> RewardReading | None:
    """A reward-probe row (``GET /agents/{id}/gate/reward`` → ``fixes[]``) as a
    reading; ``None`` when the row carries no proxy counts (an older report)."""
    proxy = row.get("proxy") or {}
    if proxy.get("before_runs") is None or proxy.get("after_runs") is None:
        return None
    ds = row.get("downstream") or {}
    before, after = ds.get("before") or {}, ds.get("after") or {}
    return RewardReading(
        id=str(row.get("fix_id") or row.get("id")), title=str(row.get("title") or ""),
        shipped_at=_dt(row.get("shipped_at")),
        proxy_before=Counts(int(proxy["before_runs"]), int(proxy.get("before_count") or 0)),
        proxy_after=Counts(int(proxy["after_runs"]), int(proxy.get("after_count") or 0)),
        downstream_before=_counts(before, "external", "external_succeeded") or Counts(0, 0),
        downstream_after=_counts(after, "external", "external_succeeded") or Counts(0, 0),
        self_report_before=_counts(before, "outcomes", "succeeded"),
        self_report_after=_counts(after, "outcomes", "succeeded"),
        detail={"server_verdict": row.get("verdict"), "server_basis": row.get("basis"),
                "server_note": row.get("note"), "reward_probe_id": row.get("id")},
    )


__all__ = [
    "EVAL_BASE",
    "SenseLabApiError",
    "SenseLabLoop",
    "aa_from_report",
    "decision_from_row",
    "reward_from_row",
]
