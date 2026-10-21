"""Thin client for the SenseLab Pro evaluation API (``/api/v1/eval``).

The ``senselab-repair`` arm drives the repair loop from the harness: grade the episode's
sealed trace with a judge, propose a fix from the failing verdict, run its Tier 1 replay,
and ship it when the test passes. Only the endpoints that loop needs are wrapped; each
returns the server's JSON as-is so the arm records whatever the server said.

Safety: the repair loop writes corrective memory into the store the benchmark reads from,
and the Pro process it talks to must have its repair agent enabled. Both are dev-only
concerns, so :func:`assert_dev` refuses a base URL that does not look like a development
or local deployment unless ``CL_REPAIR_ALLOW_ANY_HOST=1`` is set.
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from . import config

_DEV_MARKERS = ("dev", "staging", "localhost", "127.0.0.1", "0.0.0.0", ".local")


class ProApiError(RuntimeError):
    def __init__(self, status: int, detail: Any, path: str) -> None:
        super().__init__(f"{path} -> {status}: {detail}")
        self.status = status
        self.detail = detail
        self.path = path


def is_dev_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(m in host for m in _DEV_MARKERS)


def assert_dev(*urls: str) -> None:
    """Refuse to point the repair loop at anything that is not a dev deployment."""
    if os.environ.get("CL_REPAIR_ALLOW_ANY_HOST", "0") in ("1", "true", "yes"):
        return
    bad = [u for u in urls if u and not is_dev_url(u)]
    if bad:
        raise RuntimeError(
            "the senselab-repair arm writes corrective memory and needs the Pro repair agent; "
            f"refusing non-dev host(s) {bad}. Set AMFS_HTTP_URL / AMFS_PRO_URL to a dev "
            "deployment or CL_REPAIR_ALLOW_ANY_HOST=1 to override."
        )


class ProClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, *,
                 timeout: float = 120.0) -> None:
        self.base_url = (base_url or config.AMFS_PRO_URL).rstrip("/")
        key = api_key or config.env("AMFS_EVAL_API_KEY") or config.env("AMFS_API_KEY", required=True)
        self._client = httpx.Client(base_url=self.base_url, headers={"X-AMFS-API-Key": key},
                                    timeout=httpx.Timeout(timeout, connect=10.0))

    # -- plumbing -------------------------------------------------------------------
    def _call(self, method: str, path: str, *, json: Any = None, params: dict[str, Any] | None = None,
              retries: int = 4, timeout: float | None = None) -> Any:
        for attempt in range(retries + 1):
            kw: dict[str, Any] = {}
            if timeout is not None:
                kw["timeout"] = httpx.Timeout(timeout, connect=10.0)
            resp = self._client.request(method, path, json=json, params=params, **kw)
            if resp.status_code in (429, 502, 503, 504) and attempt < retries:
                wait = min(float(resp.headers.get("Retry-After", 1.0)) * (1.5 ** attempt), 20.0)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except ValueError:
                    detail = resp.text[:300]
                raise ProApiError(resp.status_code, detail, path)
            if not resp.content:
                return {}
            return resp.json()
        raise RuntimeError("unreachable")

    def close(self) -> None:
        self._client.close()

    # -- judges ---------------------------------------------------------------------
    def ensure_judge(self, judge_id: str, agent_id: str, name: str, prompt: str, *,
                     model: str | None = None, mode: str = "on_demand") -> dict[str, Any]:
        """Create the judge, or return the existing one (judges are per agent). ``mode`` is
        ``on_demand`` (graded when asked) or ``continuous`` (the eval worker grades every
        trace the agent seals)."""
        try:
            return self._call("GET", f"/api/v1/eval/judges/{judge_id}", params={"agent_id": agent_id})
        except ProApiError as e:
            if e.status != 404:
                raise
        body: dict[str, Any] = {"id": judge_id, "agent_id": agent_id, "name": name, "prompt": prompt,
                                "scoring_type": "binary", "mode": mode, "enabled": True}
        if model:
            body["model"] = model
        try:
            return self._call("POST", "/api/v1/eval/judges", json=body)
        except ProApiError as e:
            if e.status == 409:  # raced with another cell
                return self._call("GET", f"/api/v1/eval/judges/{judge_id}", params={"agent_id": agent_id})
            raise

    def judge(self, trace_id: str, judge_id: str, *, agent_id: str | None = None,
              force: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"trace_id": trace_id, "judge_id": judge_id, "persist": True, "force": force}
        if agent_id:
            body["agent_id"] = agent_id
        return self._call("POST", "/api/v1/eval/judge", json=body)

    # -- fixes ----------------------------------------------------------------------
    def propose(self, agent_id: str, *, verdict_id: str | None = None, behavior_id: str | None = None,
                lever_override: str | None = None, requested_by: str = "cl-benchmark") -> dict[str, Any]:
        body: dict[str, Any] = {"agent_id": agent_id, "requested_by": requested_by}
        if verdict_id:
            body["verdict_id"] = verdict_id
        if behavior_id:
            body["behavior_id"] = behavior_id
        if lever_override:
            body["lever_override"] = lever_override
        return self._call("POST", "/api/v1/eval/fixes/propose", json=body)

    def test_fix(self, fix_id: str, *, now: bool = True) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/test", params={"now": "true" if now else "false"})

    def get_fix(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}")

    def wait_tested(self, fix_id: str, *, timeout_s: float = 180.0, poll_s: float = 3.0) -> dict[str, Any]:
        """Poll until the fix leaves ``proposed``/``testing``."""
        deadline = time.monotonic() + timeout_s
        fix = self.get_fix(fix_id)
        while fix.get("status") in ("proposed", "testing") and time.monotonic() < deadline:
            time.sleep(poll_s)
            fix = self.get_fix(fix_id)
        return fix

    def approve_memory(self, fix_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/approve-memory")

    def fix_events(self, fix_id: str, limit: int = 50) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/events", params={"limit": limit})

    def list_fixes(self, agent_id: str, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Every fix of one agent (all pages), optionally in one or several comma-separated statuses."""
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"agent_id": agent_id, "limit": min(limit, 500)}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            page = self._call("GET", "/api/v1/eval/fixes", params=params)
            items.extend(i for i in (page.get("items") or []) if isinstance(i, dict))
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or len(items) >= limit:
                return items

    def fix_tests(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/tests")

    def rollback(self, fix_id: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/rollback", json={"reason": reason[:2000]})

    def prevention(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/prevention")

    def recheck(self, fix_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/recheck")

    # -- canary (Tier 3) ------------------------------------------------------------
    def agent_canary(self, agent_id: str) -> dict[str, Any]:
        """``{"active": fix | None, "waiting": [fix, ...]}`` — the agent's canary slot."""
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/canary")

    def canary_assignment(self, agent_id: str, session: str, *, situation: str | None = None) -> dict[str, Any]:
        """The arm the hosted router would put *session* in. ``situation`` is what the session
        would declare on the wire (``X-AMFS-Situation``); from the situation-aware routing of
        2026-10-20 a canary scoped to a situation answers ``outside`` / ``undeclared`` for a
        session not in it."""
        params: dict[str, Any] = {"session": session}
        if situation:
            params["situation"] = situation
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/canary/assignment", params=params)

    def start_canary(self, fix_id: str, *, pct: int | None = None, min_outcomes: int | None = None,
                     timeout_days: int | None = None) -> dict[str, Any]:
        body = {k: v for k, v in dict(pct=pct, min_outcomes=min_outcomes, timeout_days=timeout_days).items()
                if v is not None}
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary", json=body)

    def stop_canary(self, fix_id: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary/stop", json={"reason": reason[:2000]})

    def evaluate_canary(self, fix_id: str) -> dict[str, Any]:
        """Re-count both arms and act on the decision, as the hourly sweep would."""
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary/evaluate")

    # -- customer replay (Tier 2) -----------------------------------------------------
    def request_replay(self, fix_id: str, *, wait_hours: int | None = None) -> dict[str, Any]:
        body = {"wait_hours": wait_hours} if wait_hours else {}
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/replay", json=body)

    def grade_replay(self, fix_id: str) -> dict[str, Any]:
        """Grade what the agent has replayed so far and move the fix if the replay decided
        (what the hourly replay sweep does). Grading judges each replayed trace on demand,
        so the call can take a minute for a suite of cases."""
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/replay/grade", timeout=280.0)

    def test_replay_webhook(self, agent_id: str) -> dict[str, Any]:
        """One signed ``ping`` to the agent's replay webhook; what came back."""
        return self._call("POST", f"/api/v1/eval/agents/{agent_id}/repair-settings/replay-webhook/test",
                          timeout=60.0)

    def feedback_contract(self, agent_id: str, policy: str | None = None) -> dict[str, Any]:
        params = {"policy": policy} if policy else None
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/feedback-contract", params=params)

    # -- account-level -----------------------------------------------------------------
    def readiness(self) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/readiness")

    def budget(self) -> dict[str, Any]:
        """``spent_usd`` is every platform-side LLM call the account caused this month
        (judges, repair agent, stand-in replay, lever, discovery), settled as it happens.
        The runner snapshots it so a run's platform spend is a difference of two readings."""
        return self._call("GET", "/api/v1/eval/budget")

    def repairs_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/repairs/summary", params={"agent_id": agent_id} if agent_id else None)

    def verdicts(self, agent_id: str, *, limit: int = 200, **params: Any) -> dict[str, Any]:
        q: dict[str, Any] = {"agent_id": agent_id, "limit": limit}
        q.update({k: v for k, v in params.items() if v is not None})
        return self._call("GET", "/api/v1/eval/verdicts", params=q)

    def incidents(self, agent_id: str, *, limit: int = 100) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/incidents", params={"agent_id": agent_id, "limit": limit})

    def behaviors(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/behaviors", params={"agent_id": agent_id})

    # -- repair settings ------------------------------------------------------------
    def get_repair_settings(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/repair-settings")

    def set_repair_settings(self, agent_id: str, **fields: Any) -> dict[str, Any]:
        body = {k: v for k, v in fields.items() if v is not None}
        return self._call("PUT", f"/api/v1/eval/agents/{agent_id}/repair-settings", json=body)

    # -- repair prompt --------------------------------------------------------------
    def get_repair_prompt(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/repair-prompt")

    def set_repair_prompt(self, agent_id: str, prompt: str) -> dict[str, Any]:
        """Replace the active prompt that drafts this agent's repairs."""
        return self._call("PUT", f"/api/v1/eval/agents/{agent_id}/repair-prompt", json={"prompt": prompt})


__all__ = ["ProApiError", "ProClient", "assert_dev", "is_dev_url"]
