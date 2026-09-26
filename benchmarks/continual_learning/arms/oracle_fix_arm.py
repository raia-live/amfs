"""``senselab-oracle-fix-<lag>``: the human maintainer the repair loop is measured against.

The ``senselab`` protocol, unchanged, plus a scripted maintainer who has been told about the
regime change: ``lag`` episodes after ``change_at`` it writes one corrective entry per changed
class (``Scenario.oracle_fix_entries``) to the cell's scope at confidence 1.0, under its own
identity. Nothing else is different — the agent still has to retrieve and follow the entry,
and the stale lesson still has to lose its evidence through outcomes.

Three lags bracket a human's response time (HYPOTHESIS_RSI.md, E2/H6): 5 episodes (a
maintainer watching the queue), 10 (the weekly review), 20 (the quarter's retrospective).
The repair loop has no answer key; if it repairs within the lag of the attentive maintainer
at the fixed budget, the loop is doing a maintainer's job at a maintainer's speed.
"""

from __future__ import annotations

import time
from typing import Any

from .base import ArmAccounting
from .senselab_arm import SenseLabArm, _SenseLabSession


class SenseLabOracleFixArm(SenseLabArm):
    name = "senselab-oracle-fix"

    def __init__(self, lag: int = 10) -> None:
        super().__init__()
        self.lag = int(lag)
        self.name = f"senselab-oracle-fix-{self.lag}"

    def open(self, scope: str) -> None:
        super().open(scope)
        self._entries: list[tuple[str, str, float]] = []
        self._due: int | None = None
        self._written = False

    def configure(self, scenario) -> None:
        self._entries = list(scenario.oracle_fix_entries() or [])
        change_at = int(getattr(scenario, "change_at", 0) or 0)
        self._due = change_at + self.lag if change_at and self._entries else None
        self._maintainer = f"{scenario.agent_base}-maintainer"

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        # ``after_episode(ep)`` runs after episode ``ep``; the entries are readable from
        # episode ``change_at + lag`` on, so they are written after episode ``due - 1``.
        if self._written or self._due is None or episode + 1 < self._due:
            return None
        acct = ArmAccounting()
        t0 = time.perf_counter()
        s = _SenseLabSession(self, self._maintainer, episode)
        try:
            for key, text, conf in self._entries:
                s.write(key, text, confidence=conf, kind="fact")
                acct.ops += 1
            self._written = True
            acct.notes = {"oracle_fix": {"lag": self.lag, "readable_from": self._due, "keys": [k for k, _, _ in self._entries]}}
        except Exception as e:  # noqa: BLE001
            acct.notes = {"oracle_fix_error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            s.close()
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        return acct

    def report(self) -> dict[str, Any] | None:
        if self._due is None:
            return None
        return {"oracle_fix": {"lag": self.lag, "readable_from": self._due, "written": self._written,
                               "entries": [k for k, _, _ in self._entries]}}
