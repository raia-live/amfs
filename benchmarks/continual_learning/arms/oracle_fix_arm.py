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
        # One batch per regime change: (readable-from episode, variant, entries), in order.
        self._batches: list[dict[str, Any]] = []

    def configure(self, scenario) -> None:
        super().configure(scenario)
        self._batches = []
        change_at = int(getattr(scenario, "change_at", 0) or 0)
        if change_at:
            primary = getattr(scenario, "change_variant", "A")
            entries = list(scenario.oracle_fix_entries(primary) or [])
            if entries:
                self._batches.append({"due": change_at + self.lag, "variant": primary, "entries": entries, "written": False})
            change_at2 = int(getattr(scenario, "change_at2", 0) or 0)
            if change_at2:
                other = scenario.OTHER_VARIANT[primary]
                entries2 = list(scenario.oracle_fix_entries(other) or [])
                if entries2:
                    self._batches.append({"due": change_at2 + self.lag, "variant": other, "entries": entries2, "written": False})
        self._maintainer = f"{scenario.agent_base}-maintainer"

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        # ``after_episode(ep)`` runs after episode ``ep``; a batch is readable from episode
        # ``change_at + lag`` on, so it is written after episode ``due - 1``.
        due = [b for b in self._batches if not b["written"] and episode + 1 >= b["due"]]
        if not due:
            return None
        acct = ArmAccounting()
        t0 = time.perf_counter()
        s = _SenseLabSession(self, self._maintainer, episode)
        notes: list[dict[str, Any]] = []
        try:
            for b in due:
                for key, text, conf in b["entries"]:
                    s.write(key, text, confidence=conf, kind="fact")
                    acct.ops += 1
                b["written"] = True
                notes.append({"lag": self.lag, "variant": b["variant"], "readable_from": b["due"],
                              "keys": [k for k, _, _ in b["entries"]]})
            acct.notes = {"oracle_fix": notes[0] if len(notes) == 1 else notes}
        except Exception as e:  # noqa: BLE001
            acct.notes = {"oracle_fix_error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            s.close()
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        return acct

    def report(self) -> dict[str, Any] | None:
        if not self._batches:
            return None
        return {"oracle_fix": {"lag": self.lag, "batches": [
            {"variant": b["variant"], "readable_from": b["due"], "written": b["written"],
             "entries": [k for k, _, _ in b["entries"]]} for b in self._batches]}}
