"""``raw-traces``: the "the model already knew" control.

Same model, no memory system: before each task the agent is handed the transcripts of
every prior episode in the cell (task, the tool calls it made, what came back, the
environment's verdict), newest last, up to ``config.RAW_TRACES_TOKEN_CAP``. No search, no
notes, no reflection, no outcome attribution — whatever the agent gets right, it got right
by reading raw experience in context.

This is the floor any memory arm has to clear on the composition scenarios: if a strong
model given all the traces composes the partial discoveries as well as SenseLab does, the
composition is the model's, not the memory's. It also bounds the cost of the alternative —
the transcript grows with every episode and the cap is what keeps the arm affordable.
"""

from __future__ import annotations

from typing import Any

from .. import config
from .base import ArmAccounting, EpisodeSession, MemoryArm, MemoryHit, Outcome

_CHARS_PER_TOKEN = 4


class _RawTracesSession(EpisodeSession):
    def __init__(self, arm: "RawTracesArm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: RawTracesArm = arm
        self._actions: list[str] = []

    def briefing(self) -> str | None:
        text = self.arm.render(config.RAW_TRACES_TOKEN_CAP * _CHARS_PER_TOKEN)
        if text:
            self.acct.retrieved_bytes += len(text)
            self.acct.ops += 1
        return text

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        return []

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        return None

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items() if k != "used_memory_keys")
        self._actions.append(f"  > {tool}({args[:300]})\n    {result[:400].replace(chr(10), ' ')}")

    def end(self, outcome: Outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        lines = [f"### Episode {self.episode + 1} — {self.agent_id}", f"Task: {task_input[:600]}"]
        lines.extend(self._actions)
        lines.append(f"Verdict: {outcome.summary[:300]}")
        self.arm.transcripts.append("\n".join(lines))


class RawTracesArm(MemoryArm):
    name = "raw-traces"
    learns_from_outcomes = False
    has_memory = False   # no memory tools, no reflection turn, no seeding
    briefs = True        # ...but the loop still asks for a briefing

    def open(self, scope: str) -> None:
        super().open(scope)
        self.transcripts: list[str] = []

    def render(self, char_cap: int) -> str | None:
        if not self.transcripts:
            return None
        # A contiguous window of the most recent episodes: walk newest-first
        # and stop at the first one that does not fit, so what is omitted is
        # exactly the oldest run of episodes the header says it is. Skipping a
        # large middle episode and keeping older small ones would hand the
        # control a gapped history while claiming a recent one.
        kept: list[str] = []
        used = 0
        for t in reversed(self.transcripts):
            if used + len(t) > char_cap and kept:
                break
            kept.append(t)
            used += len(t)
        dropped = len(self.transcripts) - len(kept)
        kept.reverse()
        head = ("Transcripts of every earlier episode in this workspace, oldest first. Nothing has "
                "been summarised or filtered; read them and draw your own conclusions.")
        if dropped:
            head += f" ({dropped} oldest episode(s) omitted to fit the context budget.)"
        return head + "\n\n" + "\n\n".join(kept)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _RawTracesSession(self, agent_id, episode)

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        return None
