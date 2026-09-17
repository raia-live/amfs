"""Control arm: the agent has no memory tools at all."""

from __future__ import annotations

from .base import EpisodeSession, MemoryArm, MemoryHit


class _NoSession(EpisodeSession):
    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        return []

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        return None


class NoMemoryArm(MemoryArm):
    name = "none"
    has_memory = False

    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:  # nothing to seed into
        return None

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _NoSession(self, agent_id, episode)
