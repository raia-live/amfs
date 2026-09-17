"""Zep (cloud) arm, wired as documented.

Writes go through ``graph.add`` so Zep's temporal knowledge graph does its work: entity
and fact extraction, and invalidation of superseded facts (``invalid_at``). Reads use
``graph.search`` and return current facts (edges) plus entity summaries (nodes); edges
that Zep has invalidated are excluded, because honouring its invalidation is the point.

Graph ingestion is asynchronous and slow (60-450 s per episode observed, queue-dependent).
Blocking on it would make a 40-episode cell take hours and would not match how Zep is used
(agents do not wait for ingestion). The arm therefore waits a bounded ``READY_TIMEOUT_S``
per episode (charged as ingest wait), then continues; writes still unprocessed stay in the
pending list and are re-checked at the next read, and the number of unprocessed writes at
read time is recorded per episode as ``zep_lag_at_read`` so the ingestion lag is measured
rather than hidden.
"""

from __future__ import annotations

import time

from .. import config
from .base import EpisodeSession, MemoryArm, MemoryHit

READY_TIMEOUT_S = 60.0         # bounded per-episode readiness wait (charged to the arm)
SEED_READY_TIMEOUT_S = 900.0   # one-off bulk seeding before episode 1 (not charged)


class _ZepSession(EpisodeSession):
    def __init__(self, arm: "ZepArm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: ZepArm = arm

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        self._refresh_pending()
        self.acct.notes["zep_lag_at_read"] = max(self.acct.notes.get("zep_lag_at_read", 0), len(self.arm.pending))
        with self._timed():
            r = self.arm.client.graph.search(user_id=self.arm.user_id, query=query, limit=top_k)
        hits: list[MemoryHit] = []
        for e in (r.edges or []):
            if getattr(e, "invalid_at", None) or getattr(e, "expired_at", None):
                continue
            hits.append(MemoryHit(key=f"fact:{e.uuid_[:8]}", text=e.fact, score=float(getattr(e, "score", 0) or 0)))
        for n in (r.nodes or []):
            if n.summary:
                hits.append(MemoryHit(key=f"entity:{n.name[:24]}", text=f"{n.name}: {n.summary}",
                                      score=float(getattr(n, "score", 0) or 0)))
        if not hits:
            with self._timed():
                r2 = self.arm.client.graph.search(user_id=self.arm.user_id, query=query, limit=top_k,
                                                  scope="episodes")
            for ep in (r2.episodes or []):
                hits.append(MemoryHit(key=f"episode:{ep.uuid_[:8]}", text=ep.content, score=0.0))
        hits = hits[:top_k]
        self.acct.retrieved_bytes += sum(len(h.text) for h in hits)
        self.read_keys.extend(h.key for h in hits[:1])
        return hits

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        with self._timed():
            ep = self.arm.client.graph.add(user_id=self.arm.user_id, type="text", data=text,
                                           source_description=key)
        self.arm.pending.append(ep.uuid_)

    def _refresh_pending(self) -> None:
        """Drop pending writes Zep has finished processing. Zep processes a user's episodes
        roughly in order, so we pop from the front and stop at the first unprocessed one
        (bounded number of GETs even with a large seeding backlog)."""
        checked = 0
        while self.arm.pending and checked < 25:
            checked += 1
            try:
                if self.arm.client.graph.episode.get(self.arm.pending[0]).processed:
                    self.arm.pending.pop(0)
                    continue
            except Exception:  # noqa: BLE001
                pass
            break

    def wait_ready(self, _n: int | None = None, *, timeout: float | None = None, sample: int | None = None) -> None:
        t0 = time.perf_counter()
        deadline = t0 + (timeout or READY_TIMEOUT_S)
        while self.arm.pending and time.perf_counter() < deadline:
            uuid = self.arm.pending[0]
            try:
                if self.arm.client.graph.episode.get(uuid).processed:
                    self.arm.pending.pop(0)
                    continue
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0)
        self.acct.ingest_wait_ms += (time.perf_counter() - t0) * 1000
        if self.arm.pending:
            # not cleared: unprocessed writes carry over and are re-checked at the next read
            self.acct.notes["zep_unprocessed_at_deadline"] = len(self.arm.pending)

    def end(self, outcome, *, task_input, response_text, cited_keys) -> None:
        self.wait_ready()


class ZepArm(MemoryArm):
    name = "zep"

    def __init__(self) -> None:
        from zep_cloud.client import Zep

        self.client = Zep(api_key=config.env("ZEP_API_KEY", required=True))
        self.pending: list[str] = []

    def open(self, scope: str) -> None:
        super().open(scope)
        self.user_id = scope.replace("/", "-")
        try:
            self.client.user.delete(self.user_id)
        except Exception:  # noqa: BLE001
            pass
        self.client.user.add(user_id=self.user_id)

    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:
        """Bulk-load the initial store with graph.add_batch (Zep's documented path for
        pre-existing data), then wait for ingestion with the seeding deadline."""
        from zep_cloud.types import EpisodeData

        s = _ZepSession(self, agent_id, 0)
        t0 = time.perf_counter()
        try:
            for i in range(0, len(entries), 20):
                chunk = entries[i:i + 20]
                eps = self.client.graph.add_batch(
                    user_id=self.user_id,
                    episodes=[EpisodeData(data=text, type="text", source_description=key) for key, text, _ in chunk])
                self.pending.extend(e.uuid_ for e in eps)
            s.acct.ops += 1
            s.wait_ready(len(entries), timeout=SEED_READY_TIMEOUT_S)
        finally:
            s.close()
        self.seed_wait_ms = (time.perf_counter() - t0) * 1000

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _ZepSession(self, agent_id, episode)
