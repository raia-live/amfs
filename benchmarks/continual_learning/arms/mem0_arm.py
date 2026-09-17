"""Mem0 (hosted platform) arm, wired as documented.

Writes go through ``add()`` with inference ON, which is Mem0's mechanism: an LLM extracts
facts from the message and decides ADD / UPDATE / DELETE against existing memories at
write time. That is the write-time reconciliation the paper contrasts with outcome-time
reconciliation, so it is deliberately not bypassed with ``infer=False``.

Ingestion is asynchronous. After each write the arm polls until the memory is
searchable (bounded) and records the wait separately from agent latency.
"""

from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import config
from .base import SEED_READY_TIMEOUT_S, EpisodeSession, MemoryArm, MemoryHit

READY_TIMEOUT_S = 25.0


class _PacedClient:
    """Proxy around ``MemoryClient`` that paces calls process-wide (MEM0_RPM, default 240)
    and retries Mem0's 'requests too frequently' RateLimitError with back-off. Six cells
    seeding in parallel tripped the burst limit on the Pro plan within a minute."""

    def __init__(self, client) -> None:
        self._c = client
        self._interval = 60.0 / float(os.environ.get("MEM0_RPM", "240"))
        self._lock = threading.Lock()
        self._next = 0.0

    def _pace(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)

    def __getattr__(self, name):
        fn = getattr(self._c, name)
        if not callable(fn):
            return fn

        def call(*a, **kw):
            for attempt in range(7):
                self._pace()
                try:
                    return fn(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if "too frequently" in msg or "Rate limit exceeded" in msg or "429" in msg:
                        if "quota" in msg.lower():
                            raise  # monthly quota: no point retrying
                        if attempt < 6:
                            time.sleep(min(5.0 * (2 ** attempt), 90.0) + random.uniform(0, 2))
                            continue
                    raise
            raise RuntimeError("unreachable")
        return call


class _Mem0Session(EpisodeSession):
    def __init__(self, arm: "Mem0Arm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: Mem0Arm = arm
        self.hit_ids: dict[str, str] = {}

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        with self._timed():
            res = self.arm.client.search(query, filters={"user_id": self.arm.user_id},
                                         version="v2", top_k=top_k)
        rows = res.get("results", res) if isinstance(res, dict) else res
        hits = []
        for r in rows or []:
            key = (r.get("metadata") or {}).get("key") or r.get("id", "")[:8]
            hits.append(MemoryHit(key=key, text=r.get("memory", ""), score=float(r.get("score") or 0)))
            if r.get("id"):
                self.hit_ids[key] = r["id"]
        self.acct.retrieved_bytes += sum(len(h.text) for h in hits)
        self.read_keys.extend(h.key for h in hits[:1])
        return hits

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        t_write = datetime.now(timezone.utc)
        with self._timed():
            self.arm.client.add([{"role": "user", "content": text}], user_id=self.arm.user_id,
                                metadata={"key": key, "kind": kind})
        self.arm.pending.append((key, text, t_write))

    def wait_ready(self, _n: int | None = None, *, timeout: float | None = None, sample: int | None = None) -> None:
        """Block until pending writes have landed (or time out), charging the wait.

        Mem0's inference may ADD a new memory (key-tagged), or UPDATE/merge into an existing
        one (no new key). Either counts as landed: a hit tagged with our key, or any hit
        whose ``updated_at`` is after the write. ``sample`` checks only the last N pending
        writes (bulk seeding)."""
        t0 = time.perf_counter()
        deadline = t0 + (timeout or READY_TIMEOUT_S)
        todo = list(self.arm.pending)[-sample:] if sample else list(self.arm.pending)
        for key, text, t_write in todo:
            # Mem0 lands in ~5 s in pilots; sleep first, then poll every 3 s so each write costs
            # one or two billed retrievals rather than a tight loop of them.
            time.sleep(max(0.0, min(4.0, deadline - time.perf_counter())))
            while time.perf_counter() < deadline:
                res = self.arm.client.search(text[:200], filters={"user_id": self.arm.user_id},
                                             version="v2", top_k=5)
                rows = res.get("results", res) if isinstance(res, dict) else res
                if any(_landed(r, key, t_write) for r in rows or []):
                    break
                time.sleep(3.0)
            else:
                self.acct.notes["mem0_unverified_writes"] = self.acct.notes.get("mem0_unverified_writes", 0) + 1
        self.arm.pending.clear()
        self.acct.ingest_wait_ms += (time.perf_counter() - t0) * 1000

    def end(self, outcome, *, task_input, response_text, cited_keys) -> None:
        # Mem0's documented outcome signal: per-memory feedback. Given for every memory the
        # agent retrieved this episode, so the arm gets every signal the product offers.
        # (Mem0 documents it as a quality signal; it is not documented to re-rank retrieval.)
        fb = "POSITIVE" if outcome.success else "NEGATIVE"
        for key, mid in list(self.hit_ids.items())[:10]:
            try:
                self.arm.client.feedback(memory_id=mid, feedback=fb, feedback_reason=outcome.summary[:200])
            except Exception as e:  # noqa: BLE001
                self.acct.notes["mem0_feedback_error"] = str(e)[:120]
                break
        self.wait_ready()


def _landed(row: dict, key: str, t_write: datetime) -> bool:
    if ((row.get("metadata") or {}).get("key") == key):
        return True
    ts = row.get("updated_at") or row.get("created_at")
    if not ts:
        return False
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t >= t_write - timedelta(seconds=5)
    except ValueError:
        return False


class Mem0Arm(MemoryArm):
    name = "mem0"

    def __init__(self) -> None:
        from mem0 import MemoryClient

        self.client = _PacedClient(MemoryClient(api_key=config.env("MEM0_API_KEY", required=True)))
        self.pending: list[tuple[str, str, datetime]] = []

    def open(self, scope: str) -> None:
        super().open(scope)
        self.user_id = scope.replace("/", "-")
        try:
            self.client.delete_all(user_id=self.user_id)
        except Exception:  # noqa: BLE001
            pass

    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:
        """Pre-load with parallel ``add`` calls (Mem0 has no bulk endpoint), then a sampled
        readiness check with the seeding deadline. Not charged to episodes."""
        from concurrent.futures import ThreadPoolExecutor

        s = _Mem0Session(self, agent_id, 0)

        def _do(e):
            key, text, _ = e
            t_write = datetime.now(timezone.utc)
            self.client.add([{"role": "user", "content": text}], user_id=self.user_id,
                            metadata={"key": key, "kind": "fact"})
            return key, text, t_write

        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                self.pending.extend(ex.map(_do, entries))
            s.wait_ready(len(entries), timeout=SEED_READY_TIMEOUT_S, sample=3)
        finally:
            s.close()

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _Mem0Session(self, agent_id, episode)
