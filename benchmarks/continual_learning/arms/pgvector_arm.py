"""pgvector arms.

``pgvector``      — the plain RAG table most teams start with: one table, one embedding
                    column, cosine top-k, nothing else.
``pgvector-diy``  — what a competent team builds in a weekend on top of it: the same
                    table plus a recency tie-break and an LLM consolidation pass every
                    N episodes that compacts the agent's reflection notes into a single
                    "lessons learned" entry. Every token that pass spends is charged to
                    the arm so the paper can price the DIY route honestly.
``pgvector-diy+outcomes``
                  — the fair ceiling for DIY: the above plus a hand-rolled outcome
                    counter. Every entry the agent cited (or the top hit of each search
                    when it cited none) gets a win or a loss per attempt; hits render
                    their tally, entries with more losses than wins are demoted, and
                    the consolidation prompt sees the tallies. This is the honest answer
                    to "couldn't I just add a success column?" — and what it does *not*
                    do (credit split, surprise-scaled updates, contrast lessons, regime
                    detection, briefing sections, sealed traces) is the product's claim.

Both use a local in-process embedder (fastembed, bge-small-en-v1.5) so embedding cost
and latency are identical and near zero; the comparison is about what happens around
retrieval, not the embedder.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .. import config
from .base import ArmAccounting, EpisodeSession, MemoryArm, MemoryHit

_EMBEDDER = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding

        _EMBEDDER = TextEmbedding(config.EMBED_MODEL)
    return _EMBEDDER


def embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in embedder().embed(texts)]


DDL = """
CREATE TABLE IF NOT EXISTS cl_entries (
  id BIGSERIAL PRIMARY KEY,
  scope TEXT NOT NULL,
  key TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'experience',
  text TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (scope, key)
);
CREATE INDEX IF NOT EXISTS cl_entries_scope_idx ON cl_entries (scope);
CREATE TABLE IF NOT EXISTS cl_outcomes (
  scope TEXT NOT NULL,
  key TEXT NOT NULL,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  last_failed BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (scope, key)
);
"""


class _PgSession(EpisodeSession):
    def __init__(self, arm: "PgVectorArm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: PgVectorArm = arm

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        with self._timed():
            qv = embed([query])[0]
            with self.arm.conn.cursor() as cur:
                if self.arm.recency_boost:
                    cur.execute(
                        """
                        SELECT key, text, 1 - (embedding <=> %s::vector) AS sim,
                               EXTRACT(EPOCH FROM (now() - created_at)) AS age_s
                        FROM cl_entries WHERE scope = %s
                        ORDER BY embedding <=> %s::vector LIMIT %s
                        """,
                        (qv, self.arm.scope, qv, top_k * 3),
                    )
                    rows = cur.fetchall()
                    # recency tie-break: newest of the candidate pool gets +0.05, oldest +0
                    if rows:
                        ages = np.array([float(r[3]) for r in rows], dtype=float)  # EXTRACT returns Decimal
                        span = max(ages.max() - ages.min(), 1.0)
                        boosted = [(k, t, float(s) + 0.05 * (1 - (float(a) - ages.min()) / span))
                                   for k, t, s, a in rows]
                        boosted.sort(key=lambda r: -r[2])
                        rows = boosted[:top_k]
                else:
                    cur.execute(
                        """
                        SELECT key, text, 1 - (embedding <=> %s::vector) AS sim
                        FROM cl_entries WHERE scope = %s
                        ORDER BY embedding <=> %s::vector LIMIT %s
                        """,
                        (qv, self.arm.scope, qv, top_k),
                    )
                    rows = cur.fetchall()
        hits = [MemoryHit(key=r[0], text=r[1], score=float(r[2])) for r in rows]
        if self.arm.track_outcomes:
            hits = self._with_outcomes(hits, top_k)
        self.acct.retrieved_bytes += sum(len(h.text) for h in hits)
        self.read_keys.extend(h.key for h in hits[:1])
        return hits

    # -- DIY outcome counter (pgvector-diy+outcomes only) ---------------------------------
    def _with_outcomes(self, hits: list[MemoryHit], top_k: int) -> list[MemoryHit]:
        if not hits:
            return hits
        with self.arm.conn.cursor() as cur:
            cur.execute("SELECT key, wins, losses, last_failed FROM cl_outcomes WHERE scope = %s AND key = ANY(%s)",
                        (self.arm.scope, [h.key for h in hits]))
            tally = {k: (w, l, lf) for k, w, l, lf in cur.fetchall()}
        for h in hits:
            w, l, _ = tally.get(h.key, (0, 0, False))
            h.success_count, h.failure_count = w, l
            h.evidence_status = ("untested" if not (w or l) else "discredited" if l > w else
                                 "contested" if l else "validated")
            # a naive "success column" demotion: each net loss costs a tenth of a cosine point
            h.score += 0.1 * (w - l) if (w or l) else 0.0
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def _tally(self, keys: list[str], won: bool) -> None:
        keys = [k for k in dict.fromkeys(keys) if k]
        if not keys:
            return
        with self._timed(), self.arm.conn.cursor() as cur:
            for k in keys:
                cur.execute(
                    """
                    INSERT INTO cl_outcomes (scope, key, wins, losses, last_failed)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (scope, key) DO UPDATE
                      SET wins = cl_outcomes.wins + EXCLUDED.wins,
                          losses = cl_outcomes.losses + EXCLUDED.losses,
                          last_failed = EXCLUDED.last_failed
                    """,
                    (self.arm.scope, k, int(won), int(not won), not won),
                )
            self.arm.conn.commit()

    def end(self, outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        if not self.arm.track_outcomes:
            return
        self._tally(cited_keys or self.read_keys, outcome.success)

    def attempt_failed(self, attempt: int, severity: str, cited_keys: list[str], answer: str) -> None:
        if not self.arm.track_outcomes:
            return
        self._tally(cited_keys or self.read_keys[-1:], False)

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        with self._timed():
            v = embed([text])[0]
            with self.arm.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cl_entries (scope, key, kind, text, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (scope, key) DO UPDATE
                      SET text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                          kind = EXCLUDED.kind, created_at = now()
                    """,
                    (self.arm.scope, key, kind, text, v),
                )
            self.arm.conn.commit()


class PgVectorArm(MemoryArm):
    name = "pgvector"
    recency_boost = False
    consolidate = False
    track_outcomes = False

    def __init__(self) -> None:
        import psycopg

        self.conn = psycopg.connect(config.PG_DSN)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(DDL)
        self.conn.commit()

    def open(self, scope: str) -> None:
        super().open(scope)
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM cl_entries WHERE scope = %s", (scope,))
            cur.execute("DELETE FROM cl_outcomes WHERE scope = %s", (scope,))
        self.conn.commit()

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _PgSession(self, agent_id, episode)

    def close(self) -> None:
        self.conn.close()


CONSOLIDATE_PROMPT = """You maintain the long-term notes of an operations agent.
Below are the agent's raw reflection notes from recent runs. Rewrite them into ONE
compact "lessons learned" document: keep every concrete, actionable fact (which
procedure/value/parameter worked or failed for which service), drop duplicates and
chatter, and where notes conflict keep the most recent statement and say that an
earlier note disagreed. Where a note carries an outcome tally ([N won / M failed]), keep
what won, and state explicitly that a note with more failures than wins should not be
followed. Output plain text, no preamble.

NOTES:
{notes}
"""


class PgVectorDiyArm(PgVectorArm):
    name = "pgvector-diy"
    recency_boost = True
    consolidate = True

    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:
        self._seed_keys = {k for k, _, _ in entries}
        super().seed(entries, agent_id=agent_id)

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        # runs after episodes 5, 10, 15, 20 (1-based)
        if (episode + 1) % config.STUDY.diy_consolidate_every != 0:
            return None
        acct = ArmAccounting()
        seed_keys = getattr(self, "_seed_keys", set())
        with self.conn.cursor() as cur:
            cur.execute("SELECT key, text FROM cl_entries WHERE scope = %s ORDER BY created_at", (self.scope,))
            notes = [(k, t) for k, t in cur.fetchall() if k not in seed_keys]
        if len(notes) < 2:
            return None
        t0 = time.perf_counter()
        tally: dict[str, tuple[int, int]] = {}
        if self.track_outcomes:
            with self.conn.cursor() as cur:
                cur.execute("SELECT key, wins, losses FROM cl_outcomes WHERE scope = %s", (self.scope,))
                tally = {k: (w, l) for k, w, l in cur.fetchall()}
        def _t(k: str) -> str:
            w, l = tally.get(k, (0, 0))
            return f" [{w} won / {l} failed]" if (w or l) else ""
        body = "\n".join(f"- ({k}){_t(k)} {t}" for k, t in notes)
        text, usage = llm.complete_text(CONSOLIDATE_PROMPT.format(notes=body), max_tokens=700)
        acct.extra_prompt_tokens += usage.prompt_tokens
        acct.extra_completion_tokens += usage.completion_tokens
        acct.extra_cost_usd += usage.cost_usd
        v = embed([text])[0]
        with self.conn.cursor() as cur:
            if seed_keys:
                cur.execute("DELETE FROM cl_entries WHERE scope = %s AND NOT (key = ANY(%s))",
                            (self.scope, list(seed_keys)))
            else:
                cur.execute("DELETE FROM cl_entries WHERE scope = %s AND kind = 'experience'", (self.scope,))
            cur.execute(
                "INSERT INTO cl_entries (scope, key, kind, text, embedding) VALUES (%s,%s,'experience',%s,%s::vector) "
                "ON CONFLICT (scope, key) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding, created_at=now()",
                (self.scope, f"lessons-learned-ep{episode}", text, v),
            )
        self.conn.commit()
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        acct.ops = 1
        acct.notes = {"consolidated_notes": len(notes), "chars": len(text)}
        return acct


class PgVectorDiyOutcomesArm(PgVectorDiyArm):
    """``pgvector-diy`` plus a hand-rolled outcome counter (see module docstring)."""

    name = "pgvector-diy+outcomes"
    learns_from_outcomes = True
    track_outcomes = True
