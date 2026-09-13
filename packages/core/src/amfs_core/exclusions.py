"""Which entity paths and agent identities belong to the system rather than a user.

Benchmarks and internal scratch work share the same tables as real memory. That
is deliberate — a benchmark that wrote somewhere else would not be exercising
the thing it measures — but it means every aggregate over those tables counts
them unless it is told not to, and an account's own totals are the one place
where the system's own rows are simply wrong. A user reading "1,204 memories"
should be reading the number of memories they have.

The rules are prefix rules, matched against the leading segment, and narrow on
purpose: ``bench-`` and ``bench/`` are excluded, ``benchmarking-agent`` is not,
because the second is a name a user is entitled to give their own agent without
disappearing from their own dashboard. Same for ``_system`` against
``_systemic``.

One definition, in two forms
----------------------------
Aggregation happens in SQL where an adapter can express it and in Python where
it cannot — a room's per-user visibility semantics, for instance, or an adapter
with no query language. Both forms live here so that the two answers cannot
drift apart, which is the failure this module was extracted to end: the pattern
existed in four places, three of them carrying a comment asking whoever edited
one to remember the others.

The two forms are held equivalent by a test that runs the same paths through
both.
"""

from __future__ import annotations

import re

#: Entity paths that are the system's own workspace.
EXCLUDED_ENTITY_RE = re.compile(r"^(?:_system(?:/|$)|_?bench[-/])", re.I)

#: Agent identities that are benchmark harnesses rather than someone's agent.
EXCLUDED_AGENT_RE = re.compile(r"^_?bench[-/]", re.I)

#: Identities the server itself writes under.
SYSTEM_AGENT_IDS = frozenset({"amfs-server", "system", "amfs"})

#: The same two rules as SQL predicates, for adapters that aggregate in the
#: database. POSIX case-insensitive (``!~*``) stands in for ``re.I``. No ``%``
#: appears in either, so they need no doubling when psycopg is given
#: parameters alongside them.
ENTITY_PATH_NOT_EXCLUDED_SQL = r"entity_path !~* '^(_system(/|$)|_?bench[-/])'"
AGENT_ID_NOT_EXCLUDED_SQL = (
    r"agent_id !~* '^_?bench[-/]' "
    r"AND agent_id NOT IN ('amfs-server', 'system', 'amfs')"
)


def is_excluded_entity(entity_path: str | None) -> bool:
    """Whether *entity_path* is the system's own rather than a user's."""
    return bool(entity_path) and bool(EXCLUDED_ENTITY_RE.match(entity_path or ""))


def is_excluded_agent(agent_id: str | None) -> bool:
    """Whether *agent_id* is a benchmark harness or the server itself."""
    if not agent_id:
        return False
    return bool(EXCLUDED_AGENT_RE.match(agent_id)) or agent_id in SYSTEM_AGENT_IDS


def is_excluded_entry(entry: object) -> bool:
    """Whether an entry should be left out of an account's own totals.

    Either rule is enough. A benchmark writing to a real-looking path is still
    the benchmark's row, and a real agent writing into ``_system`` is still the
    system's.
    """
    path = getattr(entry, "entity_path", None)
    provenance = getattr(entry, "provenance", None)
    agent_id = getattr(provenance, "agent_id", None)
    return is_excluded_entity(path) or is_excluded_agent(agent_id)
