"""What it means for one entity path to cover another.

Entity paths are hierarchical — ``amfs-internal/deploy`` sits under
``amfs-internal`` — but nearly every read matches them by equality. That is the
right default for ``amfs_read``, where the caller names the entry it wants. It
is the wrong default for anything that opens a session, because the only scope
such a caller can derive without being told is the repository root, and no
entry is ever written there: the knowledge lives at ``<repo>/deploy`` and
``<repo>/ci-cd``. An exact match on the root reaches none of it and reports,
truthfully and uselessly, that nothing is known.

Covering is equality **or** a prefix that ends at a separator. ``a/b`` covers
``a/b`` and ``a/b/c``; it does not cover ``a/bc``, which is a different topic
that merely starts with the same letters. That distinction is the whole reason
this is not a bare ``LIKE 'a/b%'``.

One definition, in two forms
----------------------------
Filtering happens in SQL where an adapter has a query language and in Python
where it does not. Both forms live here so the two answers cannot drift apart —
the same reasoning, and the same failure, that produced :mod:`amfs_core.exclusions`.
The two are held equivalent by a test that runs one set of paths through both.

On escaping
-----------
The SQL form puts its wildcard in the *parameter*, never in the statement, so
the pattern is user data and ``%`` or ``_`` inside an entity path would
otherwise act as a wildcard: a scope of ``a_b`` would quietly also cover
``axb``. Those are escaped, and the escape character is escaped first, which is
the ordering that a later pass cannot undo. Postgres' default LIKE escape is
the backslash, so no ``ESCAPE`` clause is needed to read the result back.

The same rule as a range
------------------------
"Everything beneath ``a/b``" is every path that starts with ``a/b/``, and in
byte order those are exactly the paths from ``a/b/`` up to, but not including,
``a/b0``: ``0`` is the byte after ``/``, so ``a/b0`` is the first string that
no longer starts with ``a/b/``. :func:`descendants_range_sql` states the rule
that way, with Postgres' byte-order comparison operators.

It selects the same rows as :func:`descendants_sql`, and the tests hold the
two to each other. It is a separate form, used only where a caller asks for
it, because of what the planner does with each. A range is something a b-tree
over ``entity_path text_pattern_ops`` can seek; a ``LIKE`` whose pattern
arrives as a parameter is not, in a database whose collation is not ``C``. So
the two reach the same rows by different plans, and for a nearest-neighbour
search the plan is not neutral: read through the vector index and filtered
afterwards, a search sees the nearest rows of the whole table that happen to
lie beneath the path; read through the b-tree and sorted, it sees every row
beneath the path. Swapping one form for the other under an existing caller
would change what that caller is handed, so it is not swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SqlScope", "covers", "descendants_range_sql", "descendants_sql", "normalize_scope"]


@dataclass(frozen=True)
class SqlScope:
    """A visibility rule as a SQL predicate over ``amfs_memory_entries``.

    ``clause`` is a boolean expression that may reference the unqualified
    columns ``agent_id`` and ``entity_path`` and uses ``%s`` placeholders for
    every value in ``params``, in order. An adapter appends it to its own WHERE
    conditions with ``AND (clause)`` so the database filters the rows, instead
    of the caller materialising every row in the namespace and filtering them
    in Python — which is what a 100K-entry account turns into a 20-second
    request that holds the GIL for its neighbours.

    The clause is written by the layer that knows the rule (a per-user room
    visibility filter, say); the adapter treats it as a trusted fragment, the
    same way :func:`descendants_sql` treats its column name. Values never go in
    the clause, only in ``params``. ``None`` in an adapter signature means
    "no scope": the caller may see every row.
    """

    clause: str
    params: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # A tuple so the object is hashable and a caller's list is not shared.
        object.__setattr__(self, "params", tuple(self.params))

    def apply(self, conditions: list[str], params: list[Any]) -> None:
        """Append this scope to a WHERE being built as parallel lists."""
        conditions.append(f"({self.clause})")
        params.extend(self.params)

    @classmethod
    def all_of(cls, *scopes: SqlScope | None) -> SqlScope | None:
        """The conjunction of the given scopes, ignoring ``None``; ``None``
        when there is nothing to conjoin. A row must satisfy every rule."""
        present = [s for s in scopes if s is not None]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return cls(
            " AND ".join(f"({s.clause})" for s in present),
            tuple(p for s in present for p in s.params),
        )


def normalize_scope(prefix: str) -> str:
    """Strip trailing separators so ``a/b/`` and ``a/b`` mean one thing."""
    return prefix.rstrip("/")


def covers(prefix: str, path: str) -> bool:
    """Whether *path* is *prefix* itself or something beneath it.

    The Python form. Equality or a separator-terminated prefix, so ``a/b``
    covers ``a/b/c`` but never ``a/bc``.
    """
    root = normalize_scope(prefix)
    return path == root or path.startswith(root + "/")


def _escape_like(value: str) -> str:
    """Neutralise LIKE metacharacters in user-supplied path text.

    The backslash goes first: escaping it after ``%`` would also escape the
    backslashes just introduced, turning each wildcard back into live syntax.
    """
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def descendants_sql(column: str, prefix: str) -> tuple[str, list[str]]:
    """The SQL form: a condition and its parameters, for *column*.

    Returns an OR of equality and a separator-terminated prefix rather than a
    bare ``LIKE``, for the reason in the module docstring. *column* is a
    trusted identifier supplied by the caller, never user input; the prefix is
    user input and travels only as a parameter.
    """
    root = normalize_scope(prefix)
    return (
        f"({column} = %s OR {column} LIKE %s)",
        [root, _escape_like(root) + "/%"],
    )


_SEPARATOR = "/"
# The byte after the separator. ``<prefix>/`` to ``<prefix>0`` is the half-open
# range of every string that starts with ``<prefix>/``.
_AFTER_SEPARATOR = chr(ord(_SEPARATOR) + 1)


def descendants_range_sql(column: str, prefix: str) -> tuple[str, list[str]]:
    """The SQL form as a byte-order range: the same rows as
    :func:`descendants_sql`, stated so that a b-tree can seek them.

    An OR of equality and the range of everything beneath the prefix. The
    lower bound is the prefix with its separator, the upper bound the same
    with the separator replaced by the next byte, so the range holds exactly
    the paths that start with ``<prefix>/``. Nothing in it is a pattern, so
    there is nothing to escape: the path is compared, never interpreted.
    *column* is a trusted identifier supplied by the caller, never user input;
    the prefix is user input and travels only as parameters.

    Postgres only (``~>=~`` and ``~<~``), and not a drop-in replacement where
    the statement orders by vector distance: see the module docstring.
    """
    root = normalize_scope(prefix)
    return (
        f"({column} = %s OR ({column} ~>=~ %s AND {column} ~<~ %s))",
        [root, root + _SEPARATOR, root + _AFTER_SEPARATOR],
    )
