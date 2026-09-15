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
"""

from __future__ import annotations

__all__ = ["covers", "descendants_sql", "normalize_scope"]


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
