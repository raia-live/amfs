"""What one entity path covers, in both of the forms that answer it.

Reads match entity paths by equality nearly everywhere, which is correct when
the caller names the entry it wants and wrong when the caller could only derive
a scope. A session opener knows the repository it was started in and nothing
finer, and no entry is ever written at a repository root — the knowledge sits at
``<repo>/deploy`` and ``<repo>/ci-cd``. Matched exactly, a briefing on the root
reports that nothing is known, which is true of the root and false of the repo.

The interesting cases are not "the prefix matches". They are the near miss and
the escaping. ``a/b`` must not cover ``a/bc``, a different topic that happens to
share letters, which is why the rule is equality-or-prefix-with-separator and
not ``LIKE 'a/b%'``. And because the SQL form's wildcard travels as a parameter
built from user text, a scope of ``a_b`` must not quietly also cover ``axb``.

Following amfs_core.exclusions, the SQL and Python forms are held equivalent
here by running one set of paths through both, so they cannot drift apart. The
LIKE emulator below is deliberately written from the Postgres semantics rather
than from the implementation it checks.
"""

from __future__ import annotations

import re

from amfs_core.models import SearchQuery
from amfs_core.scope import covers, descendants_sql, normalize_scope


def _like(pattern: str, value: str) -> bool:
    """Postgres ``LIKE`` with its default backslash escape, as a regex.

    Written independently of :func:`descendants_sql` on purpose: a shared helper
    would make the equivalence test agree with itself rather than with Postgres.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            i += 1
            if i < len(pattern):
                out.append(re.escape(pattern[i]))
        elif char == "%":
            out.append(".*")
        elif char == "_":
            out.append(".")
        else:
            out.append(re.escape(char))
        i += 1
    return re.fullmatch("".join(out), value, re.S) is not None


def _sql_covers(prefix: str, path: str) -> bool:
    """Evaluate the SQL form's condition the way the database would."""
    clause, params = descendants_sql("entity_path", prefix)
    assert clause == "(entity_path = %s OR entity_path LIKE %s)"
    equals, like = params
    return path == equals or _like(like, path)


class TestWhatCoveringMeans:
    def test_a_scope_covers_itself(self) -> None:
        assert covers("amfs-internal", "amfs-internal")

    def test_a_scope_covers_the_topics_beneath_it(self) -> None:
        """The case the whole change exists for."""
        assert covers("amfs-internal", "amfs-internal/deploy")
        assert covers("amfs-internal", "amfs-internal/ci-cd")

    def test_a_scope_covers_arbitrary_depth(self) -> None:
        assert covers("a", "a/b/c/d")

    def test_a_scope_does_not_cover_a_name_that_merely_starts_the_same(self) -> None:
        """The near miss, and the reason this is not a bare prefix LIKE.

        ``amfs`` and ``amfs-internal`` are two different repositories. A briefing
        on one must not quietly serve the other's memory.
        """
        assert not covers("amfs", "amfs-internal")
        assert not covers("a/b", "a/bc")

    def test_a_deeper_scope_does_not_cover_its_parent(self) -> None:
        assert not covers("a/b", "a")

    def test_unrelated_paths_are_not_covered(self) -> None:
        assert not covers("a/b", "x/y")

    def test_a_trailing_separator_means_the_same_scope(self) -> None:
        assert covers("a/b/", "a/b")
        assert covers("a/b/", "a/b/c")
        assert normalize_scope("a/b/") == "a/b"


class TestTheSqlForm:
    def test_it_names_the_column_it_was_given(self) -> None:
        clause, _ = descendants_sql("ae.entity_path", "a")
        assert clause == "(ae.entity_path = %s OR ae.entity_path LIKE %s)"

    def test_the_wildcard_travels_as_a_parameter_not_as_statement_text(self) -> None:
        """So the scope is data, and a `%` in it cannot become syntax."""
        clause, params = descendants_sql("entity_path", "a")
        assert "%s" in clause
        assert "'" not in clause
        assert params == ["a", "a/%"]

    def test_a_scope_containing_a_wildcard_is_escaped(self) -> None:
        """Unescaped, a scope of ``a_b`` would also cover ``axb``."""
        assert _sql_covers("a_b", "a_b/c")
        assert not _sql_covers("a_b", "axb/c")

        assert _sql_covers("a%b", "a%b/c")
        assert not _sql_covers("a%b", "anything-else/c")

    def test_the_escape_character_is_escaped_before_the_wildcards(self) -> None:
        """Ordering, which a later pass cannot undo.

        Escaping ``%`` first and the backslash second would re-escape the
        backslashes just introduced and turn each wildcard back into syntax.
        """
        assert _sql_covers("a\\b", "a\\b/c")
        assert not _sql_covers("a\\b", "a\\x/c")


class TestTheTwoFormsAgree:
    def test_the_sql_and_python_forms_answer_identically(self) -> None:
        """The drift guard, as in amfs_core.exclusions."""
        scopes = ["a", "a/b", "amfs", "amfs-internal", "a_b", "a%b", "a\\b", "a/b/"]
        paths = [
            "a",
            "a/b",
            "a/bc",
            "a/b/c",
            "a/b/c/d",
            "amfs",
            "amfs-internal",
            "amfs-internal/deploy",
            "a_b",
            "a_b/c",
            "axb",
            "axb/c",
            "a%b",
            "a%b/c",
            "a\\b",
            "a\\b/c",
            "a\\x/c",
            "x/y",
            "",
        ]
        for scope in scopes:
            for path in paths:
                assert covers(scope, path) == _sql_covers(scope, path), (
                    f"forms disagree for scope={scope!r} path={path!r}"
                )


class TestTheDefault:
    def test_a_query_matches_one_path_unless_it_asks_otherwise(self) -> None:
        """An existing caller's results must not change under this feature."""
        assert SearchQuery(entity_path="a").include_descendants is False
