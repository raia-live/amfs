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

from amfs_core.models import SearchQuery, SemanticQuery
from amfs_core.scope import covers, descendants_range_sql, descendants_sql, normalize_scope


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


def _range_covers(prefix: str, path: str) -> bool:
    """Evaluate the range form's condition the way the database would.

    Postgres' ``~>=~`` and ``~<~`` compare the encoded bytes, whatever the
    database's collation says about the characters. Written from that, not
    from :func:`descendants_range_sql`, for the reason ``_like`` is.
    """
    clause, params = descendants_range_sql("entity_path", prefix)
    assert clause == "(entity_path = %s OR (entity_path ~>=~ %s AND entity_path ~<~ %s))"
    equals, lower, upper = params
    raw = path.encode("utf-8")
    return path == equals or lower.encode("utf-8") <= raw < upper.encode("utf-8")


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


class TestTheRangeForm:
    """The same rule stated as a byte-order range, for a b-tree to seek."""

    def test_it_names_the_column_it_was_given(self) -> None:
        clause, _ = descendants_range_sql("ae.entity_path", "a")
        assert clause == (
            "(ae.entity_path = %s OR (ae.entity_path ~>=~ %s AND ae.entity_path ~<~ %s))"
        )

    def test_the_path_travels_as_parameters_not_as_statement_text(self) -> None:
        clause, params = descendants_range_sql("entity_path", "a")
        assert clause.count("%s") == 3
        assert "'" not in clause
        assert params == ["a", "a/", "a0"]

    def test_the_range_is_everything_that_starts_with_the_prefix_and_a_separator(self) -> None:
        """``0`` is the byte after ``/``: the first string past the range."""
        assert ord("0") == ord("/") + 1
        _, (_, lower, upper) = descendants_range_sql("entity_path", "a/b/")
        assert (lower, upper) == ("a/b/", "a/b0")
        assert _range_covers("a/b", "a/b/")
        assert _range_covers("a/b", "a/b/\uffff")
        assert not _range_covers("a/b", "a/b0")
        assert not _range_covers("a/b", "a/b.")
        assert not _range_covers("a/b", "a/bc")

    def test_nothing_in_a_scope_is_a_pattern(self) -> None:
        """No wildcard to escape, and no escape character to get wrong."""
        assert _range_covers("a_b", "a_b/c")
        assert not _range_covers("a_b", "axb/c")
        assert _range_covers("a%b", "a%b/c")
        assert not _range_covers("a%b", "anything-else/c")
        assert _range_covers("a\\b", "a\\b/c")
        assert not _range_covers("a\\b", "a\\x/c")

    def test_a_scope_outside_ascii_is_compared_by_its_bytes(self) -> None:
        """Byte order, not the collation's idea of the alphabet."""
        assert _range_covers("café", "café/menu")
        assert not _range_covers("café", "cafe/menu")
        assert not _range_covers("café", "caféine/menu")
        assert not _range_covers("cafe", "café/menu")
        assert not _range_covers("A", "a/b")


class TestTheTwoFormsAgree:
    def test_the_sql_and_python_forms_answer_identically(self) -> None:
        """The drift guard, as in amfs_core.exclusions."""
        scopes = ["a", "a/b", "amfs", "amfs-internal", "a_b", "a%b", "a\\b", "a/b/", "café", "A"]
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
            "a/",
            "a0",
            "a.",
            "a/b/",
            "a/b0",
            "A/b",
            "café",
            "café/menu",
            "caféine",
            "cafe/menu",
            "a/\U0001f600",
        ]
        for scope in scopes:
            for path in paths:
                assert covers(scope, path) == _sql_covers(scope, path), (
                    f"forms disagree for scope={scope!r} path={path!r}"
                )
                assert covers(scope, path) == _range_covers(scope, path), (
                    f"range form disagrees for scope={scope!r} path={path!r}"
                )


class TestTheDefault:
    def test_the_form_existing_callers_get_is_the_one_they_had(self) -> None:
        """The range selects the same rows by a different plan, and for a
        nearest-neighbour search the plan decides what is found. So the form a
        caller gets without asking stays the LIKE."""
        clause, params = descendants_sql("entity_path", "a_b")
        assert clause == "(entity_path = %s OR entity_path LIKE %s)"
        assert params == ["a_b", "a\\_b/%"]

    def test_a_query_searches_everything_it_may_see_unless_it_asks_otherwise(self) -> None:
        """Naming the account leaves out what another account shares with the
        caller, so no existing caller may get it without asking."""
        from amfs_http.models import RetrieveRequest

        assert SearchQuery(entity_path="a").own_account_only is False
        assert SemanticQuery(text="t", entity_path="a").own_account_only is False
        assert RetrieveRequest(query="q").own_account_only is False

    def test_the_cross_encoder_ranks_unless_the_caller_declines_it(self) -> None:
        from amfs_http.models import RetrieveRequest

        assert RetrieveRequest(query="q").rerank is True

    def test_a_query_matches_one_path_unless_it_asks_otherwise(self) -> None:
        """An existing caller's results must not change under this feature."""
        assert SearchQuery(entity_path="a").include_descendants is False
        assert SemanticQuery(text="t", entity_path="a").include_descendants is False

    def test_the_retrieve_request_is_exact_unless_it_asks_otherwise(self) -> None:
        from amfs_http.models import RetrieveRequest

        assert RetrieveRequest(query="q", entity_path="a").include_descendants is False
