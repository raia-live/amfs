"""Shared namespaces are returned only when they are asked for by name.

An entity path of the form @<name>/<topic> is a shared namespace: its
contents belong to a group, and people outside the account reading them can
write to it. That is the feature. The hazard is where those writes end up.

An unscoped query — search with no entity_path, list() with no argument, the
retrieval that backs "what do I know about this" — puts its results into an
agent's context without anyone having chosen them. If shared content can
land there, then anyone who can write to a shared namespace can put text in
front of an agent that is working on something else entirely. They do not
need to breach anything; they write an ordinary memory and wait. Naming the
path is the act of deciding to trust it, so scoped reads are untouched and
only the ambient ones filter.

These are SQL-shape tests. The behaviour they protect is invisible in normal
use and the failure is silent, so the guard is easy to drop in a refactor
and nothing will notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / (
    "packages/adapters/postgres/src/amfs_postgres"
)
SYNC = SRC / "adapter.py"
ASYNC = SRC / "async_adapter.py"

GUARD = "_EXCLUDE_SHARED_PATHS"


def _method(path: Path, name: str) -> str:
    """The source of one method, up to the next def at the same indent."""
    src = path.read_text()
    start = src.index(f"    def {name}(") if f"    def {name}(" in src else src.index(
        f"    async def {name}("
    )
    rest = src[start + 10 :]
    end = re.search(r"\n    (?:async )?def ", rest)
    return src[start : start + 10 + (end.start() if end else len(rest))]


class TestTheGuardIsDefinedOnce:
    def test_it_matches_prefixed_paths_only(self) -> None:
        """A bare '@name' with no topic is not a shared namespace, and an
        ordinary path that merely contains an @ must not be caught."""
        src = SYNC.read_text()
        assert "entity_path NOT LIKE '@%%/%%'" in src

    def test_its_wildcards_are_escaped_for_psycopg(self) -> None:
        """psycopg reads '%' as the start of a placeholder on any query run
        with parameters, so a single '%' here raises ProgrammingError the
        moment an unscoped query also binds a value — which every one of them
        does. The doubling is load-bearing, not style: consecutive LIKE
        wildcards collapse, so the escaped form means the same thing on the
        paths where psycopg passes the string through untouched.
        """
        line = next(
            ln for ln in SYNC.read_text().splitlines()
            if ln.startswith(f"{GUARD} =")
        )
        assert "%" in line, "the guard no longer uses a LIKE pattern"
        assert not re.search(r"(?<!%)%(?!%)", line), (
            f"unescaped '%' in {line.strip()} — psycopg will reject any "
            f"unscoped query that binds parameters"
        )

    def test_the_async_adapter_shares_the_definition(self) -> None:
        """Two copies of a security rule is one copy that gets forgotten."""
        assert f"from amfs_postgres.adapter import {GUARD}" in ASYNC.read_text()
        assert f'{GUARD} = ' not in ASYNC.read_text()


class TestUnscopedReadsExcludeSharedNamespaces:
    @pytest.mark.parametrize(
        "path,method",
        [
            (SYNC, "list"),
            (SYNC, "search"),
            (SYNC, "semantic_search"),
            (ASYNC, "list"),
            (ASYNC, "search"),
            (ASYNC, "semantic_search"),
        ],
    )
    def test_the_guard_is_applied_when_no_path_is_given(self, path, method) -> None:
        body = _method(path, method)
        assert GUARD in body, (
            f"{path.name}:{method} can return shared-namespace entries to a "
            f"query that never asked for them"
        )

    @pytest.mark.parametrize(
        "path,method",
        [
            (SYNC, "list"),
            (SYNC, "search"),
            (SYNC, "semantic_search"),
            (ASYNC, "list"),
            (ASYNC, "search"),
            (ASYNC, "semantic_search"),
        ],
    )
    def test_it_is_the_else_of_the_entity_path_filter(self, path, method) -> None:
        """Applying it unconditionally would break the scoped read too, which
        is the one case that must keep working: naming the path is how a
        caller says they want it."""
        body = _method(path, method)
        assert re.search(
            r'conditions\.append\("entity_path = %s"\)\s*\n'
            r'\s*params\.append\([^)]+\)\s*\n'
            r'\s*else:\s*\n'
            r'\s*conditions\.append\(' + GUARD + r'\)',
            body,
        ), f"{path.name}:{method} does not gate the guard on the path being absent"

    def test_entity_summaries_is_always_filtered(self) -> None:
        """It takes no entity_path at all and groups by it, so a shared
        namespace's topics would be listed as though they were this
        account's own."""
        assert GUARD in _method(SYNC, "entity_summaries")

    @pytest.mark.parametrize("method", ["stats", "stats_extended"])
    def test_the_stats_aggregates_are_always_filtered(self, method) -> None:
        """Same shape as entity_summaries — no entity_path to opt in with, and
        a breakdown that groups by entity_path. Shared entries would both name
        their topics and inflate every total the stats page shows."""
        assert GUARD in _method(SYNC, method)


class TestScopedReadsAreUntouched:
    def test_read_does_not_filter(self) -> None:
        """read() names both the path and the key. That is as explicit as a
        request gets, and filtering it would make shared rooms unreadable."""
        assert GUARD not in _method(SYNC, "read")

    def test_read_at_version_does_not_filter(self) -> None:
        assert GUARD not in _method(SYNC, "read_at_version")
