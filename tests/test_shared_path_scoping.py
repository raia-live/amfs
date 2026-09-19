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

import ast
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
    """The source of one method (or module-level function), up to the next
    def at the same indent."""
    src = path.read_text()
    for indent in ("    ", ""):
        for prefix in (f"{indent}def {name}(", f"{indent}async def {name}("):
            marker = f"\n{prefix}"
            if marker in src:
                start = src.index(marker) + 1
                rest = src[start + 10 :]
                end = re.search(rf"\n{indent}(?:async )?def ", rest)
                return src[start : start + 10 + (end.start() if end else len(rest))]
    raise AssertionError(f"{path.name} has no def {name}")


#: Where a read builds its WHERE clause when that is not the method itself.
#: ``list`` and ``count_entries`` on both adapters share the module-level
#: ``list_conditions`` in adapter.py, so a page and the ``total`` it reports
#: cannot disagree about which rows exist and the guard is applied to the
#: unscoped read in one place; these tests follow it there.
WHERE_BUILDER: dict[tuple[Path, str], tuple[Path, str]] = {
    (SYNC, "list"): (SYNC, "list_conditions"),
    (ASYNC, "list"): (SYNC, "list_conditions"),
}


def _where_body(path: Path, method: str) -> str:
    """The source that decides which rows *method* reads."""
    return _method(*WHERE_BUILDER.get((path, method), (path, method)))


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
        """Two copies of a security rule is one copy that gets forgotten.

        Parsed rather than string-matched, because the substring version of this
        test was a false negative for as long as the import stayed on one line
        and then silently for as long as it did not: wrapping it in parentheses
        -- which any formatter will do once a fourth name is added -- made the
        assertion unsatisfiable while the code was still correct. A security
        guard whose test fails for cosmetic reasons gets its test relaxed.
        """
        imported = {
            alias.name
            for node in ast.walk(ast.parse(ASYNC.read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.module == "amfs_postgres.adapter"
            for alias in node.names
        }
        assert GUARD in imported, (
            f"{GUARD} is not imported from amfs_postgres.adapter; the async "
            f"adapter has either lost the guard or grown its own copy"
        )

        assigned = {
            target.id
            for node in ast.walk(ast.parse(ASYNC.read_text()))
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert GUARD not in assigned, "a second definition of the guard"


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
        body = _where_body(path, method)
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
        body = _where_body(path, method)
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
