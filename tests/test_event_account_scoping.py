"""Event-log aggregates are scoped to one account.

Every other query in the adapter is isolated by row-level security, so a
missing account filter is invisible in review — the RLS policy is assumed to
be there. `amfs_events` has no RLS and no policies at all, which makes it the
one table where the filter has to be written by hand.

The weekly reuse deltas in `stats_extended` were missing it. The counts feed
the "Rework avoided"/"Memories reused" trend on the dashboard, so every
account was shown the whole deployment's reuse: measured on production, an
account whose real trend was +106% rendered as +472%.

These are SQL-shape tests. The failure is silent — a leak in a number that
still looks entirely plausible — so nothing will notice if the filter is
dropped in a refactor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / (
    "packages/adapters/postgres/src/amfs_postgres"
)
SYNC = SRC / "adapter.py"


def _method(path: Path, name: str) -> str:
    """The source of one method, up to the next def at the same indent."""
    src = path.read_text()
    start = src.index(f"    def {name}(")
    rest = src[start + 10 :]
    end = re.search(r"\n    (?:async )?def ", rest)
    return src[start : start + 10 + (end.start() if end else len(rest))]


class TestTheEventsTableHasNoRlsToFallBackOn:
    def test_no_migration_enables_rls_on_amfs_events(self) -> None:
        """If this ever fails, RLS now covers the table and the hand-written
        filters below stop being the only thing standing between one account
        and another's timeline — re-read them before relaxing anything."""
        sql = [
            p.read_text()
            for p in (Path(__file__).resolve().parents[1]).rglob("*.sql")
        ]
        enabled = [
            s for s in sql
            if re.search(r"amfs_events\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY", s, re.I)
        ]
        assert not enabled, (
            "amfs_events now has RLS; the explicit account filters may be "
            "redundant, but verify before removing them"
        )


class TestWeeklyReuseDeltasAreScoped:
    def test_the_event_query_filters_by_account(self) -> None:
        body = _method(SYNC, "stats_extended")
        assert 'event_conditions.append("account_id = %s")' in body, (
            "stats_extended counts read events without scoping them to the "
            "current account — every account sees the whole deployment"
        )

    def test_it_fails_closed_without_a_tenant_context(self) -> None:
        """The dangerous branch is the one where the context is missing: an
        omitted filter there matches every row in the table, which is the leak
        arriving by a different route. Rows are written with the account id of
        whoever logged them, so a context-less caller is asking about the
        untenanted (NULL) rows and nothing else."""
        body = _method(SYNC, "stats_extended")
        assert re.search(
            r'event_conditions\.append\("account_id = %s"\)\s*\n'
            r'\s*event_params\.append\([^)]+\)\s*\n'
            r'\s*else:\s*\n'
            r'\s*event_conditions\.append\("account_id IS NULL"\)',
            body,
        ), "stats_extended does not fall back to account_id IS NULL"

    def test_the_filter_reaches_the_query(self) -> None:
        """The conditions are joined into `event_where` and interpolated; a
        filter appended to a list that the SQL never uses would pass the
        assertions above and still leak."""
        body = _method(SYNC, "stats_extended")
        assert 'event_where = " AND ".join(event_conditions)' in body
        assert "WHERE {event_where}" in body
        assert "FROM amfs_events" in body


class TestTheTimelineStaysScoped:
    @pytest.mark.parametrize("method", ["list_events", "get_event"])
    def test_event_readers_filter_by_account(self, method) -> None:
        """These return whole event rows rather than counts, so the same
        omission would disclose another account's agent names and summaries."""
        body = _method(SYNC, method)
        if "_event_conditions(" in body:
            # The WHERE clause is built by a helper shared with the async adapter;
            # the filter has to be in there, and the method has to pass the account.
            assert "account_id=self._get_current_account_id()" in body, (
                f"{method} does not pass the current account to _event_conditions"
            )
            body += _method(SYNC, "_event_conditions")
        assert "account_id = %s" in body, (
            f"{method} reads amfs_events without an account filter"
        )
