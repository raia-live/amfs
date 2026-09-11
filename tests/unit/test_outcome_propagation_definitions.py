"""The three definitions of ``amfs_propagate_outcome`` must stay in step.

This function is defined in three places in this package, and every one of them
is a ``CREATE OR REPLACE``:

  * ``schema.sql``, applied when a database is bootstrapped;
  * ``adapter.py``'s ``_apply_migrations``, applied at startup whenever the
    schema fingerprint changes;
  * ``migrations/007_outcome_propagation_row_copy.sql``, applied by whoever runs
    the migration files.

Whichever ran last silently *is* the function. That makes a stale copy not dead
code but a regression waiting for the next deploy, and it is exactly what
happened: ``schema.sql`` kept the pre-006 inverted multipliers long after 006
corrected them, so a success eroded confidence whenever the bootstrap ran last.

A behavioural test cannot catch this, because it only ever exercises whichever
definition won in the test database. So these are source tests, and the thing
they assert is agreement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2] / "packages/adapters/postgres/src/amfs_postgres"

_SOURCES = {
    "schema.sql": _PKG / "schema.sql",
    "adapter.py": _PKG / "adapter.py",
    "migrations/007": _PKG / "migrations/007_outcome_propagation_row_copy.sql",
}

#: The corrected direction, from migration 006: success reinforces, failure erodes.
_EXPECTED_MULTIPLIERS = {
    "critical_failure": "0.85",
    "failure": "0.90",
    "minor_failure": "0.92",
    "success": "1.03",
    "p1_incident": "0.85",
    "p2_incident": "0.90",
    "regression": "0.92",
    "clean_deploy": "1.03",
}


def _definition(path: Path) -> str:
    """The function body as it appears in ``path``, whitespace-normalised.

    Normalising is what lets the copy embedded in Python indentation be
    compared with the two in plain SQL files.
    """
    text = path.read_text()
    start = text.index("CREATE OR REPLACE FUNCTION amfs_propagate_outcome")
    end = text.index("LANGUAGE plpgsql", start)
    return re.sub(r"\s+", " ", text[start:end])


@pytest.fixture(scope="module")
def definitions() -> dict[str, str]:
    return {name: _definition(path) for name, path in _SOURCES.items()}


def test_all_three_places_still_define_it(definitions: dict[str, str]) -> None:
    """If a fourth appears, it has to be added here deliberately.

    A new copy that nobody compared is how this broke the first time.
    """
    assert set(definitions) == set(_SOURCES)
    for name, sql in definitions.items():
        assert sql, f"{name} no longer contains a definition"


@pytest.mark.parametrize("outcome,multiplier", sorted(_EXPECTED_MULTIPLIERS.items()))
def test_every_definition_agrees_on_the_multiplier(
    definitions: dict[str, str], outcome: str, multiplier: str
) -> None:
    """Success must reinforce confidence everywhere, not just where 006 ran."""
    for name, sql in definitions.items():
        assert f"WHEN '{outcome}' THEN multiplier := {multiplier};" in sql, (
            f"{name} does not apply {multiplier} to {outcome}. If this is the "
            "inverted table (success 0.97, critical_failure 1.15), a success is "
            "eroding confidence wherever this definition ran last."
        )


def test_every_definition_clamps(definitions: dict[str, str]) -> None:
    """Without the clamp, repeated failures push confidence past 1.0."""
    for name, sql in definitions.items():
        assert "LEAST(1.0, GREATEST(0.0," in sql, f"{name} does not clamp"


def test_every_definition_copies_the_row(definitions: dict[str, str]) -> None:
    """A column list silently resets every column added after it was written.

    See the header of migration 007. The invariant is not "does the list
    contain X" but "is there a list at all".
    """
    for name, sql in definitions.items():
        assert "to_jsonb(cur)" in sql, (
            f"{name} does not copy the row, so every column it fails to name is "
            "reset to its default on each outcome"
        )
        assert "INSERT INTO amfs_memory_entries (" not in sql, (
            f"{name} enumerates columns again; that snapshot of the schema goes "
            "stale the next time anyone runs ALTER TABLE"
        )


@pytest.mark.parametrize(
    "field", ["id", "version", "confidence", "outcome_count", "superseded_at"]
)
def test_every_definition_overrides_the_new_version_fields(
    definitions: dict[str, str], field: str
) -> None:
    """Copying the row means being explicit about what a new version changes.

    Missing ``id`` collides with the superseded row's primary key; missing
    ``superseded_at`` copies it, and the new version is born invisible to every
    ``superseded_at IS NULL`` read in the product.
    """
    for name, sql in definitions.items():
        assert f"'{field}'" in sql, f"{name} does not set {field} on the new version"


def test_every_definition_scopes_the_lookup_by_account(
    definitions: dict[str, str],
) -> None:
    """Otherwise an outcome in one account reinforces another's entry.

    Specifically ``IS NOT DISTINCT FROM``, not
    ``account_id = NEW.account_id OR NEW.account_id IS NULL``. The second looks
    equivalent and is not: it matches entries in every account whenever the
    outcome row carries no account, and nothing in this adapter sets one.
    """
    for name, sql in definitions.items():
        assert "account_id IS NOT DISTINCT FROM NEW.account_id" in sql, (
            f"{name} does not scope the entry lookup by account"
        )
        assert "NEW.account_id IS NULL" not in sql, (
            f"{name} uses the permissive account clause: an outcome with no "
            "account would reinforce entries in every account"
        )
