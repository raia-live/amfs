"""The definitions of ``amfs_propagate_outcome`` must stay in step.

The outcome trigger and its helpers are defined in two SQL sources in this
package, both ``CREATE OR REPLACE``:

  * ``schema.sql``, applied when a database is bootstrapped;
  * ``migrations/008_outcome_evidence.sql`` (the helpers) and
    ``migrations/009_action_priors.sql`` (the step function and the trigger,
    which 009 redefines), applied by whoever runs the migration files *and*
    by ``adapter.py``'s ``_apply_migrations``, which executes the files
    verbatim (``_OUTCOME_EVIDENCE_SQL``, ``_ACTION_PRIORS_SQL``) rather than
    carrying a third copy.

Whichever ran last silently *is* the function. That makes a stale copy not dead
code but a regression waiting for the next deploy, and it is exactly what
happened twice before: ``schema.sql`` kept the pre-006 inverted multipliers
long after 006 corrected them, and the adapter kept 006's column list after
007 replaced it with a row copy.

A behavioural test cannot catch this, because it only ever exercises whichever
definition won in the test database. So these are source tests, and the thing
they assert is agreement — and that the Python arithmetic in
``amfs_core.evidence`` uses the same constants the SQL does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from amfs_core import evidence as ev

_PKG = Path(__file__).resolve().parents[2] / "packages/adapters/postgres/src/amfs_postgres"

_SOURCES = {
    "schema.sql": _PKG / "schema.sql",
    "migrations/008": _PKG / "migrations/008_outcome_evidence.sql",
    "migrations/009": _PKG / "migrations/009_action_priors.sql",
}

#: function -> the migration that owns its current definition.
_FUNCTIONS = {
    "amfs_outcome_is_success": "migrations/008",
    "amfs_outcome_severity": "migrations/008",
    "amfs_outcome_multiplier": "migrations/008",
    "amfs_apply_outcome_step": "migrations/009",
    "amfs_propagate_outcome": "migrations/009",
}


def _definition(text: str, name: str) -> str:
    """The body of ``name`` as it appears in ``text``, whitespace-normalised."""
    start = text.index(f"CREATE OR REPLACE FUNCTION {name}")
    end = text.index("LANGUAGE", start)
    return re.sub(r"\s+", " ", text[start:end]).strip()


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    return {name: path.read_text() for name, path in _SOURCES.items()}


@pytest.mark.parametrize("function,owner", sorted(_FUNCTIONS.items()))
def test_both_sources_define_it_identically(
    sources: dict[str, str], function: str, owner: str
) -> None:
    assert _definition(sources["schema.sql"], function) == _definition(sources[owner], function), (
        f"{function} differs between schema.sql and {owner}"
    )


def test_009_runs_after_008_so_its_step_wins() -> None:
    adapter_src = (_PKG / "adapter.py").read_text()
    body = adapter_src[adapter_src.index("def _apply_migrations"):]
    assert body.index("cur.execute(_OUTCOME_EVIDENCE_SQL)") < body.index("cur.execute(_ACTION_PRIORS_SQL)")
    # 009 retires the seven-argument step that 008 (re)creates.
    m009 = (_PKG / "migrations/009_action_priors.sql").read_text()
    assert "DROP FUNCTION IF EXISTS amfs_apply_outcome_step(TEXT, UUID, TEXT[], TEXT, NUMERIC, TEXT, JSONB);" in m009
    assert "p_agent TEXT" in m009


def test_adapter_applies_the_migration_file_not_a_copy() -> None:
    adapter_src = (_PKG / "adapter.py").read_text()
    assert "CREATE OR REPLACE FUNCTION amfs_propagate_outcome" not in adapter_src, (
        "adapter.py grew its own copy of the trigger; apply the file instead"
    )
    assert "008_outcome_evidence.sql" in adapter_src
    assert "cur.execute(_OUTCOME_EVIDENCE_SQL)" in adapter_src
    assert "009_action_priors.sql" in adapter_src
    assert "cur.execute(_ACTION_PRIORS_SQL)" in adapter_src

    from amfs_postgres import adapter as adapter_mod

    assert adapter_mod._OUTCOME_EVIDENCE_SQL == _SOURCES["migrations/008"].read_text()
    assert adapter_mod._ACTION_PRIORS_SQL == _SOURCES["migrations/009"].read_text()


def test_migration_file_is_in_the_schema_fingerprint() -> None:
    adapter_src = (_PKG / "adapter.py").read_text()
    fp = adapter_src[adapter_src.index("def _schema_fingerprint"):]
    fp = fp[: fp.index("def _expected_tables")]
    assert "_OUTCOME_EVIDENCE_SQL" in fp
    assert "_ACTION_PRIORS_SQL" in fp


def test_sql_constants_match_python_evidence_model(sources: dict[str, str]) -> None:
    sql = sources["migrations/008"]
    step = _definition(sources["migrations/009"], "amfs_apply_outcome_step")
    assert f"* {ev.EVIDENCE_DECAY}" in step
    assert f"({ev.PRIOR_STRENGTH:.1f} * prior + e_s) / ({ev.PRIOR_STRENGTH:.1f} + e_s + e_f)" in step
    assert f"new_conf < {ev.DISCREDIT_THRESHOLD}" in step
    # First-strike tolerance (009) and the validators cap.
    assert f"COALESCE(cur.success_count, 0) >= {ev.FIRST_STRIKE_MIN_WINS}" in step
    assert "COALESCE(cur.failure_count, 0) = 0" in step
    assert f"jsonb_array_length(vals) > {ev.MAX_VALIDATORS}" in step
    severity = _definition(sql, "amfs_outcome_severity")
    for outcome, weight in ev.SEVERITY.items():
        assert f"WHEN '{outcome}' THEN {weight:.1f}" in severity, outcome
    success = _definition(sql, "amfs_outcome_is_success")
    for outcome in ev.SUCCESS_TYPES:
        assert f"'{outcome}'" in success


_LEGACY_MULTIPLIERS = {
    "critical_failure": "0.85",
    "failure": "0.90",
    "minor_failure": "0.92",
    "success": "1.03",
    "p1_incident": "0.85",
    "p2_incident": "0.90",
    "regression": "0.92",
    "clean_deploy": "1.03",
}


@pytest.mark.parametrize("outcome,multiplier", sorted(_LEGACY_MULTIPLIERS.items()))
def test_legacy_multipliers_kept_for_the_opt_in_model(
    sources: dict[str, str], outcome: str, multiplier: str
) -> None:
    body = _definition(sources["migrations/008"], "amfs_outcome_multiplier")
    assert f"WHEN '{outcome}' THEN {multiplier}" in body


def test_trigger_applies_attempts_before_the_terminal_outcome(sources: dict[str, str]) -> None:
    body = _definition(sources["migrations/009"], "amfs_propagate_outcome")
    attempts_at = body.index("jsonb_array_elements(COALESCE(NEW.attempts")
    terminal_at = body.index("NEW.causal_entry_keys, NEW.outcome_type")
    assert attempts_at < terminal_at
    assert "ORDER BY COALESCE((value->>'attempt')::INTEGER, 0)" in body


def test_trigger_keeps_the_account_clause(sources: dict[str, str]) -> None:
    body = _definition(sources["migrations/009"], "amfs_apply_outcome_step")
    assert "account_id IS NOT DISTINCT FROM p_account" in body
    assert "to_jsonb(cur) || jsonb_build_object(" in body
