"""t2-11e: the SQL evidence functions are rendered from the Python model.

Migration 012 is a generated file. If it and ``render_evidence_sql()`` differ,
a constant or table changed on one side only — the exact drift migrations 008
and 009 made possible by carrying the arithmetic twice.
"""

from __future__ import annotations

from pathlib import Path

from amfs_core import evidence as ev
from amfs_core.models import OUTCOME_MULTIPLIERS
from amfs_postgres import evidence_sql

MIGRATIONS = Path(evidence_sql.__file__).parent / "migrations"


def test_migration_012_is_the_rendering_of_the_python_model() -> None:
    committed = (MIGRATIONS / evidence_sql.MIGRATION_NAME).read_text(encoding="utf-8")
    assert committed == evidence_sql.render_evidence_sql(), (
        "migration 012 differs from amfs_core.evidence; regenerate with "
        "`python -m amfs_postgres.evidence_sql > .../migrations/012_evidence_from_model.sql`"
    )


def test_the_rendering_carries_every_constant_and_table_of_the_model() -> None:
    sql = evidence_sql.render_evidence_sql()
    assert f"c_decay CONSTANT NUMERIC := {ev.EVIDENCE_DECAY!r}" in sql
    assert f"c_prior_strength CONSTANT NUMERIC := {float(ev.PRIOR_STRENGTH)!r}" in sql
    assert f"c_discredit CONSTANT NUMERIC := {ev.DISCREDIT_THRESHOLD!r}" in sql
    assert f"c_first_strike_wins CONSTANT INTEGER := {ev.FIRST_STRIKE_MIN_WINS}" in sql
    assert f"c_max_validators CONSTANT INTEGER := {ev.MAX_VALIDATORS}" in sql
    assert f"c_verified CONSTANT NUMERIC := {float(ev.VERIFIED_EVIDENCE_WEIGHT)!r}" in sql
    assert f"c_window CONSTANT INTEGER := {ev.UPLIFT_WINDOW}" in sql
    assert f"c_window_days CONSTANT INTEGER := {ev.UPLIFT_WINDOW_DAYS}" in sql
    credit = sql.split("CREATE OR REPLACE FUNCTION amfs_uplift_credit(", 1)[1].split("$$ LANGUAGE", 1)[0]
    assert f"GREATEST({float(ev.UPLIFT_FLOOR)!r}," in credit
    assert f"/ (sessions + {ev.UPLIFT_SHRINK}))" in credit
    for name, sev in ev.SEVERITY.items():
        assert f"WHEN '{name}' THEN {float(sev)!r}" in sql
    for key, mult in OUTCOME_MULTIPLIERS.items():
        name = key.value if hasattr(key, "value") else str(key)
        assert f"WHEN '{name}' THEN {float(mult)!r}" in sql
    for name in ev.SUCCESS_TYPES:
        assert f"'{name}'" in sql.split("amfs_outcome_is_known")[0]
    # No literal of the old hand-written bodies survives outside the constants:
    # the step reads the named constants, not 0.8 / 2.0 / 0.5 / 3 / 10 inline.
    step = sql.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1]
    body = step.split("BEGIN", 1)[1].split("$$ LANGUAGE", 1)[0]
    for literal in ("* 0.8", "(2.0 * prior", "< 0.5", ">= 0.5", ">= 3 THEN", "> 10 THEN", "- 10", "* 1.5"):
        assert literal not in body, literal


def test_the_rendering_replaces_the_functions_008_and_009_defined() -> None:
    """The same four scalar functions, the step with its ninth argument
    (verification) in place of 009's eight-argument one, and the trigger
    redefined to pass it — the 009 trigger would otherwise keep calling the
    overload this file drops."""
    sql = evidence_sql.render_evidence_sql()
    for fn in (
        "amfs_outcome_is_success(t TEXT)",
        "amfs_outcome_is_known(t TEXT)",
        "amfs_outcome_severity(t TEXT)",
        "amfs_outcome_multiplier(t TEXT)",
        "amfs_uplift_credit(",
        "amfs_refresh_uplift(",
        "amfs_propagate_outcome()",
    ):
        assert f"CREATE OR REPLACE FUNCTION {fn}" in sql
    assert sql.count("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(") == 1
    nine = (MIGRATIONS / "009_action_priors.sql").read_text(encoding="utf-8")
    sig_009 = nine.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1].split(
        ") RETURNS", 1
    )[0]
    sig_012 = sql.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1].split(
        ") RETURNS", 1
    )[0]
    assert sig_012 == sig_009.rstrip() + ",\n    p_verified BOOLEAN\n"
    assert (
        "DROP FUNCTION IF EXISTS amfs_apply_outcome_step("
        "TEXT, UUID, TEXT[], TEXT, NUMERIC, TEXT, JSONB, TEXT);"
    ) in sql
    trigger = sql.split("CREATE OR REPLACE FUNCTION amfs_propagate_outcome()", 1)[1]
    assert trigger.count("NEW.agent_id, verified") == 2
    assert "PERFORM amfs_refresh_uplift(NEW.namespace, NEW.account_id, all_keys)" in trigger


def test_the_columns_come_first_so_a_mirror_can_anchor_on_them() -> None:
    """The hosted product mirrors this file into a tenant migration and
    compares the two from the first ``ALTER TABLE`` on, header excluded."""
    sql = evidence_sql.render_evidence_sql()
    first_statement = next(line for line in sql.splitlines() if line and not line.startswith("--"))
    assert first_statement == "ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS verified_by TEXT;"
    assert "ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS uplift DOUBLE PRECISION;" in sql
    assert (
        "ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS uplift_sessions INTEGER NOT NULL DEFAULT 0;"
    ) in sql


def test_the_weight_carries_credit_and_verification() -> None:
    sql = evidence_sql.render_evidence_sql()
    step = sql.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1].split("$$ LANGUAGE", 1)[0]
    assert "* amfs_uplift_credit(cur.uplift, cur.uplift_sessions, is_ok)" in step
    assert "* CASE WHEN COALESCE(p_verified, FALSE) THEN c_verified ELSE 1.0 END" in step
    assert "/ n_keys" in step
