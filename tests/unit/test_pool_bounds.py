"""How many database connections an adapter is allowed to hold.

Every pool multiplies against every instance of the process holding it, and
the database has a fixed max_connections. These tests pin the rules that let
an operator spend that budget from the environment, and the rule that a
caller who names a size for a particular adapter still gets it.
"""

from __future__ import annotations

import pytest

from amfs_postgres.adapter import (
    _DEFAULT_POOL_MAX,
    _DEFAULT_POOL_MIN,
    pool_bounds,
    statement_timeout_options,
)


class TestWithNothingConfigured:
    def test_the_defaults_are_unchanged(self):
        assert pool_bounds() == (_DEFAULT_POOL_MIN, _DEFAULT_POOL_MAX)


class TestFromTheEnvironment:
    def test_the_ceiling_is_taken_from_the_variable(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "4")
        assert pool_bounds() == (2, 4)

    def test_the_floor_is_taken_from_the_variable(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MIN", "1")
        assert pool_bounds() == (1, _DEFAULT_POOL_MAX)

    def test_a_ceiling_below_the_default_floor_lowers_the_floor(self, monkeypatch):
        # psycopg_pool refuses min_size > max_size, and an operator who caps a
        # service at one connection means one, not a crash at boot.
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "1")
        assert pool_bounds() == (1, 1)

    @pytest.mark.parametrize("junk", ["", "  ", "lots", "4.5", "0", "-3"])
    def test_a_value_that_is_not_a_usable_size_falls_back(self, monkeypatch, junk):
        # A typo in a deployment variable should not stop a service starting.
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", junk)
        assert pool_bounds() == (_DEFAULT_POOL_MIN, _DEFAULT_POOL_MAX)


class TestWhenTheCallerNamesASize:
    def test_an_explicit_ceiling_beats_the_environment(self, monkeypatch):
        # The background worker asking for two connections is describing what it
        # is for, and is not competing with what the request path may have.
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "20")
        assert pool_bounds(min_pool_size=1, max_pool_size=2) == (1, 2)

    def test_naming_only_one_end_leaves_the_other_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MIN", "3")
        assert pool_bounds(max_pool_size=6) == (3, 6)

    def test_an_explicit_floor_above_an_environment_ceiling_yields_to_it(
        self, monkeypatch
    ):
        # The ceiling is what the database can bear; the floor is a preference.
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "2")
        assert pool_bounds(min_pool_size=8) == (2, 2)


class TestTheStatementCeiling:
    def test_it_is_off_unless_asked_for(self, monkeypatch):
        monkeypatch.delenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", raising=False)
        assert statement_timeout_options() is None

    def test_a_blank_value_is_not_a_ceiling(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", "   ")
        assert statement_timeout_options() is None

    def test_it_becomes_a_libpq_option(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", "300s")
        assert statement_timeout_options() == "-c statement_timeout=300s"
