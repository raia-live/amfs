"""Postgres adapter tests — runs the full adapter contract suite.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
Example: AMFS_TEST_PG_DSN=postgresql://localhost/amfs_test pytest tests/integration/test_postgres_adapter.py
"""

from __future__ import annotations

import os

import pytest

from tests.integration.adapter_contract import AdapterContractTests

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="AMFS_TEST_PG_DSN not set — skipping Postgres tests",
)


@pytest.fixture
def adapter():
    """Create a PostgresAdapter with a fresh schema for each test."""
    from amfs_postgres.adapter import PostgresAdapter
    import psycopg

    # Clean tables before each test
    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS amfs_outcomes CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_memory_entries CASCADE")
    conn.close()

    a = PostgresAdapter(dsn=PG_DSN, namespace="test", auto_schema=True)
    yield a
    a.close()


class TestPostgresAdapter(AdapterContractTests):
    """Run all contract tests against the PostgresAdapter."""

    pass
