"""The tenant settings, against a real transaction-mode pooler.

RLS on tenant tables reads four Postgres settings, and how long those settings
last decides whether the isolation holds. They used to be set session-scoped,
once per pool checkout. With ``psycopg_pool`` alone that is sound: a checkout
owns its backend for the whole block.

Put PgBouncer in ``pool_mode = transaction`` in front and it stops being sound,
because one client connection then maps to a *different* server backend per
transaction. A setting written outside a transaction lands on whichever backend
happened to serve that statement and stays there, and the next query — a new
transaction — can be answered by a backend still carrying another tenant's
values. That is a cross-account read, and unit tests cannot see it: it lives
entirely in the relationship between the pooler and the session.

So this file runs a real PgBouncer. It has two halves, and the first matters as
much as the second:

``TestTheHarnessCanSeeTheLeak`` reproduces the bug with the old session-scoped
pattern and asserts that the leak *happens*. Without it, a passing suite would
be indistinguishable from a harness that cannot detect the problem — which is
the failure mode that let this bug exist while tests were green.

``TestTheAdapterDoesNotLeak`` then drives the real checkout wrapper through the
same pooler and asserts it holds.

Requires ``AMFS_TEST_PG_DSN`` and a ``pgbouncer`` binary on PATH; skipped
otherwise, so the default unit run is unaffected.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from psycopg.conninfo import conninfo_to_dict  # noqa: E402

DSN = os.environ.get("AMFS_TEST_PG_DSN")
PGBOUNCER = shutil.which("pgbouncer")

pytestmark = [
    pytest.mark.skipif(not DSN, reason="AMFS_TEST_PG_DSN not set"),
    pytest.mark.skipif(not PGBOUNCER, reason="pgbouncer not on PATH"),
]

#: The four settings, and the one the test reads back. Kept as a literal rather
#: than imported so that a rename in the adapter cannot quietly make this test
#: assert nothing.
ACCOUNT_GUC = "amfs.current_account_id"

#: Session-scoped, which is what this file exists to show is unsafe here.
SET_SESSION = (
    f"SELECT set_config('{ACCOUNT_GUC}', %s, false), "
    "set_config('amfs.current_team_id', %s, false), "
    "set_config('amfs.is_account_admin', %s, false), "
    "set_config('amfs.current_user_id', %s, false)"
)

TABLE = "pooler_leak_rows"

#: Every connection in this file is made as this role, not as the owner of the
#: test database. That is not tidiness: superusers and BYPASSRLS roles ignore row
#: level security entirely, even with FORCE, so running as the usual test
#: superuser would make every isolation assertion here pass without RLS ever
#: being consulted. ``TestTheHarnessCanSeeTheLeak`` checks this held.
APP_ROLE = "amfs_pooler_leak_app"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def accounts() -> tuple[str, str]:
    return str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture(scope="module")
def table(accounts: tuple[str, str]) -> str:
    """A table with the same RLS shape the real policies use, plus the app role.

    Purpose-built rather than reusing ``amfs_memory_entries``, because what is
    under test is the *scope* of the settings rather than any particular policy,
    and a table of two rows makes a leak unambiguous. The predicate is copied in
    form from the real ones: ``NULLIF(current_setting(..., true), '')``, so an
    absent setting reads empty, becomes NULL, and matches nothing.

    FORCE ROW LEVEL SECURITY as well as ENABLE, so that the table's owner is not
    exempt either. Belt and braces given the app role below, but the combination
    is what the real tenant tables use and the test should not be more permissive
    than production.
    """
    a, b = accounts
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute(
            f"CREATE TABLE {TABLE} (account_id uuid NOT NULL, label text NOT NULL)"
        )
        conn.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
        conn.execute(
            f"CREATE POLICY tenant_isolation ON {TABLE} USING ("
            f"  account_id = NULLIF(current_setting('{ACCOUNT_GUC}', true), '')::uuid"
            f")"
        )
        conn.execute(
            f"INSERT INTO {TABLE} (account_id, label) VALUES (%s, %s), (%s, %s)",
            (a, "belongs-to-a", b, "belongs-to-b"),
        )
        conn.execute(
            f"DO $$ BEGIN "
            f"  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') "
            f"  THEN CREATE ROLE {APP_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS; END IF; "
            f"END $$"
        )
        # NOSUPERUSER/NOBYPASSRLS again on the already-exists path, so a role
        # left over from an earlier run cannot silently be the privileged one.
        conn.execute(f"ALTER ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS")
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
        conn.execute(f"GRANT SELECT ON {TABLE} TO {APP_ROLE}")
    yield TABLE
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"REVOKE ALL ON {TABLE} FROM {APP_ROLE}")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")


@pytest.fixture(scope="module")
def pooled_dsn(tmp_path_factory, table: str) -> str:
    """A DSN pointing at a PgBouncer in transaction mode, one server connection.

    ``default_pool_size = 1`` is what makes the reuse deterministic instead of
    lucky. A larger pool hides the bug behind timing, which is how it survives
    review and reaches production.

    The auth file is not optional even with ``auth_type = trust``: PgBouncer
    still refuses a user it has never heard of, which is a confusing failure to
    debug from a connection-refused error.
    """
    info = conninfo_to_dict(DSN)
    dbname = info.get("dbname") or info.get("user") or "postgres"
    host = info.get("host") or "/tmp"
    port = info.get("port") or "5432"
    user = APP_ROLE

    workdir = tmp_path_factory.mktemp("pgbouncer")
    listen_port = _free_port()

    template = (Path(__file__).parent / "pgbouncer.ini").read_text()
    ini = workdir / "pgbouncer.ini"
    userlist = workdir / "userlist.txt"
    userlist.write_text(f'"{user}" ""\n')
    ini.write_text(
        template.replace("* = host=/tmp port=5432", f"* = host={host} port={port}")
        .replace("listen_port = 6432", f"listen_port = {listen_port}")
        .replace("auth_file = /dev/null", f"auth_file = {userlist}")
    )

    proc = subprocess.Popen(
        [PGBOUNCER, str(ini)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pooled = f"host=127.0.0.1 port={listen_port} dbname={dbname} user={user}"
    try:
        deadline = time.monotonic() + 15
        while True:
            if proc.poll() is not None:
                pytest.fail(f"pgbouncer exited early:\n{proc.stdout.read()}")
            try:
                with psycopg.connect(pooled, connect_timeout=2) as probe:
                    probe.execute("SELECT 1")
                break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    proc.terminate()
                    pytest.fail("pgbouncer did not accept connections in 15s")
                time.sleep(0.2)
        yield pooled
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _first(row):
    """The single column of a one-column row, whatever the row factory is.

    Raw connections here use the default tuple rows; the adapter's pool is
    opened with ``dict_row``, so the same query comes back as a mapping. Both
    appear in this file and neither is worth changing to suit the other.
    """
    return next(iter(row.values())) if isinstance(row, dict) else row[0]


def _labels(conn, table: str) -> list[str]:
    return [_first(r) for r in conn.execute(f"SELECT label FROM {table}").fetchall()]


def _setting(conn, name: str = ACCOUNT_GUC):
    return _first(conn.execute(f"SELECT current_setting('{name}', true)").fetchone())


class TestTheHarnessCanSeeTheLeak:
    """The control. If these fail, nothing else in this file means anything.

    A test that only asserts "no leak" is passed just as happily by a harness
    that cannot produce one — the wrong pool mode, a pool big enough that
    clients never share a backend, a policy that never applied. So the old
    pattern is run through this exact harness first and the leak is asserted
    as the expected outcome.
    """

    def test_the_connecting_role_is_actually_subject_to_rls(
        self, pooled_dsn: str
    ) -> None:
        """The cheapest way for this whole file to become meaningless.

        Superusers and BYPASSRLS roles skip row level security outright, FORCE or
        not. Connect as the usual test superuser and every isolation assertion
        below passes without a policy ever being evaluated — green, and blind.
        """
        with psycopg.connect(pooled_dsn, autocommit=True) as conn:
            role, is_super, bypasses = conn.execute(
                "SELECT current_user, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user"
            ).fetchone()

        assert role == APP_ROLE
        assert not is_super, "a superuser bypasses RLS; this test would prove nothing"
        assert not bypasses, "BYPASSRLS set; this test would prove nothing"

    def test_two_clients_share_one_backend(self, pooled_dsn: str) -> None:
        """The precondition for everything below."""
        with (
            psycopg.connect(pooled_dsn, autocommit=True) as a,
            psycopg.connect(pooled_dsn, autocommit=True) as b,
        ):
            pid_a = _first(a.execute("SELECT pg_backend_pid()").fetchone())
            pid_b = _first(b.execute("SELECT pg_backend_pid()").fetchone())

        assert pid_a == pid_b, (
            "the pooler is not multiplexing these clients onto one backend, so "
            "this file cannot observe the bug it exists to test"
        )

    def test_session_scoped_settings_leak_between_clients(
        self, pooled_dsn: str, table: str, accounts: tuple[str, str]
    ) -> None:
        """The bug, reproduced.

        Each statement below is its own transaction, because both connections
        are in autocommit — which is how the adapter's pools are opened. So A's
        settings land on a backend it does not keep, B's overwrite them, and A's
        next query is answered by a backend carrying B's tenant.
        """
        account_a, account_b = accounts
        with (
            psycopg.connect(pooled_dsn, autocommit=True) as a,
            psycopg.connect(pooled_dsn, autocommit=True) as b,
        ):
            a.execute(SET_SESSION, (account_a, "", "false", ""))
            b.execute(SET_SESSION, (account_b, "", "false", ""))

            seen_by_a = _labels(a, table)

        assert seen_by_a == ["belongs-to-b"], (
            f"expected A to be reading B's row through the shared backend, got "
            f"{seen_by_a}. If this is A's own row, the harness has stopped "
            f"reproducing the leak and the assertions below prove nothing."
        )

    def test_the_setting_outlives_the_client_that_wrote_it(
        self, pooled_dsn: str
    ) -> None:
        """Residue is the mechanism. A session-scoped write is still on the
        backend after the client that made it has gone away."""
        leaked = str(uuid.uuid4())
        with psycopg.connect(pooled_dsn, autocommit=True) as first:
            first.execute(SET_SESSION, (leaked, "", "false", ""))

        with psycopg.connect(pooled_dsn, autocommit=True) as second:
            still_there = _setting(second)

        assert still_there == leaked, (
            "the pooler is resetting session state between clients, so this "
            "harness cannot show the leak"
        )


class TestTheAdapterDoesNotLeak:
    """The real checkout wrapper, through the same pooler that leaks above.

    Of the three, ``test_each_checkout_sees_only_its_own_account`` is the one
    that discriminates: revert the wrapper to session-scoped settings and it is
    the only one that fails. The other two hold even on the broken version,
    because ``psycopg_pool``'s ``reset`` callback blanks the four as a
    connection goes back, so residue is gone by the time they look for it.

    They are kept anyway, and it is worth being clear about why rather than
    presenting them as proof they are not. That reset is defence in depth for a
    path that no longer needs it, and both tests state a property that should
    stay true independently of it — so if the reset callback is ever dropped as
    redundant, these are what notice.
    """

    @pytest.fixture
    def adapter(self, pooled_dsn: str):
        from amfs_postgres.adapter import PostgresAdapter

        # auto_schema=False: the bootstrap DDL is not what is under test, and
        # running it through a transaction pooler on every test would be slow
        # and beside the point.
        return PostgresAdapter(pooled_dsn, namespace="pooler-leak", auto_schema=False)

    @staticmethod
    def _as(account: str):
        """Enter a request context for one account, the way the server does."""
        from amfs_postgres import tenant_context

        tenant_context.set_tls_tenant_account_id(account)
        tenant_context.set_tls_tenant_team_id(None)
        tenant_context.set_tls_is_account_admin(False)
        tenant_context.set_tls_user_id(None)

    @staticmethod
    def _clear():
        from amfs_postgres import tenant_context

        tenant_context.clear_tls_tenant_account_id()
        tenant_context.clear_tls_tenant_team_id()
        tenant_context.clear_tls_is_account_admin()
        tenant_context.clear_tls_user_id()

    def test_each_checkout_sees_only_its_own_account(
        self, adapter, table: str, accounts: tuple[str, str]
    ) -> None:
        """Interleaved requests on one shared backend, which is the arrangement
        that leaked in the control above."""
        account_a, account_b = accounts
        try:
            self._as(account_a)
            with adapter._pool.connection() as conn:
                first_a = _labels(conn, table)

            self._as(account_b)
            with adapter._pool.connection() as conn:
                seen_b = _labels(conn, table)

            self._as(account_a)
            with adapter._pool.connection() as conn:
                second_a = _labels(conn, table)
        finally:
            self._clear()

        assert first_a == ["belongs-to-a"]
        assert seen_b == ["belongs-to-b"]
        assert second_a == ["belongs-to-a"], (
            "A's second request was answered by a backend still carrying B"
        )

    def test_nothing_is_left_on_the_backend_afterwards(
        self, adapter, accounts: tuple[str, str]
    ) -> None:
        """The property that makes the pooler safe, asserted directly.

        Transaction-local settings are discarded by Postgres at commit, so the
        shared backend carries no tenant once the request is over. The first half
        is the part only this test covers: that the tenant *is* visible inside
        its own transaction. Getting the scope right and the placement wrong —
        setting the four before the BEGIN, where autocommit throws them away —
        produces a system that leaks nothing and also reads nothing.
        """
        account_a, _ = accounts
        try:
            self._as(account_a)
            with adapter._pool.connection() as conn:
                assert _setting(conn) == account_a, (
                    "the tenant must be visible inside its own transaction"
                )
        finally:
            self._clear()

        with psycopg.connect(adapter._dsn, autocommit=True) as after:
            residue = _setting(after)

        assert residue in (None, ""), (
            f"the backend is still carrying {residue!r} after the request "
            f"finished; the next borrower would inherit it"
        )

    def test_a_request_with_no_tenant_reads_nothing(
        self, adapter, table: str, accounts: tuple[str, str]
    ) -> None:
        """Fails closed. A checkout with no tenant must not inherit the previous
        one — which, on a shared backend, is the difference between an empty
        result and someone else's data."""
        account_a, _ = accounts
        try:
            self._as(account_a)
            with adapter._pool.connection() as conn:
                assert _labels(conn, table) == ["belongs-to-a"]

            self._clear()
            with adapter._pool.connection() as conn:
                seen = _labels(conn, table)
        finally:
            self._clear()

        assert seen == [], f"a tenant-less request read {seen}"
