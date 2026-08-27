"""The four Postgres settings that decide what a request can see.

RLS on ``amfs_memory_entries`` is not one predicate but several, layered by the
SaaS migrations, and between them they read four GUCs:

  - ``amfs.current_account_id``  the outer tenant boundary
  - ``amfs.current_team_id``     the inner boundary within an account
  - ``amfs.is_account_admin``    bypasses the team boundary
  - ``amfs.current_user_id``     grants reach *across* accounts, via rooms

The last two are the dangerous pair. Three of the four narrow what a caller
sees, so getting them wrong shows up as missing rows; ``current_user_id`` is the
one that widens, because a user invited to a room in someone else's account
reaches it through permissive policies keyed on that value and nothing else.
A stale one is a cross-account read.

They live here rather than in either adapter because the sync and async adapters
were each carrying their own copy of the same four-statement SQL, which is two
places to remember when a fifth setting arrives, and drift here is a security
bug rather than a formatting one.

Session-scoped versus transaction-local
---------------------------------------
``set_config(name, value, is_local)`` takes a flag: false lasts for the session,
true until the end of the current transaction. With ``psycopg_pool`` alone,
false set once per checkout is sound — a checkout owns its backend for the whole
block, and every checkout overwrites all four before running anything.

It stops being sound the moment a transaction-mode pooler sits in front, because
then one client connection maps to a different backend per transaction: the
values set at checkout land on whichever backend served that statement, and the
next query can be answered by a backend still carrying another tenant's session.
That is a cross-tenant read, and it was reproduced against a real PgBouncer
before this was changed. Pooler-safe means setting them *inside* the transaction
that reads.

So the data path no longer sets them session-scoped at all: a checkout opens an
explicit transaction and sets all four inside it, and Postgres discards them at
commit. No tenant value is written session-scoped anywhere, which is what makes
the leak structurally impossible rather than merely unlikely.

The trap, which is why this was not a one-line change: with ``autocommit=True``
-- what both adapters open their pools with -- ``set_config(..., true)`` applies
to the implicit single-statement transaction it runs in and is gone before the
next statement. The GUCs read empty, ``NULLIF`` turns that into NULL, and RLS
matches nothing. It does not raise; it silently returns no rows. An empty page
that looks like an empty account is the worst failure available here, so the
local form is only ever used inside an explicit transaction and
:func:`require_open_transaction` asserts there is one rather than trusting it.

The maintenance exception
-------------------------
Two kinds of work cannot run inside a caller-imposed transaction:

* **DDL.** ``_apply_schema`` deliberately does not wrap ``schema.sql`` in one --
  a single transaction would hold ACCESS EXCLUSIVE on every object it touches
  for the whole run, which is the lock convoy that took production down on
  2026-08-12, only worse. It also sets ``lock_timeout`` session-scoped and
  relies on statement-at-a-time commit to keep partial progress on retry.
* **The backfills.** They call an embedder between statements, so a transaction
  would stay open across a batch of network round trips, and their
  embed-failed fallback issues more SQL after a caught error -- which inside a
  transaction is ``InFailedSqlTransaction`` instead of a fallback.

Those use :func:`blank_tenant_gucs` instead: still one write of all four on
every checkout, so nothing is inherited from the previous borrower, but to empty
rather than to a tenant. Empty fails closed under RLS, and clearing is the one
session-scoped write that cannot leak because there is nothing in it to leak.
Both are ops- and startup-time paths that touch no tenant rows.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

#: In the order the SQL below binds them. Adding one means adding it here and
#: nowhere else.
TENANT_GUC_NAMES: tuple[str, ...] = (
    "amfs.current_account_id",
    "amfs.current_team_id",
    "amfs.is_account_admin",
    "amfs.current_user_id",
)


def _set_config_sql(*, local: bool) -> str:
    """One statement setting all four, as literal names and a literal flag.

    Only the values are parameterised. The names and the flag are constants from
    this module, never caller input, so interpolating them cannot carry a
    payload -- and keeping them out of the parameter list is what lets the whole
    thing be a single round trip.
    """
    flag = "true" if local else "false"
    return "SELECT " + ", ".join(
        f"set_config('{name}', %s, {flag})" for name in TENANT_GUC_NAMES
    )


SET_TENANT_GUCS_SESSION = _set_config_sql(local=False)
SET_TENANT_GUCS_LOCAL = _set_config_sql(local=True)


def tenant_guc_values() -> tuple[str, str, str, str]:
    """The current request's four values, ready to bind.

    Empty string rather than None for the absent case, deliberately: the
    policies wrap each read in ``NULLIF(..., '')``, so empty becomes NULL and
    matches zero rows. A missing tenant therefore fails closed. Passing None
    would reach the same place, but only because psycopg renders it as NULL --
    empty string says it was meant.
    """
    from amfs_postgres.tenant_context import (
        get_request_is_account_admin,
        get_request_tenant_account_id,
        get_request_tenant_team_id,
        get_request_user_id,
    )

    account_id = get_request_tenant_account_id()
    team_id = get_request_tenant_team_id()
    user_id = get_request_user_id()
    return (
        account_id if account_id else "",
        team_id if team_id else "",
        "true" if get_request_is_account_admin() else "false",
        user_id if user_id else "",
    )


CLEAR_TENANT_GUCS = SET_TENANT_GUCS_SESSION
CLEARED_TENANT_GUC_VALUES: tuple[str, str, str, str] = ("", "", "false", "")


def set_tenant_gucs(cur: Any, *, local: bool) -> None:
    """Apply the current request's tenant to an open cursor.

    ``local`` has no default on purpose. It used to default to False, and a
    default here is the whole bug in one keyword: the unsafe scope was what you
    got for not thinking about it, and the difference between the two is a
    cross-tenant read rather than a style preference. Every caller now says
    which it means, and ``local=False`` with a real tenant in it no longer
    appears anywhere -- the session-scoped statement survives only to blank the
    four, via :func:`blank_tenant_gucs`.
    """
    sql = SET_TENANT_GUCS_LOCAL if local else SET_TENANT_GUCS_SESSION
    cur.execute(sql, tenant_guc_values())


async def aset_tenant_gucs(cur: Any, *, local: bool) -> None:
    """Async twin of :func:`set_tenant_gucs`. ``local`` is required there too."""
    sql = SET_TENANT_GUCS_LOCAL if local else SET_TENANT_GUCS_SESSION
    await cur.execute(sql, tenant_guc_values())


def blank_tenant_gucs(cur: Any) -> None:
    """Set all four to empty on an open cursor, for the maintenance path.

    The counterpart to :func:`set_tenant_gucs` for checkouts that cannot be
    wrapped in a transaction (see "The maintenance exception" above). It is
    session-scoped, like the old data path was, and that is safe here for a
    reason that does not generalise: there is no tenant in it. Nothing that
    could be leaked is written, and the previous borrower's tenant is still
    overwritten, so the connection cannot inherit reach it should not have.

    Under FORCE ROW LEVEL SECURITY -- which ``amfs_memory_entries`` has -- empty
    matches no rows even for the table owner, so a maintenance path that
    unexpectedly touched tenant data would read nothing rather than read
    everything. Failing closed is the point.
    """
    cur.execute(CLEAR_TENANT_GUCS, CLEARED_TENANT_GUC_VALUES)


async def ablank_tenant_gucs(cur: Any) -> None:
    """Async twin of :func:`blank_tenant_gucs`."""
    await cur.execute(CLEAR_TENANT_GUCS, CLEARED_TENANT_GUC_VALUES)


def reset_tenant_gucs(conn: Any) -> None:
    """Blank all four. Suitable as a ``psycopg_pool`` ``reset`` callback.

    Defence in depth. The data path now scopes its tenant to a transaction, so
    Postgres has already discarded it by the time a connection comes back and
    there is normally nothing here to clear. This is what covers the paths that
    do not go through that: a maintenance checkout, or a caller that reached the
    pool directly. Cheap enough to be worth not having to reason about.
    """
    with conn.cursor() as cur:
        blank_tenant_gucs(cur)


async def areset_tenant_gucs(conn: Any) -> None:
    """Async twin of :func:`reset_tenant_gucs`, for ``AsyncConnectionPool``.

    Needed separately because ``psycopg_pool`` awaits the async pool's ``reset``
    callback, so the sync function cannot be handed to both. Worth more here
    than on the sync side, not less: this pool serves the HTTP request path,
    which is the most multi-tenant thing in the system and the place where an
    idle connection left holding ``amfs.current_user_id`` would be holding the
    one setting that grants reach across accounts.
    """
    async with conn.cursor() as cur:
        await ablank_tenant_gucs(cur)


def require_open_transaction(conn: Any) -> None:
    """Refuse to proceed unless a transaction is genuinely open.

    This is the guard against the silent-empty-result failure described in the
    module docstring. If the local ``set_config`` runs outside a transaction it
    is discarded immediately and every following query reads an empty tenant, so
    the caller gets no rows and no error. Better to raise here, loudly, than to
    return an empty page that looks like an empty account.

    Public because both adapters' checkout wrappers call it, not just
    :func:`tenant_transaction`: the assertion is only worth having if it sits on
    every path that sets the local form.
    """
    status = getattr(getattr(conn, "info", None), "transaction_status", None)
    if status is None:
        return
    name = getattr(status, "name", str(status))
    if name not in ("INTRANS", "INERROR", "ACTIVE"):
        raise RuntimeError(
            "tenant GUCs: no open transaction "
            f"(transaction_status={name}). Transaction-local tenant settings "
            "would be discarded before the next statement and every read would "
            "silently return zero rows."
        )


@contextlib.contextmanager
def tenant_transaction(pool: Any) -> Iterator[Any]:
    """A connection whose tenant is scoped to one real transaction.

    The pooler-safe way to read or write tenant data: opens an explicit
    transaction, sets the four settings inside it, and lets Postgres discard
    them at commit. Nothing is left on the backend for the next borrower, which
    is what makes it safe to put a transaction-mode pooler in front.

    This is what ``_TenantRLSPoolWrapper`` now does on every checkout, so going
    through the adapter's pool already gets you this. The function stays for
    callers holding a bare pool, and for being the readable statement of the
    invariant that the wrapper implements.

    Unwraps the wrapper if handed one, purely to avoid doing the work twice: a
    nested ``transaction()`` is a savepoint and a second write of the same four
    values is a wasted round trip. Correct either way, which is the point.
    """
    inner = getattr(pool, "_inner", pool)
    with inner.connection() as conn:
        with conn.transaction():
            require_open_transaction(conn)
            with conn.cursor() as cur:
                set_tenant_gucs(cur, local=True)
            yield conn
