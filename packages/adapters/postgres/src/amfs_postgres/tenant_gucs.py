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
true until the end of the current transaction. Both adapters use false today,
set once per pool checkout, and with ``psycopg_pool`` that is sound — a checkout
owns its backend for the whole block, and every checkout overwrites all four
before running anything.

It stops being sound the moment a transaction-mode pooler sits in front, because
then one client connection maps to a different backend per transaction: the
values set at checkout land on whichever backend served that statement, and the
next query can be answered by a backend still carrying another tenant's session.
Pooler-safe means setting them *inside* the transaction that reads, which is
what :func:`tenant_transaction` is for.

The trap, which is why this is not a one-line change: with ``autocommit=True``
-- what both adapters open their pools with -- ``set_config(..., true)`` applies
to the implicit single-statement transaction it runs in and is gone before the
next statement. The GUCs read empty, ``NULLIF`` turns that into NULL, and RLS
matches nothing. It does not raise; it silently returns no rows. The same trap
is already documented for ``SET LOCAL`` in ``_apply_schema``. So the local form
is only ever used inside an explicit transaction, and
:func:`tenant_transaction` asserts it has one rather than trusting the caller.
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


def set_tenant_gucs(cur: Any, *, local: bool = False) -> None:
    """Apply the current request's tenant to an open cursor."""
    sql = SET_TENANT_GUCS_LOCAL if local else SET_TENANT_GUCS_SESSION
    cur.execute(sql, tenant_guc_values())


async def aset_tenant_gucs(cur: Any, *, local: bool = False) -> None:
    """Async twin of :func:`set_tenant_gucs`."""
    sql = SET_TENANT_GUCS_LOCAL if local else SET_TENANT_GUCS_SESSION
    await cur.execute(sql, tenant_guc_values())


def reset_tenant_gucs(conn: Any) -> None:
    """Blank all four. Suitable as a ``psycopg_pool`` ``reset`` callback.

    Defence in depth for the session-scoped path. Every checkout already
    overwrites all four before running anything, so this is not what stops a
    leak today; it stops one from surviving a checkout that failed partway, or a
    caller that reached the pool without going through the wrapper. Cheap enough
    to be worth not having to reason about.
    """
    with conn.cursor() as cur:
        cur.execute(CLEAR_TENANT_GUCS, CLEARED_TENANT_GUC_VALUES)


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
        await cur.execute(CLEAR_TENANT_GUCS, CLEARED_TENANT_GUC_VALUES)


def _assert_in_transaction(conn: Any) -> None:
    """Refuse to proceed unless a transaction is genuinely open.

    This is the guard against the silent-empty-result failure described in the
    module docstring. If the local ``set_config`` runs outside a transaction it
    is discarded immediately and every following query reads an empty tenant, so
    the caller gets no rows and no error. Better to raise here, loudly, than to
    return an empty page that looks like an empty account.
    """
    status = getattr(getattr(conn, "info", None), "transaction_status", None)
    if status is None:
        return
    name = getattr(status, "name", str(status))
    if name not in ("INTRANS", "INERROR", "ACTIVE"):
        raise RuntimeError(
            "tenant_transaction: no open transaction "
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

    Unwraps ``_TenantRLSPoolWrapper`` if handed one. The wrapper sets the same
    four settings session-scoped at checkout, and here that is not merely
    redundant: through a transaction pooler those writes land on an arbitrary
    backend and stay there, which is the leak this function exists to avoid.
    """
    inner = getattr(pool, "_inner", pool)
    with inner.connection() as conn:
        with conn.transaction():
            _assert_in_transaction(conn)
            with conn.cursor() as cur:
                set_tenant_gucs(cur, local=True)
            yield conn
