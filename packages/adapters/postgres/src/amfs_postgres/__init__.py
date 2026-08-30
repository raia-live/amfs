"""AMFS Postgres Adapter — psycopg3 with back-propagation triggers."""

from amfs_postgres.adapter import PostgresAdapter
from amfs_postgres.async_adapter import AsyncPostgresAdapter
from amfs_postgres.tenant_gucs import (
    TENANT_GUC_NAMES,
    set_tenant_gucs,
    tenant_transaction,
)

__all__ = [
    "PostgresAdapter",
    "AsyncPostgresAdapter",
    # Anything doing tenant-scoped work on its own connections needs these
    # rather than a private import: the four settings are one list in one place,
    # and tenant_transaction is the transaction-scoped form that is safe behind
    # a transaction-mode pooler. Both adapters' checkouts now do the same thing
    # by default, so this is for callers holding a bare pool.
    # See amfs_postgres.tenant_gucs.
    "TENANT_GUC_NAMES",
    "set_tenant_gucs",
    "tenant_transaction",
]
