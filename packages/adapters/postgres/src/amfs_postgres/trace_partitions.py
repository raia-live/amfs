"""Monthly range partitioning and retention for ``amfs_decision_traces``.

Traces are append-only and queried almost entirely by recency, which is the
shape range partitioning on ``created_at`` was made for: a page of recent
traces touches one or two partitions instead of an index over the whole
history, retention becomes ``DROP TABLE`` on a month instead of a
``DELETE`` that bloats the table it is meant to shrink, and payload stripping
runs one partition at a time rather than as a single table-wide ``UPDATE``.

Three entry points, all taking a cursor on an *autocommit* connection (the
adapter's maintenance checkout) because each manages its own transactions:

- :func:`migrate_to_partitioned` turns a flat table into a partitioned one,
  once, in a single transaction. Idempotent: a partitioned table is left alone.
- :func:`ensure_partitions` creates the partitions for the current month and
  the next few, so inserts never have to fall back to the DEFAULT partition.
  Cheap enough to run on every adapter start.
- :func:`apply_retention` strips captured payloads from traces past the hot
  window and optionally drops partitions past a longer one. Nothing schedules
  it; the adapter exposes it and an operator or a job calls it.
- :func:`sync_partition_rls` copies the parent's row-security policies onto each
  partition and enables row security there, which Postgres does not do by
  itself. It stops short of ``FORCE``, so the owner — the role that runs
  retention — stays exempt; that function's docstring says why.
  ``ensure_partitions`` calls it, so it runs on every start; it is public
  because it is worth being able to call and assert on directly.

Why copy rather than attach
---------------------------
The migration builds the partitioned parent beside the old table, copies the
rows into it, and drops the old table — instead of attaching the old table as
the DEFAULT partition, which would avoid the copy. Attaching was rejected
because it leaves every existing row in the default partition forever:
Postgres refuses to create a monthly partition for any range the default
partition already holds rows in, so the current month could never get its own
partition without moving rows anyway, and every query would keep scanning one
large unpruned child. Copying yields real monthly partitions from the first
day, and it costs one ``INSERT ... SELECT`` over a table the plan sizes at
thousands of rows per tenant. The whole thing is one transaction, so a failure
at any step leaves the flat table exactly as it was.

Rolling back after the fact is a plain table rebuild, because the parent has
the same columns in the same order::

    BEGIN;
    CREATE TABLE amfs_decision_traces_flat (LIKE amfs_decision_traces
        INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
    INSERT INTO amfs_decision_traces_flat SELECT * FROM amfs_decision_traces;
    ALTER TABLE amfs_decision_traces_flat ADD PRIMARY KEY (id);
    DROP TABLE amfs_decision_traces;
    ALTER TABLE amfs_decision_traces_flat RENAME TO amfs_decision_traces;
    COMMIT;

followed by recreating the indexes in ``schema.sql`` and any policies.

Primary key
-----------
A partitioned table's primary key must include the partition column, so the
key becomes ``(id, created_at)``. ``id`` is still a uuid generated per row and
``get_trace(id)`` still resolves by id alone; the lookup probes the id index of
each partition rather than one index, which at a few dozen partitions is a
handful of index probes.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

TABLE = "amfs_decision_traces"
LEGACY_TABLE = "amfs_decision_traces_legacy"
DEFAULT_PARTITION = "amfs_decision_traces_default"

_PARTITION_NAME_RE = re.compile(rf"^{TABLE}_y(\d{{4}})m(\d{{2}})$")

_PAYLOAD_COLUMNS_NONNULL = ("task_input", "response_text")
_PAYLOAD_COLUMNS_JSON_ARRAY = ("tool_calls", "query_events", "error_events")


def partition_name(year: int, month: int) -> str:
    return f"{TABLE}_y{year:04d}m{month:02d}"


def month_start(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(dt: datetime) -> datetime:
    start = month_start(dt)
    if start.month == 12:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)


def partition_bounds(name: str) -> tuple[datetime, datetime] | None:
    """``(start, end)`` for a partition we named, or None for anything else."""
    m = _PARTITION_NAME_RE.match(name)
    if not m:
        return None
    start = datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=UTC)
    return start, next_month(start)


def _relkind(cur: Any, table: str) -> str | None:
    cur.execute(
        """
        SELECT c.relkind
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s AND n.nspname = ANY (current_schemas(false))
        """,
        (table,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    kind = row["relkind"] if isinstance(row, dict) else row[0]
    return kind if isinstance(kind, str) else kind.decode()


def is_partitioned(cur: Any) -> bool:
    return _relkind(cur, TABLE) == "p"


def list_partitions(cur: Any) -> list[str]:
    cur.execute(
        """
        SELECT c.relname
        FROM pg_catalog.pg_inherits i
        JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid
        WHERE i.inhparent = %s::regclass
        ORDER BY c.relname
        """,
        (TABLE,),
    )
    return [r["relname"] if isinstance(r, dict) else r[0] for r in cur.fetchall()]


def _one(row: Any, key: str, idx: int = 0) -> Any:
    return row[key] if isinstance(row, dict) else row[idx]


# ──────────────────────────────────────────────────────────────────────
# Migration
# ──────────────────────────────────────────────────────────────────────


def _external_dependents(cur: Any) -> list[str]:
    """Objects that would stop the flat table from being dropped.

    Foreign keys from other tables and views over this one both follow the
    table through a rename and then block the drop; worse, a foreign key to a
    partitioned table must reference the whole partition key, which an
    existing ``REFERENCES amfs_decision_traces(id)`` cannot. When any exist the
    migration is skipped and reported rather than attempted and rolled back
    on every start.
    """
    cur.execute(
        """
        SELECT conrelid::regclass::text AS dep
        FROM pg_catalog.pg_constraint
        WHERE contype = 'f' AND confrelid = %s::regclass
        UNION ALL
        SELECT DISTINCT dependent.relname::text
        FROM pg_catalog.pg_depend d
        JOIN pg_catalog.pg_rewrite r ON r.oid = d.objid
        JOIN pg_catalog.pg_class dependent ON dependent.oid = r.ev_class
        WHERE d.refobjid = %s::regclass
          AND d.refclassid = 'pg_class'::regclass
          AND dependent.oid <> %s::regclass
        """,
        (TABLE, TABLE, TABLE),
    )
    return [_one(r, "dep") for r in cur.fetchall()]


def _capture_extras(cur: Any, table: str) -> dict[str, Any]:
    """Everything attached to the table that ``CREATE TABLE ... LIKE`` does not copy.

    Policies, the row-security flags, triggers, grants, owner and indexes all
    live on the *relation*, so they would follow the old table into the drop.
    They are read here and replayed onto the parent inside the same
    transaction. Policies matter most: the hosted deployment scopes tenants
    with row-level security, and a parent without them would answer every
    tenant's query with every tenant's rows.
    """
    cur.execute(
        """
        SELECT policyname, permissive, roles, cmd, qual, with_check
        FROM pg_catalog.pg_policies
        WHERE tablename = %s
        """,
        (table,),
    )
    policies = [dict(r) if isinstance(r, dict) else r for r in cur.fetchall()]

    cur.execute(
        "SELECT relrowsecurity, relforcerowsecurity "
        "FROM pg_catalog.pg_class WHERE oid = %s::regclass",
        (table,),
    )
    flags = cur.fetchone()

    cur.execute(
        """
        SELECT tgname, pg_get_triggerdef(oid) AS def
        FROM pg_catalog.pg_trigger
        WHERE tgrelid = %s::regclass AND NOT tgisinternal
        """,
        (table,),
    )
    triggers = [_one(r, "def", 1) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT grantee, privilege_type, is_grantable
        FROM information_schema.role_table_grants
        WHERE table_name = %s AND grantee <> current_user
        """,
        (table,),
    )
    grants = [dict(r) if isinstance(r, dict) else r for r in cur.fetchall()]

    cur.execute("SELECT tableowner FROM pg_catalog.pg_tables WHERE tablename = %s", (table,))
    owner_row = cur.fetchone()

    cur.execute(
        """
        SELECT i.indexname, i.indexdef, x.indisunique, x.indisprimary
        FROM pg_catalog.pg_indexes i
        JOIN pg_catalog.pg_class c ON c.relname = i.indexname
        JOIN pg_catalog.pg_index x ON x.indexrelid = c.oid
        WHERE i.tablename = %s
        """,
        (table,),
    )
    indexes = [dict(r) if isinstance(r, dict) else r for r in cur.fetchall()]

    return {
        "policies": policies,
        "rls": bool(_one(flags, "relrowsecurity")) if flags else False,
        "force_rls": bool(_one(flags, "relforcerowsecurity", 1)) if flags else False,
        "triggers": triggers,
        "grants": grants,
        "owner": _one(owner_row, "tableowner") if owner_row else None,
        "indexes": indexes,
    }


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _role_list(roles: Any) -> str:
    names = list(roles) if roles is not None else []
    if not names:
        return "PUBLIC"
    out = []
    for r in names:
        r = str(r)
        if r.lower() in ("public", "current_user", "session_user"):
            out.append(r.upper())
        else:
            out.append(_quote_ident(r))
    return ", ".join(out)


def _replay_extras(cur: Any, extras: dict[str, Any], new: str) -> None:
    """Recreate what :func:`_capture_extras` read, on the new parent.

    The definitions were captured before the rename, so they already name
    ``amfs_decision_traces`` — the name the parent now holds.
    """
    for idx in extras["indexes"]:
        if idx["indisprimary"]:
            continue  # replaced by PRIMARY KEY (id, created_at)
        definition: str = idx["indexdef"]
        if idx["indisunique"] and "created_at" not in definition:
            logger.warning(
                "trace partitioning: unique index %s cannot exist on a partitioned "
                "table without created_at — not recreated",
                idx["indexname"],
            )
            continue
        definition = re.sub(
            r"^CREATE (UNIQUE )?INDEX (?!IF NOT EXISTS)",
            lambda m: f"CREATE {m.group(1) or ''}INDEX IF NOT EXISTS ",
            definition,
        )
        cur.execute(definition)

    for trig in extras["triggers"]:
        cur.execute(trig)

    for pol in extras["policies"]:
        is_permissive = str(pol["permissive"]).upper().startswith("PERM")
        permissive = "PERMISSIVE" if is_permissive else "RESTRICTIVE"
        parts = [
            f"CREATE POLICY {_quote_ident(pol['policyname'])} ON {new}",
            f"AS {permissive}",
            f"FOR {pol['cmd'] or 'ALL'}",
            f"TO {_role_list(pol['roles'])}",
        ]
        if pol.get("qual"):
            parts.append(f"USING ({pol['qual']})")
        if pol.get("with_check"):
            parts.append(f"WITH CHECK ({pol['with_check']})")
        cur.execute(" ".join(parts))

    if extras["rls"]:
        cur.execute(f"ALTER TABLE {new} ENABLE ROW LEVEL SECURITY")
    if extras["force_rls"]:
        cur.execute(f"ALTER TABLE {new} FORCE ROW LEVEL SECURITY")

    # Grants and ownership are best-effort: a role that can rebuild the table
    # is not necessarily one that can grant to, or hand it to, another role.
    # A savepoint keeps a refusal here from undoing the migration.
    for g in extras["grants"]:
        grantee = g["grantee"]
        target = "PUBLIC" if str(grantee).upper() == "PUBLIC" else _quote_ident(str(grantee))
        suffix = " WITH GRANT OPTION" if str(g.get("is_grantable", "NO")).upper() == "YES" else ""
        cur.execute("SAVEPOINT amfs_grant")
        try:
            cur.execute(f"GRANT {g['privilege_type']} ON {new} TO {target}{suffix}")
            cur.execute("RELEASE SAVEPOINT amfs_grant")
        except Exception as exc:  # noqa: BLE001
            cur.execute("ROLLBACK TO SAVEPOINT amfs_grant")
            logger.warning("trace partitioning: could not replay grant to %s: %s", grantee, exc)

    owner = extras.get("owner")
    if owner:
        cur.execute("SAVEPOINT amfs_owner")
        try:
            cur.execute(f"ALTER TABLE {new} OWNER TO {_quote_ident(owner)}")
            cur.execute("RELEASE SAVEPOINT amfs_owner")
        except Exception as exc:  # noqa: BLE001
            cur.execute("ROLLBACK TO SAVEPOINT amfs_owner")
            logger.warning("trace partitioning: could not set owner %s: %s", owner, exc)


def _months_between(first: datetime, last: datetime) -> list[datetime]:
    months: list[datetime] = []
    m = month_start(first)
    stop = month_start(last)
    while m <= stop:
        months.append(m)
        m = next_month(m)
    return months


def _create_partition_sql(name: str, start: datetime, end: datetime) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF {TABLE} "
        f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
    )


def migrate_to_partitioned(cur: Any, *, months_ahead: int = 2, now: datetime | None = None) -> bool:
    """Convert a flat ``amfs_decision_traces`` into a partitioned one, once.

    Returns True when the table is partitioned on return — whether it already
    was or this call did it — or when partitioning is deliberately skipped for
    a permanent reason (foreign keys or views over the table), so the caller
    records the schema as applied and does not retry on every start. Raises on
    a transient failure after rolling back, so the caller can leave the schema
    marker unset and try again next start.
    """
    kind = _relkind(cur, TABLE)
    if kind is None:
        return False
    if kind == "p":
        return True
    if kind != "r":
        logger.warning("trace partitioning: %s has relkind %r — leaving it alone", TABLE, kind)
        return True

    dependents = _external_dependents(cur)
    if dependents:
        logger.error(
            "trace partitioning: %s has dependents %s (foreign keys or views) — "
            "cannot be partitioned in place; leaving the flat table. Remove or "
            "recreate the dependents against (id, created_at) and restart.",
            TABLE, dependents,
        )
        return True

    now = now or datetime.now(UTC)
    try:
        cur.execute("BEGIN")
        extras = _capture_extras(cur, TABLE)

        cur.execute(f"ALTER TABLE {TABLE} RENAME TO {LEGACY_TABLE}")
        # Index names are schema-wide: the new primary key wants the name the
        # old one holds, and the schema.sql indexes are recreated under their
        # own names after the drop below.
        cur.execute(f"ALTER INDEX IF EXISTS {TABLE}_pkey RENAME TO {LEGACY_TABLE}_pkey")
        # The copy has to see every row. Under FORCE ROW LEVEL SECURITY a
        # maintenance connection with no tenant sees none, and would copy an
        # empty table and drop the full one; disabling RLS on the doomed table
        # is what makes the count check below meaningful. Only the owner may
        # do this, and a role that is not the owner fails here, rolls back, and
        # leaves the flat table in place.
        cur.execute(f"ALTER TABLE {LEGACY_TABLE} DISABLE ROW LEVEL SECURITY")

        cur.execute(
            f"""
            CREATE TABLE {TABLE} (
                LIKE {LEGACY_TABLE} INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING GENERATED,
                PRIMARY KEY (id, created_at)
            ) PARTITION BY RANGE (created_at)
            """
        )
        cur.execute(f"CREATE TABLE {DEFAULT_PARTITION} PARTITION OF {TABLE} DEFAULT")

        cur.execute(
            f"SELECT min(created_at) AS lo, max(created_at) AS hi, count(*) AS n "
            f"FROM {LEGACY_TABLE}"
        )
        span = cur.fetchone()
        lo, hi, legacy_count = _one(span, "lo", 0), _one(span, "hi", 1), int(_one(span, "n", 2))

        wanted: dict[str, tuple[datetime, datetime]] = {}
        horizon = month_start(now)
        for _ in range(months_ahead):
            horizon = next_month(horizon)
        for m in _months_between(lo or now, max(hi or now, horizon)):
            wanted[partition_name(m.year, m.month)] = (m, next_month(m))
        for name, (start, end) in wanted.items():
            cur.execute(_create_partition_sql(name, start, end))

        cur.execute(f"INSERT INTO {TABLE} SELECT * FROM {LEGACY_TABLE}")
        cur.execute(f"SELECT count(*) AS n FROM {TABLE}")
        copied = int(_one(cur.fetchone(), "n"))
        if copied != legacy_count:
            raise RuntimeError(
                f"trace partitioning: copied {copied} rows but the flat table held "
                f"{legacy_count}; refusing to drop it"
            )

        cur.execute(f"DROP TABLE {LEGACY_TABLE}")
        _replay_extras(cur, extras, TABLE)
        cur.execute("COMMIT")
    except Exception:
        try:
            cur.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        logger.exception("trace partitioning: migration rolled back; flat table left in place")
        raise

    logger.info(
        "trace partitioning: %s is now range-partitioned by month (%d rows moved, %d partitions)",
        TABLE, legacy_count, len(wanted),
    )
    return True


# ──────────────────────────────────────────────────────────────────────
# Partition upkeep
# ──────────────────────────────────────────────────────────────────────


def _default_partition_months(cur: Any) -> list[datetime]:
    """Months for which the DEFAULT partition is holding rows.

    Rows land there only when no partition covered their ``created_at`` — a
    backdated trace, or a deployment that was not restarted for longer than
    the partitions created ahead. Each such month is given a partition of its
    own by :func:`ensure_partitions`.
    """
    cur.execute(
        f"""
        SELECT DISTINCT date_trunc('month', created_at AT TIME ZONE 'UTC') AS m
        FROM {DEFAULT_PARTITION}
        """
    )
    months = []
    for r in cur.fetchall():
        m = _one(r, "m")
        if m is None:
            continue
        if m.tzinfo is None:
            m = m.replace(tzinfo=UTC)
        months.append(month_start(m))
    return months


def _create_partition(cur: Any, name: str, start: datetime, end: datetime) -> None:
    """Create one monthly partition, moving any rows the default partition holds for it.

    Postgres refuses ``CREATE TABLE ... PARTITION OF`` for a range the default
    partition has rows in, so those rows are moved into a standalone table
    first and the table attached afterwards, all in one transaction.
    """
    cur.execute("BEGIN")
    try:
        cur.execute(
            f"SELECT 1 FROM {DEFAULT_PARTITION} WHERE created_at >= %s AND created_at < %s LIMIT 1",
            (start, end),
        )
        if cur.fetchone() is None:
            cur.execute(_create_partition_sql(name, start, end))
        else:
            cur.execute(
                f"CREATE TABLE {name} (LIKE {TABLE} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)"
            )
            cur.execute(
                f"INSERT INTO {name} SELECT * FROM {DEFAULT_PARTITION} "
                f"WHERE created_at >= %s AND created_at < %s",
                (start, end),
            )
            cur.execute(
                f"DELETE FROM {DEFAULT_PARTITION} WHERE created_at >= %s AND created_at < %s",
                (start, end),
            )
            cur.execute(
                f"ALTER TABLE {TABLE} ATTACH PARTITION {name} "
                f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
            )
        cur.execute("COMMIT")
    except Exception:
        try:
            cur.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        raise


def _parent_rls(cur: Any) -> tuple[bool, bool]:
    """Whether the parent has row-level security enabled, and forced."""
    cur.execute(
        "SELECT relrowsecurity, relforcerowsecurity FROM pg_catalog.pg_class "
        "WHERE oid = %s::regclass",
        (TABLE,),
    )
    row = cur.fetchone()
    if row is None:
        return (False, False)
    if isinstance(row, dict):
        return (bool(row["relrowsecurity"]), bool(row["relforcerowsecurity"]))
    return (bool(row[0]), bool(row[1]))


def _policies_on(cur: Any, table: str) -> dict[str, dict[str, Any]]:
    """The policies defined on one table, by name, as ``pg_policies`` reports them."""
    cur.execute(
        """
        SELECT policyname, permissive, roles, cmd, qual, with_check
        FROM pg_catalog.pg_policies
        WHERE schemaname = current_schema() AND tablename = %s
        """,
        (table,),
    )
    out: dict[str, dict[str, Any]] = {}
    for r in cur.fetchall():
        if isinstance(r, dict):
            out[r["policyname"]] = r
        else:
            out[r[0]] = {
                "policyname": r[0], "permissive": r[1], "roles": r[2],
                "cmd": r[3], "qual": r[4], "with_check": r[5],
            }
    return out


def _create_policy_sql(table: str, p: dict[str, Any]) -> str:
    """Rebuild one policy's DDL for another table.

    Replayed from the catalogue rather than restated here: the predicate belongs
    to whoever defined it on the parent — the tenant package, in the deployment
    that has one — and a copy written out in this file would be a second version
    of it, free to drift the moment either changed. This file does not need to
    know what the policy says.
    """
    parts = [f'CREATE POLICY "{p["policyname"]}" ON {table}']
    if (p.get("permissive") or "PERMISSIVE").upper() == "RESTRICTIVE":
        parts.append("AS RESTRICTIVE")
    cmd = (p.get("cmd") or "ALL").upper()
    if cmd != "ALL":
        parts.append(f"FOR {cmd}")
    roles = p.get("roles") or []
    if isinstance(roles, str):  # rendered as '{public}' by some drivers
        roles = [r for r in roles.strip("{}").split(",") if r]
    if roles and list(roles) != ["public"] and list(roles) != ["-"]:
        parts.append("TO " + ", ".join(f'"{r}"' for r in roles))
    if p.get("qual"):
        parts.append(f"USING ({p['qual']})")
    if p.get("with_check"):
        parts.append(f"WITH CHECK ({p['with_check']})")
    return " ".join(parts)


def sync_partition_rls(cur: Any) -> list[str]:
    """Give every partition the row-level security its parent carries.

    A partition inherits neither the parent's ``relrowsecurity`` nor its policies.
    Postgres applies the parent's policies to rows reached *through* the parent, but
    a query naming a partition directly is checked against that partition's own
    policies — of which a fresh partition has none — so a table whose parent is
    forced can still be read a month at a time with no tenant in scope. Nothing in
    this repository selects from a partition by name, which is what kept it latent;
    it is one convenient query away from not being.

    Enabling row security on the partition without also copying the policies would
    trade that for the opposite failure: with security enabled and no policy that
    applies, the partition yields nothing at all, and a maintenance query would
    quietly read zero rows rather than be refused. Both halves or neither.

    What this deliberately does not do: ``FORCE``
    --------------------------------------------
    The parent is forced, and this does not pass that down, because the owner is
    also the role that maintains these tables. Retention strips payloads with an
    ``UPDATE`` naming a partition, and ``ensure_partitions`` drains the default
    one by reading it, both on the maintenance checkout, which blanks the tenant
    settings by design (see ``PostgresAdapter._maintenance_connection``). Forced,
    those statements match no rows — measured, not supposed: the strip returns
    ``UPDATE 0`` and reports success having changed nothing, and months stuck in
    the default partition are never drained. Unforced, the owner is exempt from
    the policy and both keep working, while every other role reading a partition
    by name is scoped exactly as it is through the parent.

    So this closes the gap for every role except the one that owns the tables.
    Covering the owner too means giving maintenance a sanctioned way through —
    a role with ``BYPASSRLS`` for the retention and drain statements — and only
    then forcing the partitions. That is a deployment change, not one this
    function can make, and doing half of it here would swap a bypass nothing
    currently uses for a retention job that silently stops working.

    Returns the partitions changed, so a caller can log or assert on it. Idempotent:
    a partition already carrying the parent's policies is left alone.
    """
    if not is_partitioned(cur):
        return []
    enabled, _forced = _parent_rls(cur)
    if not enabled:
        # Nothing to inherit. An OSS install without the tenant package has no
        # policies on the parent either, and enabling security here would take its
        # traces away from it.
        return []

    parent_policies = _policies_on(cur, TABLE)
    changed: list[str] = []
    for name in list_partitions(cur):
        touched = False
        have = _policies_on(cur, name)
        for policy_name, policy in parent_policies.items():
            if policy_name in have:
                continue
            cur.execute(_create_policy_sql(name, policy))
            touched = True
        part_enabled, _ = _partition_rls(cur, name)
        # After the policies, never before: a partition that had security enabled
        # for the instant it had no policy would answer that instant with no rows.
        # ENABLE only — see the note on FORCE in this function's docstring.
        if not part_enabled:
            cur.execute(f"ALTER TABLE {name} ENABLE ROW LEVEL SECURITY")
            touched = True
        if touched:
            changed.append(name)
    if changed:
        logger.info(
            "trace partitioning: row-level security synced onto %s", ", ".join(changed)
        )
    return changed


def _partition_rls(cur: Any, name: str) -> tuple[bool, bool]:
    cur.execute(
        "SELECT relrowsecurity, relforcerowsecurity FROM pg_catalog.pg_class "
        "WHERE oid = %s::regclass",
        (name,),
    )
    row = cur.fetchone()
    if row is None:
        return (False, False)
    if isinstance(row, dict):
        return (bool(row["relrowsecurity"]), bool(row["relforcerowsecurity"]))
    return (bool(row[0]), bool(row[1]))


def ensure_partitions(cur: Any, *, months_ahead: int = 2, now: datetime | None = None) -> list[str]:
    """Create partitions for this month and *months_ahead* more. Returns the names created.

    A no-op on a table that is not partitioned. Also gives a partition to any
    month the DEFAULT partition is holding rows for, so the default partition
    stays empty and every query can prune.
    """
    if not is_partitioned(cur):
        return []
    existing = set(list_partitions(cur))
    now = now or datetime.now(UTC)

    wanted: dict[str, tuple[datetime, datetime]] = {}
    m = month_start(now)
    for _ in range(months_ahead + 1):
        wanted[partition_name(m.year, m.month)] = (m, next_month(m))
        m = next_month(m)
    if DEFAULT_PARTITION in existing:
        for m in _default_partition_months(cur):
            wanted.setdefault(partition_name(m.year, m.month), (m, next_month(m)))
    else:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {DEFAULT_PARTITION} PARTITION OF {TABLE} DEFAULT")

    created: list[str] = []
    for name, (start, end) in sorted(wanted.items()):
        if name in existing:
            continue
        _create_partition(cur, name, start, end)
        created.append(name)
    if created:
        logger.info("trace partitioning: created %s", ", ".join(created))
    # Every partition, not only the ones just created: this converges rather than
    # stamping at creation, so a partition made by an older version of this file,
    # or by hand, is brought into line on the next start rather than staying the
    # one month that is readable without a tenant.
    try:
        sync_partition_rls(cur)
    except Exception:
        # Altering a partition needs ownership of it. A deployment where the
        # connecting role does not own its tables should say so loudly and keep
        # serving, rather than fail to start over a gap it already had.
        logger.warning("trace partitioning: could not sync row-level security", exc_info=True)
    return created


# ──────────────────────────────────────────────────────────────────────
# Retention
# ──────────────────────────────────────────────────────────────────────


def _strip_sql(target: str, *, bounded: bool) -> str:
    sets = ", ".join(
        [f"{c} = NULL" for c in _PAYLOAD_COLUMNS_NONNULL]
        + [f"{c} = '[]'::jsonb" for c in _PAYLOAD_COLUMNS_JSON_ARRAY]
    )
    has_payload = " OR ".join(
        [f"{c} IS NOT NULL" for c in _PAYLOAD_COLUMNS_NONNULL]
        + [f"{c} <> '[]'::jsonb" for c in _PAYLOAD_COLUMNS_JSON_ARRAY]
    )
    where = f"namespace = %s AND ({has_payload})"
    if bounded:
        where += " AND created_at < %s"
    return f"UPDATE {target} SET {sets} WHERE {where}"


def apply_retention(
    cur: Any,
    *,
    namespace: str,
    hot_days: int = 30,
    strip_payloads: bool = True,
    drop_after_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Age traces out in two tiers.

    Past *hot_days*, the captured payload — ``task_input``, ``response_text``,
    ``tool_calls``, ``query_events``, ``error_events`` — is cleared while the
    trace itself, its ``causal_entries``, contexts, state diff and metadata
    stay, so explainability survives and only the bulk goes. Past
    *drop_after_days*, whole monthly partitions are dropped.

    Stripping is scoped to *namespace* and runs one partition at a time so no
    single statement touches the whole table. Dropping a partition is not
    scoped — a partition holds every namespace's traces for that month — which
    is why it is off unless asked for. On an unpartitioned table stripping
    still works as one bounded ``UPDATE`` and nothing is dropped.
    """
    now = now or datetime.now(UTC)
    hot_cutoff = now - timedelta(days=hot_days)
    drop_cutoff = now - timedelta(days=drop_after_days) if drop_after_days is not None else None
    result: dict[str, Any] = {
        "stripped_rows": 0,
        "dropped_partitions": [],
        "hot_cutoff": hot_cutoff.isoformat(),
        "drop_cutoff": drop_cutoff.isoformat() if drop_cutoff else None,
    }

    if not is_partitioned(cur):
        if strip_payloads:
            cur.execute(_strip_sql(TABLE, bounded=True), (namespace, hot_cutoff))
            result["stripped_rows"] = max(cur.rowcount, 0)
        return result

    partitions = list_partitions(cur)

    if drop_cutoff is not None:
        for name in partitions:
            bounds = partition_bounds(name)
            if bounds is None:
                continue
            _, end = bounds
            if end <= drop_cutoff:
                cur.execute(f"DROP TABLE IF EXISTS {name}")
                result["dropped_partitions"].append(name)
        partitions = [p for p in partitions if p not in result["dropped_partitions"]]

    if strip_payloads:
        for name in partitions:
            bounds = partition_bounds(name)
            if bounds is not None:
                start, end = bounds
                if start >= hot_cutoff:
                    continue
                if end <= hot_cutoff:
                    cur.execute(_strip_sql(name, bounded=False), (namespace,))
                else:
                    cur.execute(_strip_sql(name, bounded=True), (namespace, hot_cutoff))
            elif name == DEFAULT_PARTITION:
                cur.execute(_strip_sql(name, bounded=True), (namespace, hot_cutoff))
            else:
                continue
            result["stripped_rows"] += max(cur.rowcount, 0)

    return result


def create_causal_entries_index(cur: Any) -> None:
    """GIN index that serves ``causal_entries @> '[{"entity_path": ...}]'``.

    ``jsonb_path_ops`` rather than the default opclass: it only supports
    containment, which is the only operator the filter uses, and its index is
    smaller and faster for exactly that. Works the same on a flat table and a
    partitioned one, where it cascades to every partition.
    """
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_traces_causal_entries_gin
            ON {TABLE} USING gin (causal_entries jsonb_path_ops)
        """
    )
