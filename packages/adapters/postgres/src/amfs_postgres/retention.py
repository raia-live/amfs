"""Operator entry point for decision-trace retention.

Nothing schedules retention on its own; run this from a cron job, a Kubernetes
CronJob or by hand::

    python -m amfs_postgres.retention --hot-days 30
    python -m amfs_postgres.retention --hot-days 30 --drop-after-days 365
    python -m amfs_postgres.retention --ensure-partitions --dry-run

The DSN comes from ``--dsn`` or ``AMFS_POSTGRES_DSN``; the namespace from
``--namespace`` or ``AMFS_NAMESPACE``. See
:meth:`amfs_postgres.adapter.PostgresAdapter.apply_trace_retention` for what
each tier does and the row-level-security caveat.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m amfs_postgres.retention",
        description="Strip old decision-trace payloads and drop old partitions.",
    )
    parser.add_argument("--dsn", default=os.environ.get("AMFS_POSTGRES_DSN"))
    parser.add_argument("--namespace", default=os.environ.get("AMFS_NAMESPACE", "default"))
    parser.add_argument(
        "--hot-days", type=int, default=30,
        help="Traces older than this lose task_input, response_text, tool_calls, "
             "query_events and error_events (default 30).",
    )
    parser.add_argument(
        "--no-strip", action="store_true",
        help="Do not strip payloads; only drop partitions if --drop-after-days is set.",
    )
    parser.add_argument(
        "--drop-after-days", type=int, default=None,
        help="Drop whole monthly partitions older than this many days. Off by default; "
             "a partition holds every namespace's traces for its month.",
    )
    parser.add_argument(
        "--ensure-partitions", action="store_true",
        help="Also create partitions for the current month and the next two.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run and exit without touching the database.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.dsn:
        parser.error("--dsn or AMFS_POSTGRES_DSN is required")

    plan = {
        "namespace": args.namespace,
        "hot_days": args.hot_days,
        "strip_payloads": not args.no_strip,
        "drop_after_days": args.drop_after_days,
        "ensure_partitions": args.ensure_partitions,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0

    from amfs_postgres.adapter import PostgresAdapter

    adapter = PostgresAdapter(args.dsn, namespace=args.namespace, auto_schema=False)
    try:
        result: dict = {}
        if args.ensure_partitions:
            result["partitions_created"] = adapter.ensure_trace_partitions()
        result.update(
            adapter.apply_trace_retention(
                hot_days=args.hot_days,
                strip_payloads=not args.no_strip,
                drop_after_days=args.drop_after_days,
            )
        )
        print(json.dumps(result, indent=2, default=str))
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
