"""CLI entrypoint for the Cortex worker."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="amfs-cortex",
        description="AMFS Memory Cortex — streaming digest compiler",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check worker health and exit (for container health checks)",
    )
    parser.add_argument(
        "--debounce-ms",
        type=int,
        default=int(os.environ.get("AMFS_CORTEX_DEBOUNCE_MS", "3000")),
        help="Debounce interval for digest recompilation (default: 3000ms)",
    )
    parser.add_argument(
        "--no-advisory-lock",
        action="store_true",
        help="Disable advisory lock (not recommended for multi-replica)",
    )
    parser.add_argument(
        "--recompile-only",
        action="store_true",
        help="Recompile all digests once and exit (no streaming)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("amfs_cortex")

    dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if not dsn:
        logger.error("AMFS_POSTGRES_DSN environment variable is required")
        sys.exit(1)

    if args.health:
        _health_check(dsn)
        return

    from amfs_postgres.adapter import PostgresAdapter
    from amfs_cortex.compiler import DigestCompiler
    from amfs_cortex.worker import CortexWorker

    namespace = os.environ.get("AMFS_NAMESPACE", "default")
    adapter = PostgresAdapter(dsn=dsn, namespace=namespace)

    strategies = []
    try:
        from amfs_cortex_pro import get_pro_strategies
        strategies = get_pro_strategies()
        logger.info("Pro compilation strategies loaded")
    except ImportError:
        pass

    compiler = DigestCompiler(
        adapter=adapter,
        strategies=strategies or None,
        namespace=namespace,
    )

    if args.recompile_only:
        count = compiler.recompile_all()
        logger.info("Recompiled %d digests", count)
        adapter.close()
        return

    worker = CortexWorker(
        dsn=dsn,
        compiler=compiler,
        debounce_ms=args.debounce_ms,
        use_advisory_lock=not args.no_advisory_lock,
    )

    import signal

    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        worker.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Starting Cortex worker (namespace=%s, debounce=%dms)", namespace, args.debounce_ms)
    worker.run()
    adapter.close()


def _health_check(dsn: str) -> None:
    """Check if the advisory lock is held and digests are being compiled."""
    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True)
        row = conn.execute(
            "SELECT count(*) FROM amfs_digests WHERE compiled_at > NOW() - INTERVAL '5 minutes'"
        ).fetchone()
        recent_count = row[0] if row else 0
        conn.close()

        if recent_count > 0:
            print(f"healthy: {recent_count} digests compiled in last 5 minutes")
            sys.exit(0)
        else:
            print("degraded: no recent digest compilations")
            sys.exit(0)
    except Exception as e:
        print(f"unhealthy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
