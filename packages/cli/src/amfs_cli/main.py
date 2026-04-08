"""AMFS CLI entry point."""

from __future__ import annotations

import typer

from amfs_cli.init import init_command
from amfs_cli.login import login_command
from amfs_cli.recall import recall_command
from amfs_cli.search import search_command
from amfs_cli.snapshot import app as snapshot_app
from amfs_cli.inspect import app as inspect_app
from amfs_cli.stats import stats_command
from amfs_cli.status import status_command
from amfs_cli.watch import watch_command
from amfs_cli.whoami import whoami_command
from amfs_cli.write import write_command

app = typer.Typer(
    name="amfs",
    help="AMFS — Agent Memory File System CLI",
    no_args_is_help=True,
)

app.command(name="init", help="Initialise an AMFS project")(init_command)
app.command(name="login", help="Authenticate with an AMFS server")(login_command)
app.command(name="whoami", help="Show current authentication info")(whoami_command)
app.command(name="write", help="Write a memory entry")(write_command)
app.command(name="search", help="Search memory entries")(search_command)
app.command(name="recall", help="Recall an agent-scoped entry")(recall_command)
app.command(name="stats", help="Show memory statistics")(stats_command)
app.command(name="status", help="Show config and connection health")(status_command)
app.command(name="watch", help="Live-stream memory changes (SSE)")(watch_command)
app.add_typer(snapshot_app, name="snapshot", help="Export and restore memory snapshots")
app.add_typer(inspect_app, name="inspect", help="List, read, and diff memory entries")


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"amfs-cli {version('amfs-cli')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback,
        is_eager=True, help="Show version and exit",
    ),
) -> None:
    """AMFS — Agent Memory File System CLI."""


if __name__ == "__main__":
    app()
