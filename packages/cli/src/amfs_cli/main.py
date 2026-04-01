"""AMFS CLI entry point."""

from __future__ import annotations

import typer

from amfs_cli.init import init_command
from amfs_cli.snapshot import app as snapshot_app
from amfs_cli.inspect import app as inspect_app

app = typer.Typer(
    name="amfs",
    help="AMFS — Agent Memory File System CLI",
    no_args_is_help=True,
)

app.command(name="init", help="Initialise an AMFS project")(init_command)
app.add_typer(snapshot_app, name="snapshot", help="Export and restore memory snapshots")
app.add_typer(inspect_app, name="inspect", help="List, read, and diff memory entries")


if __name__ == "__main__":
    app()
