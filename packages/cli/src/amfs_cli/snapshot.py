"""Snapshot CLI commands — export and restore."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from amfs.config import load_config_or_default
from amfs.factory import create_adapter_from_config
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output snapshot file path"),
    config: Path | None = typer.Option(None, "--config", "-c", help="AMFS config file"),
    entity: str | None = typer.Option(None, "--entity", "-e", help="Filter to entity path"),
    include_superseded: bool = typer.Option(False, "--superseded", help="Include superseded versions"),
) -> None:
    """Export current memory state to a JSON snapshot."""
    cfg = load_config_or_default(config)
    adapter = create_adapter_from_config(cfg)
    exporter = SnapshotExporter(adapter)
    count = exporter.export(output, entity_path=entity, include_superseded=include_superseded)
    console.print(f"[green]Exported {count} entries to {output}[/green]")


@app.command()
def restore(
    input_path: Path = typer.Argument(..., help="Snapshot file to restore"),
    config: Path | None = typer.Option(None, "--config", "-c", help="AMFS config file"),
) -> None:
    """Restore memory entries from a snapshot file."""
    cfg = load_config_or_default(config)
    adapter = create_adapter_from_config(cfg)
    importer = SnapshotImporter(adapter)
    count = importer.restore(input_path)
    console.print(f"[green]Restored {count} entries from {input_path}[/green]")
