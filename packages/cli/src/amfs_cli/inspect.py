"""Inspect CLI commands — list, read, and diff memory entries."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from amfs.config import load_config_or_default
from amfs.factory import create_adapter_from_config

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_entries(
    entity: str | None = typer.Argument(None, help="Filter to entity path"),
    config: Path | None = typer.Option(None, "--config", "-c", help="AMFS config file"),
    superseded: bool = typer.Option(False, "--superseded", help="Include superseded versions"),
) -> None:
    """List memory entries."""
    cfg = load_config_or_default(config)
    adapter = create_adapter_from_config(cfg)
    entries = adapter.list(entity, include_superseded=superseded)

    if not entries:
        console.print("[dim]No entries found.[/dim]")
        return

    table = Table(title="Memory Entries")
    table.add_column("Entity", style="cyan")
    table.add_column("Key", style="green")
    table.add_column("Version", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Agent", style="magenta")
    table.add_column("Written At")

    for e in entries:
        table.add_row(
            e.entity_path,
            e.key,
            str(e.version),
            f"{e.confidence:.4f}",
            e.provenance.agent_id,
            e.provenance.written_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


@app.command()
def read(
    entity: str = typer.Argument(..., help="Entity path"),
    key: str = typer.Argument(..., help="Key name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="AMFS config file"),
) -> None:
    """Read a specific memory entry and print its value."""
    cfg = load_config_or_default(config)
    adapter = create_adapter_from_config(cfg)
    entry = adapter.read(entity, key)

    if entry is None:
        console.print(f"[red]Entry not found: {entity}/{key}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{entity}/{key}[/bold] v{entry.version}")
    console.print(f"Confidence: {entry.confidence:.4f}")
    console.print(f"Agent: {entry.provenance.agent_id} ({entry.provenance.session_id})")
    console.print(f"Written: {entry.provenance.written_at}")
    if entry.ttl_at:
        console.print(f"TTL: {entry.ttl_at}")
    console.print()
    console.print_json(json.dumps(entry.value, default=str))


@app.command()
def diff(
    entity: str = typer.Argument(..., help="Entity path"),
    key: str = typer.Argument(..., help="Key name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="AMFS config file"),
) -> None:
    """Show version history diff for a key."""
    cfg = load_config_or_default(config)
    adapter = create_adapter_from_config(cfg)
    entries = adapter.list(entity, include_superseded=True)

    # Filter to just this key and sort by version
    key_entries = sorted(
        [e for e in entries if e.key == key],
        key=lambda e: e.version,
    )

    if not key_entries:
        console.print(f"[red]No entries found for {entity}/{key}[/red]")
        raise typer.Exit(code=1)

    for i, entry in enumerate(key_entries):
        console.print(f"[bold]v{entry.version}[/bold] (confidence: {entry.confidence:.4f})")
        console.print(f"  Agent: {entry.provenance.agent_id}")
        console.print(f"  Written: {entry.provenance.written_at}")

        if i > 0:
            prev_val = json.dumps(key_entries[i - 1].value, sort_keys=True, default=str)
            curr_val = json.dumps(entry.value, sort_keys=True, default=str)
            if prev_val != curr_val:
                console.print(f"  [red]- {prev_val}[/red]")
                console.print(f"  [green]+ {curr_val}[/green]")
            else:
                console.print("  [dim](value unchanged)[/dim]")
        console.print()
