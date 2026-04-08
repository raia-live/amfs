"""amfs status — show configuration and connection health."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def status_command() -> None:
    """Show current AMFS configuration, adapter, and connection health."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    config_path = Path("amfs.yaml")
    table.add_row("Config file", str(config_path) if config_path.exists() else "[dim]not found[/dim]")

    http_url = os.environ.get("AMFS_HTTP_URL")
    api_key = os.environ.get("AMFS_API_KEY")
    pg_dsn = os.environ.get("AMFS_POSTGRES_DSN")
    data_dir = os.environ.get("AMFS_DATA_DIR")

    cred_file = Path.home() / ".config" / "amfs" / "credentials.json"
    if cred_file.exists():
        import json
        creds = json.loads(cred_file.read_text(encoding="utf-8"))
        if not http_url:
            http_url = creds.get("url")
        if not api_key:
            api_key = creds.get("api_key")

    if http_url:
        table.add_row("Mode", "[cyan]Remote (HTTP API)[/cyan]")
        table.add_row("Server URL", f"[green]{http_url}[/green]")
        table.add_row("API Key", f"[dim]{api_key[:12]}{'•' * 20}[/dim]" if api_key else "[yellow]not set[/yellow]")
    elif pg_dsn:
        table.add_row("Mode", "[cyan]Postgres[/cyan]")
        host = pg_dsn.split("@")[-1].split("/")[0] if "@" in pg_dsn else pg_dsn[:40]
        table.add_row("DSN", f"[dim]{host}[/dim]")
    elif data_dir:
        table.add_row("Mode", "[cyan]Filesystem[/cyan]")
        table.add_row("Data dir", data_dir)
    elif config_path.exists():
        table.add_row("Mode", "[cyan]Config file[/cyan]")
    else:
        table.add_row("Mode", "[yellow]Default (filesystem .amfs/)[/yellow]")

    if http_url:
        with console.status("[cyan]Checking connection...[/cyan]"):
            try:
                import httpx
                headers = {"X-AMFS-API-Key": api_key} if api_key else {}
                resp = httpx.get(f"{http_url.rstrip('/')}/api/v1/health", headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    table.add_row("Connection", "[green]Healthy[/green]")
                else:
                    table.add_row("Connection", f"[yellow]HTTP {resp.status_code}[/yellow]")
            except Exception as exc:
                table.add_row("Connection", f"[red]Failed: {exc}[/red]")

    console.print(Panel(table, title="[bold]AMFS Status[/bold]", border_style="cyan"))
