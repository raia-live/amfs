"""amfs whoami — show current authentication info."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from amfs_cli.remote import api_get

console = Console()


def whoami_command(
    format: str = typer.Option("panel", "--format", "-f", help="Output format: panel, json"),
) -> None:
    """Show who you are authenticated as on the remote AMFS server."""
    with console.status("[cyan]Checking authentication...[/cyan]"):
        try:
            info = api_get("/api/v1/auth/whoami")
        except Exception as exc:
            console.print(f"[red]Failed to reach server:[/red] {exc}")
            raise typer.Exit(code=1)

    if format == "json":
        console.print_json(json.dumps(info, default=str))
        return

    if not info.get("authenticated"):
        console.print(Panel(
            "[yellow]Not authenticated.[/yellow]\n\n"
            "Run [bold]amfs login[/bold] to connect to an AMFS server.",
            title="[bold]AMFS Auth[/bold]",
            border_style="yellow",
        ))
        raise typer.Exit(code=1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Mode", info.get("mode", "unknown"))

    if info.get("mode") == "pro":
        table.add_row("Account", info.get("account_id", "—"))
        table.add_row("Key Type", _format_key_type(info.get("key_type")))
        table.add_row("Admin", "[green]yes[/green]" if info.get("is_admin") else "no")
        table.add_row("Rate Limit", f"{info.get('rate_limit_rpm', '?')} RPM")

        scopes = info.get("scopes", [])
        if scopes:
            for i, s in enumerate(scopes):
                label = "Scopes" if i == 0 else ""
                perm = s.get("permission", "?")
                pattern = s.get("entity_path_pattern", "*")
                table.add_row(label, f"[cyan]{pattern}[/cyan] → {perm}")
        else:
            table.add_row("Scopes", "[dim]none[/dim]")
    else:
        table.add_row("Auth", "[green]authenticated[/green]")

    console.print(Panel(
        table,
        title="[bold cyan]AMFS Identity[/bold cyan]",
        border_style="cyan",
    ))


def _format_key_type(kt: str | None) -> str:
    if not kt:
        return "—"
    colors = {"admin": "red", "agent": "green", "ingestion": "yellow", "readonly": "dim"}
    color = colors.get(kt, "white")
    return f"[{color}]{kt}[/{color}]"
