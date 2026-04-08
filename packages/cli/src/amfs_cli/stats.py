"""amfs stats — display memory statistics."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def stats_command(
    format: str = typer.Option("panel", "--format", "-f", help="Output format: panel, json"),
    remote: bool | None = typer.Option(None, "--remote", "-r", help="Use HTTP API (auto-detected from login)"),
    local: bool = typer.Option(False, "--local", "-L", help="Force local adapter"),
) -> None:
    """Show memory statistics (entries, agents, entities)."""
    from amfs_cli.remote import has_remote_config
    use_remote = (remote is True) or (remote is None and not local and has_remote_config())

    if use_remote:
        from amfs_cli.remote import api_get

        with console.status("[cyan]Fetching stats...[/cyan]"):
            data = api_get("/api/v1/stats")
    else:
        from amfs import AgentMemory

        with console.status("[cyan]Computing stats...[/cyan]"):
            mem = AgentMemory(agent_id="cli")
            s = mem.stats()
            data = {
                "total_entries": s.total_entries,
                "total_entities": s.total_entities,
                "total_agents": s.total_agents,
                "avg_confidence": round(s.avg_confidence, 4) if s.avg_confidence else 0,
                "memory_types": s.memory_types if hasattr(s, "memory_types") else {},
            }

    if format == "json":
        console.print_json(json.dumps(data, default=str))
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Entries", str(data.get("total_entries", 0)))
    table.add_row("Total Entities", str(data.get("total_entities", 0)))
    table.add_row("Total Agents", str(data.get("total_agents", 0)))
    table.add_row("Avg Confidence", f"{data.get('avg_confidence', 0):.4f}")

    types = data.get("memory_types", {})
    if types:
        table.add_row("", "")
        for mt, count in types.items():
            table.add_row(f"  {mt}", str(count))

    console.print(Panel(table, title="[bold]Memory Statistics[/bold]", border_style="cyan"))
