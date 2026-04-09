"""amfs search — search memory entries."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=False)


def search_command(
    query: str = typer.Argument(None, help="Search query text"),
    entity: str | None = typer.Option(None, "--entity", "-e", help="Filter by entity path"),
    agent_id: str | None = typer.Option(None, "--agent", "-a", help="Filter by agent ID"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum confidence"),
    limit: int = typer.Option(50, "--limit", "-l", help="Max results"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
    remote: bool | None = typer.Option(None, "--remote", "-r", help="Use HTTP API (auto-detected from login)"),
    local: bool = typer.Option(False, "--local", "-L", help="Force local adapter"),
) -> None:
    """Search memory entries by query, entity, agent, or confidence."""
    from amfs_cli.remote import has_remote_config
    use_remote = (remote is True) or (remote is None and not local and has_remote_config())

    if use_remote:
        from amfs_cli.remote import api_post

        with console.status("[cyan]Searching...[/cyan]"):
            results = api_post("/api/v1/search", {
                "query": query,
                "entity_path": entity,
                "agent_id": agent_id,
                "min_confidence": min_confidence,
                "limit": limit,
            })

        if format == "json":
            console.print_json(json.dumps(results, default=str))
            return

        if not results:
            console.print("[dim]No results found.[/dim]")
            return

        table = Table(title=f"Search Results ({len(results)} entries)")
        table.add_column("Entity", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Confidence", justify="right")
        table.add_column("Agent", style="magenta")
        table.add_column("Value", max_width=50)

        for r in results:
            table.add_row(
                r.get("entity_path", ""),
                r.get("key", ""),
                f"{r.get('confidence', 0):.4f}",
                r.get("agent_id", ""),
                str(r.get("value", ""))[:50],
            )
        console.print(table)
    else:
        from amfs import AgentMemory

        with console.status("[cyan]Searching...[/cyan]"):
            mem = AgentMemory(agent_id="cli")
            results = mem.search(
                query=query,
                entity_path=entity,
                agent_id=agent_id,
                min_confidence=min_confidence,
                limit=limit,
            )

        if format == "json":
            console.print_json(json.dumps(
                [{"entity_path": e.entity_path, "key": e.key, "value": e.value,
                  "confidence": e.confidence, "agent_id": e.provenance.agent_id}
                 for e in results],
                default=str,
            ))
            return

        if not results:
            console.print("[dim]No results found.[/dim]")
            return

        table = Table(title=f"Search Results ({len(results)} entries)")
        table.add_column("Entity", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Confidence", justify="right")
        table.add_column("Agent", style="magenta")
        table.add_column("Value", max_width=50)

        for e in results:
            table.add_row(
                e.entity_path,
                e.key,
                f"{e.confidence:.4f}",
                e.provenance.agent_id,
                str(e.value)[:50],
            )
        console.print(table)
