"""amfs recall — recall agent-scoped memory."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def recall_command(
    entity: str = typer.Argument(..., help="Entity path"),
    key: str = typer.Argument(..., help="Memory key"),
    agent_id: str = typer.Option("cli", "--agent", "-a", help="Agent ID to recall as"),
    format: str = typer.Option("panel", "--format", "-f", help="Output format: panel, json"),
    remote: bool | None = typer.Option(None, "--remote", "-r", help="Use HTTP API (auto-detected from login)"),
    local: bool = typer.Option(False, "--local", "-L", help="Force local adapter"),
) -> None:
    """Recall an agent-scoped memory entry."""
    from amfs_cli.remote import has_remote_config
    use_remote = (remote is True) or (remote is None and not local and has_remote_config())

    if use_remote:
        from amfs_cli.remote import api_get

        with console.status("[cyan]Recalling...[/cyan]"):
            result = api_get(
                f"/api/v1/entries/{entity}/{key}",
                params={"agent_id": agent_id},
            )

        if format == "json":
            console.print_json(json.dumps(result, default=str))
            return

        if not result:
            console.print(f"[dim]No entry found for {entity}/{key}[/dim]")
            return

        console.print(Panel(
            f"[bold]{entity}/{key}[/bold] v{result.get('version', '?')}\n"
            f"Confidence: [green]{result.get('confidence', 0):.4f}[/green]\n"
            f"Agent: [magenta]{result.get('agent_id', '')}[/magenta]\n\n"
            f"{json.dumps(result.get('value'), indent=2, default=str)}",
            title="[bold cyan]Recall[/bold cyan]",
            border_style="cyan",
        ))
    else:
        from amfs import AgentMemory

        with console.status("[cyan]Recalling...[/cyan]"):
            mem = AgentMemory(agent_id=agent_id)
            entry = mem.recall(entity, key)

        if entry is None:
            console.print(f"[dim]No entry found for {entity}/{key}[/dim]")
            raise typer.Exit(code=1)

        if format == "json":
            console.print_json(json.dumps({
                "entity_path": entry.entity_path, "key": entry.key,
                "value": entry.value, "version": entry.version,
                "confidence": entry.confidence,
                "agent_id": entry.provenance.agent_id,
            }, default=str))
            return

        console.print(Panel(
            f"[bold]{entity}/{key}[/bold] v{entry.version}\n"
            f"Confidence: [green]{entry.confidence:.4f}[/green]\n"
            f"Agent: [magenta]{entry.provenance.agent_id}[/magenta]\n\n"
            f"{json.dumps(entry.value, indent=2, default=str)}",
            title="[bold cyan]Recall[/bold cyan]",
            border_style="cyan",
        ))
