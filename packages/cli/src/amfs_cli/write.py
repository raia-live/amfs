"""amfs write — write a memory entry."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

console = Console()


def write_command(
    entity: str = typer.Argument(..., help="Entity path (e.g. myapp/auth)"),
    key: str = typer.Argument(..., help="Memory key"),
    value: str = typer.Argument(
        None, help="JSON value (reads from stdin if omitted)"
    ),
    agent_id: str = typer.Option("cli", "--agent", "-a", help="Agent ID"),
    confidence: float = typer.Option(1.0, "--confidence", "-c", help="Confidence score"),
    memory_type: str = typer.Option("fact", "--type", "-t", help="Memory type: fact, belief, experience"),
    remote: bool = typer.Option(False, "--remote", "-r", help="Write via HTTP API instead of local adapter"),
) -> None:
    """Write a memory entry to AMFS."""
    if value is None:
        if sys.stdin.isatty():
            console.print("[dim]Enter JSON value (Ctrl-D to finish):[/dim]")
        value = sys.stdin.read().strip()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value

    if remote:
        from amfs_cli.remote import api_post

        with console.status("[cyan]Writing entry...[/cyan]"):
            result = api_post("/api/v1/entries", {
                "entity_path": entity,
                "key": key,
                "value": parsed,
                "confidence": confidence,
                "memory_type": memory_type,
            })
        console.print(f"[green]Written[/green] {entity}/{key} v{result.get('version', '?')}")
    else:
        from amfs import AgentMemory

        with console.status("[cyan]Writing entry...[/cyan]"):
            mem = AgentMemory(agent_id=agent_id)
            entry = mem.write(entity, key, parsed, confidence=confidence, memory_type=memory_type)
        console.print(f"[green]Written[/green] {entity}/{key} v{entry.version}")
