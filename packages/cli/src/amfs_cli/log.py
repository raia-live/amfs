"""amfs log — show commit history."""

from __future__ import annotations

import json

import typer

from amfs import AgentMemory


def log_command(
    limit: int = typer.Option(20, "--limit", "-n", help="Max commits to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show the commit log — atomic groups of memory writes."""
    with AgentMemory(agent_id="amfs-cli/log") as mem:
        commits = mem.commit_log(limit=limit)

    if not commits:
        typer.echo("No commits found.")
        return

    if json_output:
        typer.echo(json.dumps(
            [c.model_dump(mode="json") for c in commits],
            indent=2,
            default=str,
        ))
        return

    for c in commits:
        typer.secho(f"commit {c.id}", fg=typer.colors.YELLOW, bold=True)
        typer.echo(f"Author: {c.author_agent_id}")
        typer.echo(f"Date:   {c.created_at.isoformat()}")
        typer.echo(f"Branch: {c.branch}")
        if c.message:
            typer.echo(f"\n    {c.message}\n")
        else:
            typer.echo()
        if c.entries:
            for entry in c.entries:
                typer.echo(f"  {entry.entity_path}/{entry.key} v{entry.version}")
        typer.echo()
