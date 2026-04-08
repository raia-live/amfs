"""amfs watch — live-stream memory changes via SSE."""

from __future__ import annotations

import json

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


def watch_command(
    entity: str | None = typer.Argument(None, help="Filter to entity path (default: all)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max rows to display"),
) -> None:
    """Live-stream memory changes from a remote AMFS server (SSE)."""
    from amfs_cli.remote import get_client

    client = get_client()
    params = {"entity_path": entity or "*"}

    events: list[dict] = []

    def _build_table() -> Table:
        table = Table(title="Live Memory Stream", expand=True)
        table.add_column("Entity", style="cyan", max_width=30)
        table.add_column("Key", style="green", max_width=20)
        table.add_column("Agent", style="magenta", max_width=15)
        table.add_column("Version", justify="right")
        table.add_column("Value", max_width=40)
        for ev in events[-limit:]:
            table.add_row(
                ev.get("entity_path", ""),
                ev.get("key", ""),
                ev.get("agent_id", ""),
                str(ev.get("version", "")),
                str(ev.get("value", ""))[:40],
            )
        return table

    console.print(f"[cyan]Connecting to SSE stream...[/cyan] (entity: {entity or '*'})")
    console.print("[dim]Press Ctrl-C to stop[/dim]\n")

    try:
        with Live(_build_table(), console=console, refresh_per_second=2) as live:
            with client.stream(
                "GET", "/api/v1/stream", params=params, timeout=None
            ) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    while "\n\n" in buf:
                        msg, buf = buf.split("\n\n", 1)
                        for line in msg.split("\n"):
                            if line.startswith("data:"):
                                raw = line[5:].strip()
                                if raw:
                                    try:
                                        events.append(json.loads(raw))
                                        live.update(_build_table())
                                    except json.JSONDecodeError:
                                        pass
    except KeyboardInterrupt:
        console.print(f"\n[dim]Stopped. Received {len(events)} events.[/dim]")
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Stream error:[/red] {exc.response.status_code}")
        raise typer.Exit(code=1)
    finally:
        client.close()
