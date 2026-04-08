"""amfs login — store credentials for remote AMFS access."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from amfs_cli.remote import save_credentials, get_client

console = Console()


def login_command(
    url: str = typer.Option(..., "--url", "-u", prompt="AMFS server URL", help="AMFS HTTP API URL"),
    api_key: str = typer.Option(..., "--key", "-k", prompt="API key", hide_input=True, help="AMFS API key"),
) -> None:
    """Store AMFS credentials for remote CLI access."""
    with console.status("[cyan]Verifying connection...[/cyan]"):
        try:
            client = get_client.__wrapped__ if hasattr(get_client, "__wrapped__") else None  # noqa: F841
            import httpx
            with httpx.Client(
                base_url=url.rstrip("/"),
                headers={"X-AMFS-API-Key": api_key},
                timeout=10.0,
            ) as client:
                resp = client.get("/api/v1/health")
                resp.raise_for_status()
        except Exception as exc:
            console.print(f"[red]Connection failed:[/red] {exc}")
            raise typer.Exit(code=1)

    path = save_credentials(url, api_key)
    console.print(
        Panel(
            f"[green]Credentials saved to[/green] {path}\n\n"
            f"  Server: [cyan]{url}[/cyan]\n"
            f"  Key:    [dim]{api_key[:12]}{'•' * 20}[/dim]",
            title="[bold green]Login successful[/bold green]",
            border_style="green",
        )
    )
