"""amfs login — store credentials for remote AMFS access."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from amfs_cli.remote import DEFAULT_API_URL, save_credentials

console = Console()


def login_command(
    url: str = typer.Option(
        DEFAULT_API_URL, "--url", "-u",
        help="AMFS HTTP API URL (defaults to production)",
    ),
    api_key: str = typer.Option(
        ..., "--key", "-k",
        prompt="API key (from amfs.sense-lab.ai/settings/api-keys)",
        hide_input=True,
        help="AMFS API key",
    ),
) -> None:
    """Authenticate with an AMFS server and store credentials locally.

    By default connects to the AMFS production API. Use --url to point
    to a self-hosted instance.

    Get your API key at https://amfs.sense-lab.ai/settings/api-keys
    """
    import httpx

    with console.status(f"[cyan]Verifying credentials against {url}...[/cyan]"):
        try:
            with httpx.Client(
                base_url=url.rstrip("/"),
                headers={"X-AMFS-API-Key": api_key},
                timeout=10.0,
            ) as client:
                resp = client.get("/api/v1/auth/whoami")
                resp.raise_for_status()
                info = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                console.print("[red]Authentication failed — invalid or expired API key.[/red]")
            elif exc.response.status_code == 404:
                console.print(
                    "[yellow]Server does not support /api/v1/auth/whoami yet.[/yellow]\n"
                    "Falling back to basic health check..."
                )
                try:
                    with httpx.Client(
                        base_url=url.rstrip("/"),
                        headers={"X-AMFS-API-Key": api_key},
                        timeout=10.0,
                    ) as client2:
                        resp2 = client2.get("/health")
                        resp2.raise_for_status()
                        info = {"authenticated": True, "mode": "unknown"}
                except Exception as exc2:
                    console.print(f"[red]Connection failed:[/red] {exc2}")
                    raise typer.Exit(code=1)
            else:
                console.print(f"[red]Server error:[/red] HTTP {exc.response.status_code}")
                raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[red]Connection failed:[/red] {exc}")
            raise typer.Exit(code=1)

    if not info.get("authenticated"):
        console.print("[red]API key was not accepted by the server.[/red]")
        raise typer.Exit(code=1)

    path = save_credentials(url, api_key)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")

    is_default = url == DEFAULT_API_URL
    table.add_row("Server", f"[cyan]{url}[/cyan]{' (production)' if is_default else ''}")
    table.add_row("Key", f"[dim]{api_key[:12]}{'•' * 20}[/dim]")
    table.add_row("Mode", info.get("mode", "unknown"))

    if info.get("mode") == "pro":
        table.add_row("Account", info.get("account_id", "—"))
        table.add_row("Key Type", info.get("key_type", "—"))
        table.add_row("Admin", "[green]yes[/green]" if info.get("is_admin") else "no")
        table.add_row("Rate Limit", f"{info.get('rate_limit_rpm', '?')} RPM")
        scopes = info.get("scopes", [])
        if scopes:
            scope_str = ", ".join(
                f"{s['entity_path_pattern']} ({s['permission']})"
                for s in scopes
            )
            table.add_row("Scopes", scope_str)

    table.add_row("Saved to", str(path))

    console.print(Panel(
        table,
        title="[bold green]Login successful[/bold green]",
        border_style="green",
    ))
