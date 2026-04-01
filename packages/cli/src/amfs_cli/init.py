"""amfs init — scaffold an AMFS project in one command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()

_DEFAULT_CONFIG = """\
# AMFS Configuration
# Docs: https://github.com/raia-live/amfs

namespace: {namespace}

layers:
  primary:
    adapter: filesystem
    options:
      root: .amfs
"""

_GITIGNORE = """\
# AMFS local memory store
.amfs/
"""


def init_command(
    directory: Path = typer.Argument(
        Path("."), help="Directory to initialise (default: current dir)"
    ),
    namespace: str = typer.Option(
        "default", "--namespace", "-n", help="Namespace for this project"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config"
    ),
) -> None:
    """Initialise an AMFS project — creates config file and data directory."""
    directory = directory.resolve()
    config_path = directory / "amfs.yaml"
    amfs_dir = directory / ".amfs"

    if config_path.exists() and not force:
        console.print(
            f"[yellow]amfs.yaml already exists at {config_path}. "
            f"Use --force to overwrite.[/yellow]"
        )
        raise typer.Exit(code=1)

    directory.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_DEFAULT_CONFIG.format(namespace=namespace), encoding="utf-8")
    amfs_dir.mkdir(exist_ok=True)

    gitignore = directory / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".amfs/" not in content:
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write("\n" + _GITIGNORE)
            console.print("[dim]Updated .gitignore with .amfs/[/dim]")
    else:
        gitignore.write_text(_GITIGNORE, encoding="utf-8")
        console.print("[dim]Created .gitignore[/dim]")

    console.print("[green bold]AMFS initialised![/green bold]")
    console.print(f"  Config:    {config_path}")
    console.print(f"  Data dir:  {amfs_dir}")
    console.print(f"  Namespace: {namespace}")
    console.print()
    console.print("[dim]Quick start:[/dim]")
    console.print('  from amfs import AgentMemory')
    console.print('  mem = AgentMemory(agent_id="my-agent")')
    console.print('  mem.write("my-service", "key", {"hello": "world"})')
