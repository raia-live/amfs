"""CLI commands for managing AMFS connectors."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()

_CONNECTOR_YAML_TEMPLATE = textwrap.dedent("""\
    name: {name}
    version: "0.1.0"
    description: "AMFS connector for {name}"
    author: ""
    license: "Apache-2.0"
    homepage: ""
    tags:
      - {name}

    events:
      - "{name}.event"

    outputs:
      entity_path: "{name}/{{{{event_id}}}}"
      key: "event"

    entry_point: "amfs_connector_{underscored}:{class_name}"

    dependencies: []

    config_schema:
      api_key:
        type: string
        required: true
        description: "API key for {name}"
""")

_PYPROJECT_TEMPLATE = textwrap.dedent("""\
    [project]
    name = "amfs-connector-{name}"
    version = "0.1.0"
    description = "AMFS connector for {name}"
    requires-python = ">=3.11"
    license = "Apache-2.0"
    dependencies = [
        "amfs-connectors>=0.1.0",
    ]

    [project.entry-points."amfs.connectors"]
    {name} = "amfs_connector_{underscored}:{class_name}"

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [tool.hatch.build.targets.wheel]
    packages = ["src/amfs_connector_{underscored}"]
""")

_INIT_PY_TEMPLATE = textwrap.dedent("""\
    \"\"\"AMFS connector for {name}.\"\"\"

    from __future__ import annotations

    from typing import Any
    from uuid import uuid4

    from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult


    class {class_name}(ConnectorABC):
        \"\"\"Connector that ingests events from {name}.\"\"\"

        def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
            event_id = self.extract_event_id(raw_event)
            return [
                IngestionResult(
                    connector_id=self.config.id,
                    event_id=event_id,
                    entity_path=f"{name}/{{event_id}}",
                    key="event",
                    action="write",
                    details=raw_event,
                )
            ]

        def validate_event(self, raw_event: dict[str, Any]) -> bool:
            return isinstance(raw_event, dict)
""")

_TEST_TEMPLATE = textwrap.dedent("""\
    \"\"\"Tests for the {name} connector.\"\"\"

    from __future__ import annotations

    from amfs_connectors.base import ConnectorConfig
    from amfs_connectors.testing import MockAMFS

    from amfs_connector_{underscored} import {class_name}


    def test_transform_basic_event() -> None:
        config = ConnectorConfig(
            name="{name}",
            connector_type="{name}",
            entity_path="{name}",
        )
        connector = {class_name}(config)
        mock = MockAMFS()

        raw_event = {{"type": "{name}.event", "data": {{"key": "value"}}}}
        results = connector.transform(raw_event)
        mock.apply_results(results)

        assert len(results) == 1
        assert results[0].success
        assert results[0].action == "write"
        assert mock.total_operations == 1


    def test_validate_event() -> None:
        config = ConnectorConfig(
            name="{name}",
            connector_type="{name}",
            entity_path="{name}",
        )
        connector = {class_name}(config)

        assert connector.validate_event({{"type": "test"}})
        assert not connector.validate_event("not a dict")  # type: ignore[arg-type]
""")

_README_TEMPLATE = textwrap.dedent("""\
    # amfs-connector-{name}

    AMFS connector for **{name}**.

    ## Installation

    ```bash
    pip install amfs-connector-{name}
    ```

    Or install from source:

    ```bash
    pip install -e .
    ```

    ## Usage

    Once installed, the connector is automatically discovered by AMFS:

    ```bash
    amfs connector list
    ```

    ## Configuration

    Add the connector settings in your AMFS configuration or pass them via
    the `config_schema` fields defined in `connector.yaml`.

    ## Development

    ```bash
    # Run tests
    amfs connector test

    # Validate manifest
    amfs connector validate
    ```

    ## License

    Apache-2.0
""")


def _to_class_name(name: str) -> str:
    """Convert a connector name like 'my-service' to 'MyServiceConnector'."""
    parts = name.replace("_", "-").split("-")
    return "".join(p.capitalize() for p in parts) + "Connector"


def _to_underscored(name: str) -> str:
    """Convert a connector name to a valid Python identifier (underscored)."""
    return name.replace("-", "_")


@app.command()
def init(
    name: str = typer.Argument(..., help="Connector name (e.g. 'slack', 'jira')"),
    directory: Optional[Path] = typer.Option(None, "--dir", "-d", help="Parent directory (default: cwd)"),
) -> None:
    """Scaffold a new connector project."""
    parent = directory or Path.cwd()
    project_dir = parent / f"amfs-connector-{name}"

    if project_dir.exists():
        console.print(f"[red]Directory already exists: {project_dir}[/red]")
        raise typer.Exit(1)

    underscored = _to_underscored(name)
    class_name = _to_class_name(name)
    fmt = dict(name=name, underscored=underscored, class_name=class_name)

    pkg_dir = project_dir / "src" / f"amfs_connector_{underscored}"
    test_dir = project_dir / "tests"

    pkg_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    (project_dir / "connector.yaml").write_text(_CONNECTOR_YAML_TEMPLATE.format(**fmt))
    (project_dir / "pyproject.toml").write_text(_PYPROJECT_TEMPLATE.format(**fmt))
    (project_dir / "README.md").write_text(_README_TEMPLATE.format(**fmt))
    (pkg_dir / "__init__.py").write_text(_INIT_PY_TEMPLATE.format(**fmt))
    (test_dir / f"test_{underscored}.py").write_text(_TEST_TEMPLATE.format(**fmt))

    console.print(f"[green]Scaffolded connector project at {project_dir}[/green]")
    console.print()
    table = Table(title="Generated files")
    table.add_column("File", style="cyan")
    table.add_column("Purpose")
    table.add_row("connector.yaml", "Connector manifest")
    table.add_row("pyproject.toml", "Package configuration with entry point")
    table.add_row(f"src/amfs_connector_{underscored}/__init__.py", "Connector implementation")
    table.add_row(f"tests/test_{underscored}.py", "Tests using MockAMFS")
    table.add_row("README.md", "Usage instructions")
    console.print(table)


@app.command()
def install(
    name: str = typer.Argument(..., help="Connector name or 'github:user/repo'"),
) -> None:
    """Install a connector from PyPI or GitHub."""
    if name.startswith("github:"):
        repo = name.removeprefix("github:")
        target = f"git+https://github.com/{repo}.git"
        console.print(f"[blue]Installing from GitHub: {repo}[/blue]")
    else:
        target = f"amfs-connector-{name}"
        console.print(f"[blue]Installing from PyPI: {target}[/blue]")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", target],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print(f"[green]Successfully installed {target}[/green]")
    else:
        console.print(f"[red]Installation failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)


@app.command(name="list")
def list_connectors() -> None:
    """List installed connectors."""
    from amfs_connectors.registry import ConnectorRegistry

    registry = ConnectorRegistry()
    installed = registry.list_installed()

    if not installed:
        console.print("[yellow]No connectors installed.[/yellow]")
        return

    table = Table(title="Installed Connectors")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Entry Point", style="dim")

    for info in installed:
        table.add_row(
            info["name"],
            info["source"],
            info.get("entry_point", "—"),
        )

    console.print(table)


@app.command()
def discover(
    topic: str = typer.Option("amfs-connector", "--topic", "-t", help="GitHub topic to search"),
) -> None:
    """Search GitHub for community connectors."""
    import httpx

    url = "https://api.github.com/search/repositories"
    params = {"q": f"topic:{topic}", "sort": "stars", "order": "desc", "per_page": 25}

    try:
        resp = httpx.get(url, params=params, timeout=15, headers={"Accept": "application/vnd.github+json"})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]GitHub API error: {exc}[/red]")
        raise typer.Exit(1) from exc

    data = resp.json()
    items = data.get("items", [])

    if not items:
        console.print(f"[yellow]No repositories found with topic '{topic}'.[/yellow]")
        return

    table = Table(title=f"GitHub connectors (topic: {topic})")
    table.add_column("Repository", style="cyan")
    table.add_column("Stars", justify="right")
    table.add_column("Description")
    table.add_column("Install", style="dim")

    for repo in items:
        full_name = repo["full_name"]
        table.add_row(
            full_name,
            str(repo.get("stargazers_count", 0)),
            (repo.get("description") or "")[:60],
            f"amfs connector install github:{full_name}",
        )

    console.print(table)


@app.command()
def test(
    path: Path = typer.Argument(Path("."), help="Connector project directory"),
) -> None:
    """Run connector tests against MockAMFS."""
    test_dir = path / "tests"
    if not test_dir.is_dir():
        console.print(f"[red]No tests/ directory found in {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Running connector tests in {path}…[/blue]")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v"],
        cwd=str(path),
    )

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def validate(
    path: Path = typer.Argument(Path("."), help="Path to connector directory or connector.yaml"),
) -> None:
    """Validate a connector.yaml manifest."""
    from amfs_connectors.manifest import load_manifest, validate_manifest

    try:
        manifest = load_manifest(path)
    except Exception as exc:
        console.print(f"[red]Failed to load manifest: {exc}[/red]")
        raise typer.Exit(1) from exc

    warnings = validate_manifest(path)

    console.print(f"[green]Manifest loaded: {manifest.name} v{manifest.version}[/green]")

    if warnings:
        console.print(f"\n[yellow]{len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  ⚠ {w}")
    else:
        console.print("[green]No warnings — manifest looks good![/green]")
