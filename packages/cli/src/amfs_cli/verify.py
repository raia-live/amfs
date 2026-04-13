"""amfs verify — check content integrity of the memory store."""

from __future__ import annotations

import typer

from amfs import AgentMemory


def verify_command(
    entity_path: str = typer.Argument(None, help="Verify only entries under this path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details for each issue"),
) -> None:
    """Verify content hashes and integrity chains in the memory store."""
    with AgentMemory(agent_id="amfs-cli/verify") as mem:
        report = mem.verify(entity_path)

    total = report["total_checked"]
    valid = report["valid"]
    corrupted = report["corrupted"]
    chain_breaks = report["chain_breaks"]

    typer.echo(f"Checked {total} entries: {valid} valid", nl=False)

    if corrupted:
        typer.echo(f", {len(corrupted)} CORRUPTED", nl=False)
    if chain_breaks:
        typer.echo(f", {len(chain_breaks)} chain breaks", nl=False)

    typer.echo()

    if not corrupted and not chain_breaks:
        typer.secho("All entries OK.", fg=typer.colors.GREEN)
        return

    if corrupted:
        typer.secho(f"\nCorrupted entries ({len(corrupted)}):", fg=typer.colors.RED)
        for item in corrupted:
            typer.echo(f"  {item['entity_path']}/{item['key']} v{item['version']}")
            if verbose:
                typer.echo(f"    expected: {item['expected_hash'][:16]}...")
                typer.echo(f"    actual:   {item['actual_hash'][:16]}...")

    if chain_breaks:
        typer.secho(f"\nChain breaks ({len(chain_breaks)}):", fg=typer.colors.YELLOW)
        for item in chain_breaks:
            typer.echo(f"  {item['entity_path']}/{item['key']} v{item['version']}")
            if verbose:
                typer.echo(f"    expected: {item['expected_chain'][:16]}...")
                typer.echo(f"    actual:   {item['actual_chain'][:16]}...")

    raise typer.Exit(code=1)
