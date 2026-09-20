"""amfs replay — answer SenseLab's replay requests from the command line.

``amfs replay serve`` starts a receiver for the repair loop's Tier 2 webhook
around a runner you name (``module:function``), on the branch each request
carries; ``amfs replay simulate`` sends one signed request to a receiver so
the whole path can be exercised without the SaaS side.
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()

SECRET_ENV = "AMFS_REPLAY_SECRET"


def load_runner(spec: str) -> Any:
    """Import ``package.module:callable``. The callable is the
    :data:`amfs.replay.Runner`: ``run(task_input, memory) -> answer``."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise typer.BadParameter(
            "expected module:function, e.g. my_agent.replay:run", param_hint="--run"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise typer.BadParameter(
            f"could not import {module_name!r}: {exc}", param_hint="--run"
        ) from exc
    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise typer.BadParameter(f"{module_name} has no attribute {attr!r}", param_hint="--run")
    if not callable(target):
        raise typer.BadParameter(f"{spec} is not callable", param_hint="--run")
    return target


def _secret(option: str | None, *, allow_none: bool) -> str | None:
    secret = option if option is not None else os.environ.get(SECRET_ENV)
    if secret:
        return secret
    if allow_none:
        return None
    raise typer.BadParameter(
        f"no secret: pass --secret or set {SECRET_ENV} (or --insecure to accept unsigned requests)",
        param_hint="--secret",
    )


@app.command()
def serve(
    run: str = typer.Option(
        ..., "--run", "-r", help="Runner as module:function — run(task_input, memory) -> answer"
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Interface to bind"),
    port: int = typer.Option(8787, "--port", "-p", help="Port to listen on"),
    path: str = typer.Option(
        "/amfs/replay", "--path", help="URL path the webhook is registered on"
    ),
    secret: str | None = typer.Option(
        None, "--secret", help=f"Shared webhook secret (default: ${SECRET_ENV})"
    ),
    insecure: bool = typer.Option(False, "--insecure", help="Accept unsigned requests (no secret)"),
    agent_id: str | None = typer.Option(
        None, "--agent", "-a", help="Agent id when a request names none"
    ),
    inline: bool = typer.Option(
        False,
        "--inline",
        help="Run inside the request and answer 200 (default: 202 and a worker thread)",
    ),
    timeout: float = typer.Option(600.0, "--timeout", help="Seconds a background run may take"),
) -> None:
    """Serve a replay receiver around RUN until interrupted.

    Each request runs RUN(task_input, memory) with memory on the request's
    branch and commits the outcome with the attributes the grader reads.
    """
    from amfs.replay import ReplayReceiver
    from amfs.replay import serve as _serve

    runner = load_runner(run)
    shared = _secret(secret, allow_none=insecure)
    if shared is None:
        console.print("[yellow]--insecure: accepting unsigned requests[/yellow]")
    receiver = ReplayReceiver(
        secret=shared,
        run=runner,
        agent_id=agent_id,
        background=not inline,
        run_timeout=timeout,
    )

    def _ready(url: str) -> None:
        console.print(f"[green]amfs replay receiver[/green] listening at [bold]{url}[/bold]")
        console.print(
            f"[dim]runner {run} · {'inline' if inline else 'background'} runs · "
            f"{'signed' if shared else 'unsigned'} requests[/dim]"
        )

    _serve(receiver, host=host, port=port, path=path, ready=_ready)


@app.command()
def simulate(
    url: str = typer.Argument(..., help="Receiver URL, e.g. http://localhost:8787/amfs/replay"),
    task: str = typer.Option(
        "customer says the invoice email never arrived", "--task", "-t", help="task_input to send"
    ),
    branch: str = typer.Option(
        "repair/simulated", "--branch", "-b", help="Memory branch the run must read"
    ),
    agent_id: str = typer.Option(
        "simulated-agent", "--agent", "-a", help="agent_id on the request"
    ),
    secret: str | None = typer.Option(
        None, "--secret", help=f"Shared webhook secret (default: ${SECRET_ENV})"
    ),
    insecure: bool = typer.Option(False, "--insecure", help="Send unsigned"),
    ping: bool = typer.Option(False, "--ping", help="Send a ping instead of a replay request"),
    payload_file: Path | None = typer.Option(
        None, "--payload", help="JSON file to send as the body instead"
    ),
    expected: str | None = typer.Option(None, "--expected", help='JSON for the "expected" field'),
) -> None:
    """Send one signed replay request (or ping) to a receiver and print the answer."""
    import httpx
    from amfs.replay import DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, sign_replay_body

    shared = _secret(secret, allow_none=insecure)
    if payload_file is not None:
        payload = json.loads(payload_file.read_text())
    elif ping:
        payload = {
            "event": "ping",
            "delivery_id": str(uuid4()),
            "sent_at": datetime.now(UTC).isoformat(),
            "agent_id": agent_id,
        }
    else:
        payload = {
            "event": "replay_requested",
            "delivery_id": str(uuid4()),
            "sent_at": datetime.now(UTC).isoformat(),
            "fix_id": str(uuid4()),
            "agent_id": agent_id,
            "branch": branch,
            "case_id": str(uuid4()),
            "case_set_id": str(uuid4()),
            "task_input": task,
            "expected": json.loads(expected) if expected else {},
            "deadline_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "instructions": "Simulated by `amfs replay simulate`.",
        }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        EVENT_HEADER: str(payload.get("event", "replay_requested")),
        DELIVERY_HEADER: str(payload.get("delivery_id", "")),
    }
    if shared:
        headers[SIGNATURE_HEADER] = sign_replay_body(shared, body)
    response = httpx.post(url, content=body, headers=headers, timeout=30.0)
    colour = "green" if response.is_success else "red"
    console.print(f"[{colour}]HTTP {response.status_code}[/{colour}]")
    try:
        console.print_json(json.dumps(response.json()))
    except ValueError:
        console.print(response.text)
    if not response.is_success:
        raise typer.Exit(code=1)
