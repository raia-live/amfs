#!/usr/bin/env python3
"""Post-publish live smoke test for a published MCP server.

Unlike the in-process tool-surface smoke (which imports the package directly),
this drives the *published* server exactly as a real client does:

  1. spawns it with ``uvx <package>[@<version>]`` — resolving the package and
     all of its dependencies from PyPI, the same resolution end users get;
  2. speaks the real MCP protocol over stdio (initialize + list_tools +
     call_tool); and
  3. calls a handful of read tools against a real backend (AMFS_HTTP_URL +
     AMFS_API_KEY), asserting each returns without an error.

This is the only layer that proves "the thing we just published actually works
when a client runs it", catching dependency-resolution skew, missing tools,
transport/schema breakage, and backend-auth regressions that in-process tests
cannot see.

Env:
  MCP_PACKAGE       PyPI package to run (e.g. amfs-mcp-server-pro). Required.
  MCP_VERSION       Optional exact version to pin (e.g. 0.1.23). Recommended in
                    CI so the smoke targets the release just published.
  AMFS_HTTP_URL     Backend URL the server should talk to. Required.
  AMFS_API_KEY      API key for that backend. Required.
  MCP_SMOKE_TIMEOUT Per-call timeout seconds (default 60).

Exit code 0 = all probed tools responded without error; non-zero = failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - dependency is installed in CI
    sys.stderr.write(
        "The 'mcp' package is required. Install it with: pip install mcp\n"
    )
    raise


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.stderr.write(f"Missing required env var: {name}\n")
        sys.exit(2)
    return val


# Read tools that must respond without error against a live backend. These are
# side-effect-free and exercise identity, aggregate stats, semantic recall
# (the include_artifacts path), and keyword search (the search-recall path).
_PROBES: list[tuple[str, dict]] = [
    ("amfs_whoami", {}),
    ("amfs_stats", {}),
    ("amfs_retrieve", {"query": "live smoke test probe", "limit": 3}),
    ("amfs_search", {"query": "smoke", "limit": 3}),
]


def _extract_text(result) -> str:
    """Best-effort flatten of a call_tool result's content into text."""
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def _run() -> int:
    package = _require("MCP_PACKAGE")
    version = os.environ.get("MCP_VERSION", "").strip()
    spec = f"{package}@{version}" if version else package
    timeout = float(os.environ.get("MCP_SMOKE_TIMEOUT", "60"))

    # Pass the backend creds through to the spawned server. --refresh forces uvx
    # to re-resolve from PyPI instead of serving a stale cached build.
    child_env = dict(os.environ)
    child_env["AMFS_HTTP_URL"] = _require("AMFS_HTTP_URL")
    child_env["AMFS_API_KEY"] = _require("AMFS_API_KEY")

    params = StdioServerParameters(
        command="uvx",
        args=["--refresh", spec],
        env=child_env,
    )

    print(f"[live-smoke] launching: uvx --refresh {spec}")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)

            listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            tool_names = {t.name for t in listed.tools}
            print(f"[live-smoke] server advertised {len(tool_names)} tools")

            missing = [name for name, _ in _PROBES if name not in tool_names]
            if missing:
                sys.stderr.write(
                    f"[live-smoke] FAIL: published server is missing tools: {missing}\n"
                )
                return 1

            failures: list[str] = []
            for name, args in _PROBES:
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(name, args), timeout=timeout
                    )
                except Exception as exc:  # noqa: BLE001 - report any transport/tool error
                    failures.append(f"{name}: raised {type(exc).__name__}: {exc}")
                    continue

                if getattr(result, "isError", False):
                    failures.append(f"{name}: isError=True -> {_extract_text(result)[:300]}")
                    continue

                text = _extract_text(result)
                try:
                    json.loads(text)  # tools return JSON strings
                except (json.JSONDecodeError, TypeError):
                    failures.append(f"{name}: non-JSON response -> {text[:200]!r}")
                    continue

                print(f"[live-smoke] OK  {name}")

            if failures:
                sys.stderr.write("[live-smoke] FAIL:\n  " + "\n  ".join(failures) + "\n")
                return 1

    print("[live-smoke] all probes passed")
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_run()))
    except asyncio.TimeoutError:
        sys.stderr.write("[live-smoke] FAIL: timed out talking to the server\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
