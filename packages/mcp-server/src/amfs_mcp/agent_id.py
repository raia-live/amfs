"""Auto-detect agent identity from environment.

Determines the platform (cursor, claude-code, or generic) and username
so that every memory write carries automatic provenance without the
agent needing to configure anything.

IDE platforms (Cursor, Claude Code) take priority over the ``AMFS_AGENT_ID``
env var.  This prevents server-side vars (e.g. ``AMFS_AGENT_ID=amfs-server``
on Cloud Run) from leaking into local MCP sessions when inherited by the
IDE's shell.  ``AMFS_AGENT_ID`` is still honoured for headless / CI / server
contexts where no IDE signals are present.
"""

from __future__ import annotations

import getpass
import os


def detect_agent_id() -> str:
    """Build an agent_id from environment signals.

    Detection order (IDE wins over env var):
    1. Cursor — presence of ``CURSOR_SESSION_ID`` or ``VSCODE_PID``.
    2. Claude Code — presence of ``CLAUDE_CODE_SESSION``.
    3. Explicit ``AMFS_AGENT_ID`` env var (servers, CI, scripts).
    4. Fallback to ``agent/<username>``.
    """
    username = _get_username()

    if os.environ.get("CURSOR_SESSION_ID") or os.environ.get("VSCODE_PID"):
        return f"cursor/{username}"

    if os.environ.get("CLAUDE_CODE_SESSION"):
        return f"claude-code/{username}"

    explicit = os.environ.get("AMFS_AGENT_ID")
    if explicit:
        return explicit

    return f"agent/{username}"


def detect_platform() -> str:
    """Return the current platform name for metadata."""
    if os.environ.get("CURSOR_SESSION_ID") or os.environ.get("VSCODE_PID"):
        return "cursor"
    if os.environ.get("CLAUDE_CODE_SESSION"):
        return "claude-code"
    return "cli"


def _get_username() -> str:
    """Best-effort username from env or system."""
    return os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()
