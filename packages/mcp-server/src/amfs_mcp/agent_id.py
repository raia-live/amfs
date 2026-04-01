"""Auto-detect agent identity from environment.

Determines the platform (cursor, claude-code, or generic) and username
so that every memory write carries automatic provenance without the
agent needing to configure anything.
"""

from __future__ import annotations

import getpass
import os


def detect_agent_id() -> str:
    """Build an agent_id from environment signals.

    Detection order:
    1. Explicit ``AMFS_AGENT_ID`` env var (escape hatch).
    2. Cursor — presence of ``CURSOR_SESSION_ID`` or ``VSCODE_PID``.
    3. Claude Code — presence of ``CLAUDE_CODE_SESSION``.
    4. Fallback to ``agent/<username>``.
    """
    explicit = os.environ.get("AMFS_AGENT_ID")
    if explicit:
        return explicit

    username = _get_username()

    if os.environ.get("CURSOR_SESSION_ID") or os.environ.get("VSCODE_PID"):
        return f"cursor/{username}"

    if os.environ.get("CLAUDE_CODE_SESSION"):
        return f"claude-code/{username}"

    return f"agent/{username}"


def _get_username() -> str:
    """Best-effort username from env or system."""
    return os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()
