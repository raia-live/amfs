"""The SSE manager is reachable from routers mounted outside this package.

Room routes live in a separate distribution and are mounted onto this app.
They cannot see a module-level private, so they look for the manager on
`request.app.state.sse_manager` — and nothing was putting it there. The room
event stream therefore answered 503 on every connection and every broadcast
aimed at it was dropped without a word, because a broadcast with no
subscribers is indistinguishable from one nobody happened to be watching.

A one-line omission that disables a whole feature and produces no error is
worth a test, even a test this small.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def test_the_manager_is_published_on_app_state() -> None:
    from amfs_http.server import app

    assert getattr(app.state, "sse_manager", None) is not None


def test_it_is_the_same_object_the_write_path_broadcasts_on() -> None:
    """Two managers would be worse than none: writes would broadcast into one
    while subscribers waited on the other, and nothing anywhere would fail."""
    from amfs_http import server

    assert server.app.state.sse_manager is server._sse_manager


def test_it_can_broadcast_a_room_event() -> None:
    """Shape check on the interface the room routes actually call."""
    from amfs_http.server import app

    app.state.sse_manager.broadcast_room_event(
        "00000000-0000-4000-8000-000000000000", "join", {"user_id": "u1"},
    )
