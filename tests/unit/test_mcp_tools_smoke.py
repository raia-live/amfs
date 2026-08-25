"""Publish-time smoke test for the whole MCP tool surface.

Regression guard for the class of bug where a *published* release of the MCP
server crashes the instant a tool is invoked — even though every other unit
test (which exercises tools in isolation against local source) is green.

The concrete failure this targets: ``amfs_retrieve`` began forwarding an
``include_artifacts`` kwarg down the stack to a ``retrieve`` that — in the
dependency combination ``uvx`` actually resolved from PyPI — did not accept it,
raising ``TypeError: ... unexpected keyword argument 'include_artifacts'`` and
hard-breaking recall on real clients (Claude Desktop). Nothing caught it because
no test *invoked the tool end-to-end against the resolved dependency set*.

This test does exactly that: it seeds a filesystem-backed store and calls every
core read/query/write tool the way a client would, asserting each returns
well-formed JSON without raising. Run in the release workflow against the
*installed wheel* (deps resolved from PyPI, as ``uvx`` resolves them), it turns
that class of version skew into a red build instead of a broken release.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture
def srv(tmp_path: Path):
    """A server module wired to a filesystem-backed, pre-seeded store.

    Importing the tools by attribute doubles as a registration guard: if a core
    tool is renamed or dropped, the accessor in a test below fails loudly.
    """
    import amfs_mcp.server as server

    server._adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="smoke")
    server._active_identity = "smoke-agent"
    server._memories = {}
    server._last_activity = 0.0

    # Seed enough that read / search / retrieve have something to return.
    server.amfs_write("smoke/db", "retry-policy", '{"max_retries": 3, "backoff": "exponential"}')
    server.amfs_write("smoke/ops", "runbook", "the deploy runbook lives in the ops wiki")

    yield server

    for m in server._memories.values():
        m.close()
    server._memories = {}
    server._adapter = None
    server._active_identity = None


def _ok(name: str, raw: object) -> dict | list:
    """Assert a tool returned well-formed JSON and did not report an error.

    The regression surfaced as an unhandled ``TypeError`` (so merely *calling*
    the tool without an exception is the primary guard), but a tool that caught
    its own failure and serialized ``{"error": ...}`` is equally broken from a
    client's point of view, so we reject that too.
    """
    assert isinstance(raw, str), f"{name} did not return a str, got {type(raw).__name__}"
    payload = json.loads(raw)  # invalid JSON → JSONDecodeError → test fails
    if isinstance(payload, dict):
        assert "error" not in payload, f"{name} returned an error payload: {payload.get('error')!r}"
    return payload


class TestReadSurfaceSmoke:
    """Every core read/query tool answers with well-formed JSON, no exception.

    A failure here means a client (Cursor / Claude Desktop / CLI) would hit a
    broken tool the moment it calls it.
    """

    def test_whoami(self, srv):
        _ok("amfs_whoami", srv.amfs_whoami())

    def test_read(self, srv):
        _ok("amfs_read", srv.amfs_read("smoke/db", "retry-policy"))

    def test_recall(self, srv):
        _ok("amfs_recall", srv.amfs_recall("smoke/db", "retry-policy"))

    def test_list(self, srv):
        _ok("amfs_list", srv.amfs_list())

    def test_my_entries(self, srv):
        _ok("amfs_my_entries", srv.amfs_my_entries())

    def test_search_query(self, srv):
        _ok("amfs_search", srv.amfs_search(query="runbook"))

    def test_search_browse(self, srv):
        _ok("amfs_search", srv.amfs_search(entity_path="smoke/db"))

    def test_stats(self, srv):
        _ok("amfs_stats", srv.amfs_stats())

    def test_export(self, srv):
        # Bulk export writes full values to a file and returns a manifest.
        payload = _ok("amfs_export", srv.amfs_export("smoke/db"))
        assert payload["entry_count"] >= 1
        assert "path" in payload

    def test_export_inline_gate_keys_on_inline_payload_size(self, srv):
        # The inline copy embeds the raw records, which re-serialize as JSON and
        # can dwarf a compact CSV file. The gate must compare the budget to that
        # inline size (reported as inline_chars), not the file's — otherwise a
        # small CSV would smuggle a huge inline blob past the context budget.
        srv.amfs_write(
            "smoke/rows", "batch",
            json.dumps({"rows": [{"v": "x" * 200} for _ in range(20)]}),
        )
        below = _ok(
            "amfs_export",
            srv.amfs_export("smoke/rows", format="csv", row_path="rows",
                            inline_char_budget=10),
        )
        assert below["inline_chars"] > 10
        assert "inline" not in below

        at_or_above = _ok(
            "amfs_export",
            srv.amfs_export("smoke/rows", format="csv", row_path="rows",
                            inline_char_budget=below["inline_chars"]),
        )
        assert "inline" in at_or_above

    def test_export_flattens_bare_array_with_dot_row_path(self, srv):
        srv.amfs_write(
            "smoke/bare", "chunk",
            json.dumps([{"addr": "1 Main"}, {"addr": "2 Main"}]),
        )
        payload = _ok(
            "amfs_export",
            srv.amfs_export("smoke/bare", format="csv", row_path="."),
        )
        assert payload["entry_count"] == 1
        assert payload["record_count"] == 2
        csv_text = Path(payload["path"]).read_text()
        assert "1 Main" in csv_text
        assert "2 Main" in csv_text

    def test_export_hints_when_values_are_bare_arrays(self, srv):
        srv.amfs_write("smoke/bare2", "chunk", json.dumps([{"addr": "1 Main"}]))
        payload = _ok("amfs_export", srv.amfs_export("smoke/bare2", format="csv"))
        assert payload["record_count"] == 1
        assert "row_path='.'" in payload["hint"]

    def test_aggregate_local_fallback(self, srv):
        # No AMFS_HTTP_URL in this test env, so amfs_aggregate must compute
        # locally over the same aggregates module rather than erroring.
        payload = _ok("amfs_aggregate", srv.amfs_aggregate("smoke/db", op="count"))
        assert "count" in payload

    def test_retrieve(self, srv):
        # THE regression: retrieve forwards include_artifacts down the stack.
        _ok("amfs_retrieve", srv.amfs_retrieve(query="retry policy"))

    def test_retrieve_include_artifacts_toggle(self, srv):
        # Both toggles must reach the downstream retrieve without a kwarg crash —
        # this is the exact code path that broke on a stale published dependency.
        _ok("amfs_retrieve", srv.amfs_retrieve(query="runbook", include_artifacts=False))
        _ok("amfs_retrieve", srv.amfs_retrieve(query="runbook", include_artifacts=True))


class TestWriteSurfaceSmoke:
    """The core write / trace tools accept a call and return well-formed JSON."""

    def test_write(self, srv):
        _ok("amfs_write", srv.amfs_write("smoke/x", "k", "v"))

    def test_record_context(self, srv):
        _ok("amfs_record_context", srv.amfs_record_context("smoke-label", "a summary", source="smoke"))

    def test_commit_outcome(self, srv):
        _ok("amfs_commit_outcome", srv.amfs_commit_outcome("smoke-outcome", "success"))


class TestListValuePreview:
    """Multi-result responses preview long values instead of inlining them.

    A stored source file can exceed 50K characters, and a 20-result search of
    such a store measured ~348K characters (~87K tokens) — more context spent
    than the recalled memories are worth, and enough to push anything appended
    after the entries out of what a client keeps.
    """

    def test_search_previews_long_values(self, srv):
        srv.amfs_write("smoke/big", "huge-file", "x" * 60_000)
        entry = next(
            e for e in _ok("amfs_search", srv.amfs_search(entity_path="smoke/big"))["entries"]
            if e["key"] == "huge-file"
        )

        assert entry["value_truncated"] is True
        assert entry["value_chars"] == 60_000
        assert len(entry["value"]) == srv.LIST_VALUE_CHAR_LIMIT
        assert "amfs_read" in entry["full_value"]

    def test_structured_values_are_previewed_too(self, srv):
        # The largest values in a real store are objects, not strings —
        # investigations and design plans reach 20K characters as JSON — so a
        # plain isinstance(str) check would leave the worst offenders whole.
        srv.amfs_write("smoke/big", "plan", {"steps": ["do a thing"] * 2_000})
        entry = next(
            e for e in _ok("amfs_search", srv.amfs_search(entity_path="smoke/big"))["entries"]
            if e["key"] == "plan"
        )

        assert entry["value_truncated"] is True
        assert entry["value_format"] == "json"
        assert len(entry["value"]) == srv.LIST_VALUE_CHAR_LIMIT

    def test_short_values_are_untouched(self, srv):
        entry = next(
            e for e in _ok("amfs_list", srv.amfs_list("smoke/ops"))["entries"]
            if e["key"] == "runbook"
        )

        assert entry["value"] == "the deploy runbook lives in the ops wiki"
        assert "value_truncated" not in entry

    def test_exact_reads_return_the_whole_value(self, srv):
        # Truncation is a list-response concern only: amfs_read is how an agent
        # recovers what a preview left out, so it must never be abbreviated.
        srv.amfs_write("smoke/big", "huge-file", "x" * 60_000)

        assert len(_ok("amfs_read", srv.amfs_read("smoke/big", "huge-file"))["value"]) == 60_000

    def test_retrieve_previews_and_keeps_scores(self, srv):
        srv.amfs_write("smoke/big", "huge-file", "runbook " * 8_000)
        payload = _ok(
            "amfs_retrieve", srv.amfs_retrieve(query="runbook", entity_path="smoke/big")
        )
        entry = next(e for e in payload["entries"] if e["key"] == "huge-file")

        assert entry["value_truncated"] is True
        assert "_score" in entry and "_breakdown" in entry
