"""What a read tells the user it just did for them, computed once on the server.

The block existed before this and was built client-side, in two copied
``value_ledger.py`` files kept in step by a test asserting their class bodies were
source-identical. Everything without a copy showed nothing: the OSS stdio server,
the SDK, the raw API, and therefore every self-hoster and every direct caller. The
gap report had already proved the alternative — computed at the server, it reached
all three MCP surfaces with one implementation — and this is the same move applied
to the half that measures reuse rather than the absence of it.

Three properties are worth pinning, because each replaces something the old block
got wrong.

**It leads with what is true.** The estimate used to come first and the count of
memories second, so a caller relaying only the opening relayed the modelled
number. Facts first now, and the estimate labelled as one.

**It does not tell the model what to say.** The old first-of-session block opened
"SAY THIS FIRST, before you start the work" — an instruction competing with the
user's request, which is the same mechanism as the always-applied rule agents
skip. Nothing here commands.

**It describes the read it came from.** Because the block now arrives on a
response rather than accumulating in a session object, the adapter must overwrite
it on every read including one that credited nothing. Left stale, a lookup that
reused no memory would report the previous lookup's reuse.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from amfs_core.aggregates import (
    RECALL_TOKENS_CEIL,
    RECALL_TOKENS_FLOOR,
    recall_tokens_for_chars,
)
from amfs_core.models import MemoryEntry, Provenance
from amfs_core.reuse_value import reuse_value_block


def _entry(value: str = "x", *, recall_count: int = 0, agent_id: str = "author") -> MemoryEntry:
    return MemoryEntry(
        entity_path="app/auth",
        key="decision-session-store",
        value=value,
        confidence=0.9,
        recall_count=recall_count,
        provenance=Provenance(
            agent_id=agent_id, session_id="s", written_at=datetime.now(UTC)
        ),
    )


# ── the block itself ──────────────────────────────────────────────────


def test_a_lookup_that_credited_nothing_says_nothing():
    """Reporting a reuse of zero reads as a failure, which is not what happened.

    A miss and a filter-only search both credit nothing. The honest answer is
    silence, not a block whose every number is zero.
    """
    assert reuse_value_block(hits=0, content_chars=4000) is None
    assert reuse_value_block(hits=-1, content_chars=4000) is None


def test_the_facts_come_before_the_estimate():
    """A caller that relays only the opening relays something true.

    The old block put est_tokens_saved first, so the weakest claim in it — a
    number modelled from the size of the recalled text — was the one most likely
    to be repeated.
    """
    block = reuse_value_block(hits=1, content_chars=8000)
    keys = list(block)
    assert keys.index("memories_used") < keys.index("est_tokens_saved")
    assert keys.index("reused_before") < keys.index("est_tokens_saved")


def test_the_estimate_is_labelled_as_one():
    block = reuse_value_block(hits=1, content_chars=8000)
    assert block["estimate"] is True
    assert "clamped" in block["basis"]


def test_nothing_in_the_block_commands_the_model():
    """The imperative is gone, and must not come back.

    "SAY THIS FIRST, before you start the work" competed with the user's own
    request for the model's attention, which is precisely why the always-applied
    rule gets skipped. A fact the caller may relay does not compete.
    """
    for reused_before, author, reader in ((0, None, None), (3, None, None), (1, "a", "b")):
        block = reuse_value_block(
            hits=1, content_chars=8000, reused_before=reused_before,
            written_by=author, reused_by=reader,
        )
        text = json.dumps(block).lower()
        assert "say this first" not in text
        assert "before you start the work" not in text


def test_no_block_reports_a_time_or_a_cost():
    """Both were dropped for the same reason and neither returns here.

    Minutes derived from tokens restate the token count rather than corroborate
    it, and dollars priced off the answer's length came out roughly an order of
    magnitude low. A test blocked them in the old ledger; this is that test
    following the code to its new home.
    """
    block = reuse_value_block(hits=1, content_chars=8000, reused_before=2)
    text = json.dumps(block).lower()
    for banned in ("minute", "hour", "$", "usd", "cost"):
        assert banned not in text, f"{banned!r} is back in the block"


def test_the_token_credit_is_clamped_at_both_ends():
    """A one-line preference is not an investigation, and neither is a huge blob."""
    assert recall_tokens_for_chars(4) == RECALL_TOKENS_FLOOR
    assert recall_tokens_for_chars(10_000_000) == RECALL_TOKENS_CEIL
    middle = RECALL_TOKENS_FLOOR * 4 * 3
    assert recall_tokens_for_chars(middle) == middle // 4


def test_the_first_reuse_of_a_memory_is_named_as_that():
    block = reuse_value_block(hits=1, content_chars=8000, reused_before=0)
    assert "first reuse" in block["note"].lower()


def test_a_later_reuse_counts_from_the_stored_count():
    block = reuse_value_block(hits=1, content_chars=8000, reused_before=2)
    assert "third time" in block["note"].lower()


# ── the cross-surface claim ───────────────────────────────────────────


def test_one_agents_memory_reused_by_another_is_the_headline():
    """The one claim a local file or a single tool's memory cannot make.

    Both ids are already in hand where the credit is applied, so this costs
    nothing extra to produce.
    """
    block = reuse_value_block(
        hits=1, content_chars=8000, reused_before=1,
        written_by="cursor-agent", reused_by="claude-agent",
    )
    assert block["cross_surface"] == {
        "written_by": "cursor-agent", "reused_by": "claude-agent",
    }
    assert "cursor-agent" in block["note"]
    assert "claude-agent" in block["note"]


def test_an_agent_rereading_its_own_memory_is_not_a_cross_surface_moment():
    """Otherwise the headline claim fires on every ordinary lookup and means nothing."""
    block = reuse_value_block(
        hits=1, content_chars=8000, written_by="same-agent", reused_by="same-agent",
    )
    assert "cross_surface" not in block


def test_an_unknown_reader_is_not_guessed_at():
    """A client that sends no agent header must not produce a false claim."""
    block = reuse_value_block(hits=1, content_chars=8000, written_by="author", reused_by=None)
    assert "cross_surface" not in block
    block = reuse_value_block(hits=1, content_chars=8000, written_by=None, reused_by="reader")
    assert "cross_surface" not in block


# ── the server attaches it where reuse is credited ────────────────────


def test_the_server_puts_the_block_on_the_response_header():
    """A header because both read endpoints answer with a bare JSON array.

    There is no envelope to add a key to, and wrapping the array would break
    every existing client in order to carry a diagnostic.
    """
    from fastapi import Response
    from amfs_http import server as srv

    response = Response()
    request = _fake_request(agent="reader-agent")
    srv._attach_reuse_value(
        response, request, credited=_entry("y" * 8000, recall_count=2), hits=1
    )
    raw = response.headers.get(srv.REUSE_VALUE_HEADER)
    assert raw, "expected the reuse header to be set"
    block = json.loads(raw)
    assert block["memories_used"] == 1
    # The count the entry carried, not one including this reuse: the increment is
    # a separate best-effort write, so a total would be optimistic by one
    # whenever it failed.
    assert block["reused_before"] == 2
    assert block["cross_surface"]["written_by"] == "author"
    assert block["cross_surface"]["reused_by"] == "reader-agent"


def test_a_read_that_credited_nothing_sets_no_header():
    from fastapi import Response
    from amfs_http import server as srv

    response = Response()
    srv._attach_reuse_value(response, _fake_request(), credited=None, hits=0)
    assert srv.REUSE_VALUE_HEADER not in response.headers


def test_the_credit_measures_the_entry_not_the_payload():
    """Measuring the enclosing response pinned every reuse to the ceiling.

    A response carrying five entries plus scores and metadata is not what one
    credited memory saved.
    """
    from fastapi import Response
    from amfs_http import server as srv

    small = Response()
    srv._attach_reuse_value(
        small, _fake_request(), credited=_entry("tiny"), hits=1
    )
    block = json.loads(small.headers[srv.REUSE_VALUE_HEADER])
    assert block["est_tokens_saved"] == f"~{RECALL_TOKENS_FLOOR}"


def test_a_broken_block_never_breaks_the_read():
    """This is reporting. A defect in it must not change the answer.

    Same rule the recall bump above it already follows.
    """
    from fastapi import Response
    from amfs_http import server as srv

    class Exploding:
        @property
        def value(self):
            raise RuntimeError("boom")

    response = Response()
    srv._attach_reuse_value(response, _fake_request(), credited=Exploding(), hits=1)
    assert srv.REUSE_VALUE_HEADER not in response.headers


def _fake_request(*, agent: str | None = None):
    import types

    headers = {"x-amfs-agent-id": agent} if agent else {}
    return types.SimpleNamespace(headers=headers)


# ── the adapter carries it across, and does not let it go stale ───────


class _FakeHeaders(dict):
    pass


def _adapter_with_header(value: str | None):
    from amfs_adapter_http.adapter import HttpAdapter

    adapter = HttpAdapter.__new__(HttpAdapter)
    adapter._last_reuse_value = None
    adapter._last_headers = _FakeHeaders(
        {"X-SenseLab-Value": value} if value is not None else {}
    )
    return adapter


def test_the_adapter_parses_the_header():
    adapter = _adapter_with_header('{"memories_used":1,"reused_before":0}')
    adapter._capture_reuse_value()
    assert adapter._last_reuse_value == {"memories_used": 1, "reused_before": 0}


def test_a_later_read_that_credited_nothing_clears_the_block():
    """The bug this guards is silent and reports a reuse that did not happen.

    Two reads in a row, the first crediting and the second not: without the
    clear, the second read reports the first read's reuse.
    """
    adapter = _adapter_with_header('{"memories_used":1}')
    adapter._capture_reuse_value()
    assert adapter._last_reuse_value is not None

    adapter._last_headers = _FakeHeaders({})
    adapter._capture_reuse_value()
    assert adapter._last_reuse_value is None


def test_an_unparseable_header_is_dropped_not_raised():
    """A read must not fail because a diagnostic header was malformed."""
    adapter = _adapter_with_header("{not json")
    adapter._capture_reuse_value()
    assert adapter._last_reuse_value is None


def test_a_header_that_is_not_an_object_is_refused():
    adapter = _adapter_with_header("[1,2,3]")
    adapter._capture_reuse_value()
    assert adapter._last_reuse_value is None


# ── the stdio server forwards it, which is the coverage this adds ─────


def test_the_stdio_server_leads_a_read_with_the_block():
    """The surface that showed nothing at all before this.

    A self-hoster or anyone on the SDK saw no reuse line, not by decision but
    because the block was client-side in two copies and this server had no third.
    """
    import types
    from amfs_mcp import server as oss

    mem = types.SimpleNamespace(last_reuse_value={"memories_used": 1})
    out = oss._with_reuse_value(mem, {"count": 2, "entries": []})
    assert list(out)[0] == "senselab_value", "the block must lead, not trail"
    assert out["count"] == 2


def test_an_adapter_that_credits_nothing_adds_no_key():
    """Absent rather than zero: a zero reads as "your memory did nothing"."""
    import types
    from amfs_mcp import server as oss

    for value in (None, {}, "not-a-dict"):
        mem = types.SimpleNamespace(last_reuse_value=value)
        assert oss._with_reuse_value(mem, {"count": 0}) == {"count": 0}


def test_the_sdk_reads_through_to_the_adapter_on_every_path():
    """A property rather than a snapshot, so no read path can forget to capture.

    Server-side retrieve, the local scoring fallback and plain search all reach
    the same adapter attribute.
    """
    import types
    from amfs import memory as mem_mod

    m = mem_mod.AgentMemory.__new__(mem_mod.AgentMemory)
    m._adapter = types.SimpleNamespace(_last_reuse_value={"memories_used": 1})
    assert m.last_reuse_value == {"memories_used": 1}

    # A local adapter computes none, and must report none rather than a zero.
    m._adapter = types.SimpleNamespace()
    assert m.last_reuse_value is None
