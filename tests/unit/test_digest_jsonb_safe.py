"""A compiled digest is stored even when an entry fed it text ``jsonb`` rejects.

Postgres ``jsonb`` has no representation for ``\\u0000`` and refuses lone UTF-16
surrogates; ``json.dumps`` emits both as escapes without complaint. A memory
value carrying either (LLM output, a pasted log) made ``upsert_digest`` fail
with ``UntranslatableCharacter`` after the compile was done, every time the
scope was recompiled. The summary is scrubbed before it is serialised.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("amfs_postgres", reason="amfs_postgres not installed")

from amfs_postgres.adapter import _jsonb_safe


def test_nul_is_dropped_and_lone_surrogates_replaced() -> None:
    assert _jsonb_safe("a\x00b") == "ab"
    assert _jsonb_safe("x\ud800y") == "x\ufffdy"
    # A valid pair survives; it is real text.
    assert _jsonb_safe("\U0001f600") == "\U0001f600"


def test_scrubs_nested_summaries_and_keys() -> None:
    summary = {
        "hot_context": [{"value": "see\x00here", "n": 3}],
        "by_agent": {"ag\x00ent": ["\ud83dok", 1.5, None, True]},
    }
    cleaned = _jsonb_safe(summary)
    assert cleaned == {
        "hot_context": [{"value": "seehere", "n": 3}],
        "by_agent": {"agent": ["\ufffdok", 1.5, None, True]},
    }
    # And what we hand Postgres contains neither escape.
    dumped = json.dumps(cleaned)
    assert "\\u0000" not in dumped and "\\ud83d" not in dumped


def test_clean_input_is_returned_unchanged() -> None:
    summary = {"a": ["b", {"c": "d"}], "n": 1}
    assert _jsonb_safe(summary) == summary
