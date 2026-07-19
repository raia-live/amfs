"""Unit tests for the memory-augmenting OpenRouter proxy helpers (offline).

These cover the pure logic (message parsing, injection, fail-open retrieval)
without FastAPI/httpx. The full inject/forward/write-back route is covered by
tests/integration/test_openrouter_proxy.py (skipped in unit CI).
"""

from __future__ import annotations

from amfs_http import openrouter_proxy as P


def test_last_user_text_string_content():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what db does billing use?"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "and analytics?"},
    ]
    assert P._last_user_text(messages) == "and analytics?"


def test_last_user_text_content_parts():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"},
                                             {"type": "text", "text": "world"}]}]
    assert P._last_user_text(messages) == "hello world"


def test_last_user_text_none_when_absent():
    assert P._last_user_text([{"role": "system", "content": "x"}]) == ""


def test_inject_prepends_system_message():
    messages = [{"role": "user", "content": "q"}]
    out = P._inject(messages, ["fact A", "fact B"])
    assert out[0]["role"] == "system"
    assert "fact A" in out[0]["content"] and "fact B" in out[0]["content"]
    assert out[1:] == messages  # original messages preserved after the injected block


def test_inject_noop_without_facts():
    messages = [{"role": "user", "content": "q"}]
    assert P._inject(messages, []) is messages


def test_retrieve_facts_fail_open_on_get_memory_error():
    def _boom():
        raise RuntimeError("no memory")
    assert P._retrieve_facts(_boom, "svc/db", "query") == []


def test_retrieve_facts_empty_query_returns_empty():
    called = False

    def _get():
        nonlocal called
        called = True
        return object()
    assert P._retrieve_facts(_get, "svc/db", "") == []
    assert called is False


def test_retrieve_facts_uses_search_and_formats_values():
    class _Row:
        def __init__(self, value):
            self.value = value

    class _Mem:
        def search(self, *, query, entity_path, limit):
            return [_Row("The billing service runs on PostgreSQL 15."),
                    _Row({"db": "clickhouse"})]

    facts = P._retrieve_facts(lambda: _Mem(), "svc/db", "which db?")
    assert facts[0] == "The billing service runs on PostgreSQL 15."
    assert "clickhouse" in facts[1]
