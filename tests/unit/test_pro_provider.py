"""Unit tests for the optional Pro retrieval provider (graceful absence + extraction).

The live Pro retrieval path needs the amfs-internal packages and a populated
adapter (covered by the amfs-internal retrieval eval). Here we only assert the
OSS-safe behaviour: no adapter / disabled / None retriever all degrade cleanly,
and value extraction handles both object and dict shapes.
"""

from __future__ import annotations

from amfs_http import pro_provider


def test_build_returns_none_without_adapter() -> None:
    assert pro_provider.build_pro_retriever(None) is None


def test_build_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(pro_provider, "PRO_RETRIEVAL", False)
    assert pro_provider.build_pro_retriever(object()) is None


def test_retrieve_values_none_retriever() -> None:
    assert pro_provider.pro_retrieve_values(None, "scope", "q", 5) is None


def test_result_value_extraction_dict_and_object() -> None:
    class _Result:
        def __init__(self, value):
            self.entry = {"value": value}

    assert pro_provider._result_value(_Result("postgres")) == "postgres"
    assert pro_provider._result_value({"entry": {"value": 42}}) == 42
    assert pro_provider._result_value({"value": "x"}) == "x"


def test_retrieve_values_filters_none_and_empty() -> None:
    class _Retriever:
        def retrieve(self, query, *, entity_path=None, limit=10):
            return [
                type("R", (), {"entry": {"value": "a"}})(),
                type("R", (), {"entry": {"value": None}})(),
                type("R", (), {"entry": {"value": "b"}})(),
            ]

    vals = pro_provider.pro_retrieve_values(_Retriever(), "scope", "q", 5)
    assert vals == ["a", "b"]


def test_retrieve_values_empty_returns_none() -> None:
    class _Retriever:
        def retrieve(self, query, *, entity_path=None, limit=10):
            return []

    assert pro_provider.pro_retrieve_values(_Retriever(), "scope", "q", 5) is None
