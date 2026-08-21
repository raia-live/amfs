"""Unit tests for bulk aggregation + schema profiling over entry values.

These pure functions back three surfaces that must always agree: the
/api/v1/aggregate endpoint, the amfs_aggregate MCP local fallback, and the
room-index schema profile carried on entity digests. The cases here pin the
behaviours that are easy to regress: value coercion (parsed object vs JSON
string), row_path flattening, numeric ops ignoring non-numbers, group_by, and
schema inference.
"""

from __future__ import annotations

import types

import pytest

from amfs_core.aggregates import (
    AGGREGATE_OPS,
    aggregate_entries,
    coerce_value,
    iter_rows,
    room_schema_profile,
)


def _entry(value, key="k"):
    return types.SimpleNamespace(value=value, key=key)


# ── coercion ───────────────────────────────────────────────────────────


def test_coerce_passthrough_for_objects():
    assert coerce_value({"a": 1}) == {"a": 1}
    assert coerce_value([1, 2]) == [1, 2]


def test_coerce_parses_json_strings():
    assert coerce_value('{"a": 1}') == {"a": 1}
    assert coerce_value("[1, 2]") == [1, 2]


def test_coerce_leaves_plain_strings_alone():
    assert coerce_value("just text") == "just text"


# ── row flattening ───────────────────────────────────────────────────────


def test_iter_rows_without_row_path_one_row_per_entry():
    rows = iter_rows([_entry({"a": 1}), _entry({"a": 2})])
    assert rows == [{"a": 1}, {"a": 2}]


def test_iter_rows_flattens_list_valued_field():
    entries = [
        _entry({"listings": [{"p": 1}, {"p": 2}]}),
        _entry('{"listings": [{"p": 3}]}'),  # JSON string form
    ]
    rows = iter_rows(entries, row_path="listings")
    assert rows == [{"p": 1}, {"p": 2}, {"p": 3}]


# ── aggregation ──────────────────────────────────────────────────────────


def test_count_over_rows():
    entries = [_entry({"listings": [{"p": 1}, {"p": 2}]}), _entry({"listings": [{"p": 3}]})]
    res = aggregate_entries(entries, op="count", row_path="listings")
    assert res["count"] == 3
    assert res["total_rows"] == 3


def test_numeric_ops_ignore_non_numbers():
    entries = [_entry({"p": 10}), _entry({"p": "not-a-number"}), _entry({"p": "30"})]
    res = aggregate_entries(entries, op="stats", field="p")
    # "30" coerces to a number; the non-numeric string is dropped, not zeroed.
    assert res["n"] == 2
    assert res["sum"] == 40
    assert res["mean"] == 20
    assert res["min"] == 10
    assert res["max"] == 30


def test_string_and_object_values_aggregate_together():
    entries = [_entry({"p": 100}), _entry('{"p": 300}')]
    res = aggregate_entries(entries, op="mean", field="p")
    assert res["mean"] == 200


def test_group_by_partitions_rows():
    entries = [_entry({"listings": [
        {"p": 100, "city": "A"},
        {"p": 300, "city": "B"},
        {"p": 200, "city": "A"},
    ]})]
    res = aggregate_entries(entries, op="stats", field="p", group_by="city", row_path="listings")
    groups = {g["key"]: g for g in res["groups"]}
    assert groups["A"]["mean"] == 150
    assert groups["A"]["n"] == 2
    assert groups["B"]["sum"] == 300


def test_dotted_field_path():
    entries = [_entry({"meta": {"yield": 5.0}}), _entry({"meta": {"yield": 7.0}})]
    res = aggregate_entries(entries, op="mean", field="meta.yield")
    assert res["mean"] == 6.0


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        aggregate_entries([_entry({"p": 1})], op="median", field="p")


def test_numeric_op_without_field_raises():
    with pytest.raises(ValueError):
        aggregate_entries([_entry({"p": 1})], op="sum")


def test_count_needs_no_field():
    assert "count" in AGGREGATE_OPS
    assert aggregate_entries([_entry({"p": 1})], op="count")["count"] == 1


# ── schema profile ───────────────────────────────────────────────────────


def test_schema_profile_infers_fields_types_ranges():
    entries = [
        _entry({"listings": [
            {"price": 100, "city": "A", "active": True},
            {"price": 300, "city": "B", "active": False},
        ]}, key="batch-1"),
        _entry({"listings": [{"price": 200, "city": "A", "active": True}]}, key="batch-2"),
    ]
    prof = room_schema_profile(entries, row_path="listings")
    assert prof["total_rows"] == 3
    by_name = {f["field"]: f for f in prof["fields"]}
    assert by_name["price"]["numeric_range"] == {"min": 100, "max": 300}
    assert set(by_name["city"]["values"]) == {"A", "B"}
    assert by_name["city"]["cardinality"] == 2
    assert "bool" in by_name["active"]["types"]


def test_schema_profile_detects_row_path_candidates():
    entries = [_entry({"listings": [{"p": 1}, {"p": 2}]}, key="b1")]
    prof = room_schema_profile(entries)
    cands = {c["field"]: c for c in prof["row_path_candidates"]}
    assert "listings" in cands
    assert cands["listings"]["total_rows"] == 2


def test_schema_profile_counts_records_per_key():
    entries = [_entry({"a": 1}, key="b1"), _entry({"a": 2}, key="b2"), _entry({"a": 3}, key="b2")]
    prof = room_schema_profile(entries)
    assert prof["record_counts_by_key"] == {"b1": 1, "b2": 2}


def test_high_cardinality_field_hides_enum():
    entries = [_entry({"listings": [{"id": i} for i in range(100)]}, key="b1")]
    prof = room_schema_profile(entries, row_path="listings", max_enum=25)
    by_name = {f["field"]: f for f in prof["fields"]}
    assert by_name["id"]["cardinality"] == ">25"
    assert "values" not in by_name["id"]
