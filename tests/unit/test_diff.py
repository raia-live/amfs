"""Tests for structural JSON diffing and patch operations."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.diff import apply_patch, create_patch, diff_entries, diff_values
from amfs_core.models import MemoryEntry, MemoryPatch, FieldChange, Provenance


def _make_entry(
    key: str = "k1",
    version: int = 1,
    value=None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path="test/svc",
        key=key,
        version=version,
        value=value if value is not None else {"data": 1},
        provenance=Provenance(
            agent_id="test", session_id="s",
            written_at=datetime.now(timezone.utc),
        ),
    )


class TestDiffValues:
    def test_identical_values(self):
        assert diff_values({"a": 1}, {"a": 1}) == []

    def test_added_key(self):
        changes = diff_values({"a": 1}, {"a": 1, "b": 2})
        assert len(changes) == 1
        assert changes[0].operation == "add"
        assert changes[0].path == "/b"
        assert changes[0].new_value == 2

    def test_removed_key(self):
        changes = diff_values({"a": 1, "b": 2}, {"a": 1})
        assert len(changes) == 1
        assert changes[0].operation == "remove"
        assert changes[0].path == "/b"
        assert changes[0].old_value == 2

    def test_replaced_value(self):
        changes = diff_values({"a": 1}, {"a": 2})
        assert len(changes) == 1
        assert changes[0].operation == "replace"
        assert changes[0].path == "/a"

    def test_nested_change(self):
        old = {"config": {"timeout": 30, "retries": 3}}
        new = {"config": {"timeout": 60, "retries": 3}}
        changes = diff_values(old, new)
        assert len(changes) == 1
        assert changes[0].path == "/config/timeout"
        assert changes[0].operation == "replace"

    def test_type_change(self):
        changes = diff_values({"a": 1}, "string")
        assert len(changes) == 1
        assert changes[0].operation == "replace"
        assert changes[0].path == "/"

    def test_list_changes(self):
        changes = diff_values([1, 2, 3], [1, 2, 4])
        assert len(changes) == 1
        assert changes[0].path == "/2"
        assert changes[0].operation == "replace"

    def test_list_length_change(self):
        changes = diff_values([1, 2], [1, 2, 3])
        assert len(changes) == 1
        assert changes[0].operation == "add"
        assert changes[0].path == "/2"

    def test_scalar_change(self):
        changes = diff_values("old", "new")
        assert len(changes) == 1
        assert changes[0].operation == "replace"

    def test_scalar_no_change(self):
        assert diff_values("same", "same") == []


class TestDiffEntries:
    def test_added_entry(self):
        new = _make_entry(value={"x": 1})
        diff = diff_entries(None, new)
        assert diff.diff_type == "added"
        assert diff.branch_value == {"x": 1}

    def test_deleted_entry(self):
        old = _make_entry(value={"x": 1})
        diff = diff_entries(old, None)
        assert diff.diff_type == "deleted"
        assert diff.parent_value == {"x": 1}

    def test_modified_entry(self):
        old = _make_entry(value={"x": 1}, version=1)
        new = _make_entry(value={"x": 2}, version=2)
        diff = diff_entries(old, new)
        assert diff.diff_type == "modified"
        assert len(diff.field_changes) == 1
        assert diff.field_changes[0].path == "/x"

    def test_modified_entry_no_changes(self):
        old = _make_entry(value={"x": 1}, version=1)
        new = _make_entry(value={"x": 1}, version=2)
        diff = diff_entries(old, new)
        assert diff.diff_type == "modified"
        assert len(diff.field_changes) == 0


class TestCreateAndApplyPatch:
    def test_round_trip(self):
        old = _make_entry(value={"a": 1, "b": 2}, version=1)
        new = _make_entry(value={"a": 1, "b": 3, "c": 4}, version=2)
        patch = create_patch(old, new)

        assert patch.entity_path == "test/svc"
        assert patch.source_version == 1
        assert patch.target_version == 2
        assert len(patch.changes) == 2

        result = apply_patch({"a": 1, "b": 2}, patch)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_apply_add(self):
        patch = MemoryPatch(
            entity_path="test/svc",
            key="k1",
            changes=[FieldChange(path="/new_key", operation="add", new_value="new")],
        )
        result = apply_patch({"existing": 1}, patch)
        assert result == {"existing": 1, "new_key": "new"}

    def test_apply_remove(self):
        patch = MemoryPatch(
            entity_path="test/svc",
            key="k1",
            changes=[FieldChange(path="/remove_me", operation="remove", old_value="old")],
        )
        result = apply_patch({"remove_me": "old", "keep": 1}, patch)
        assert result == {"keep": 1}

    def test_apply_replace_root(self):
        patch = MemoryPatch(
            entity_path="test/svc",
            key="k1",
            changes=[FieldChange(path="/", operation="replace", old_value="old", new_value="new")],
        )
        result = apply_patch("old", patch)
        assert result == "new"

    def test_nested_patch(self):
        old = _make_entry(value={"config": {"a": 1, "b": 2}}, version=1)
        new = _make_entry(value={"config": {"a": 1, "b": 3}}, version=2)
        patch = create_patch(old, new)

        result = apply_patch({"config": {"a": 1, "b": 2}}, patch)
        assert result == {"config": {"a": 1, "b": 3}}
