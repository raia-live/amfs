"""Structural JSON diffing and patch operations for AMFS entries."""

from __future__ import annotations

from typing import Any

from amfs_core.models import DiffEntry, FieldChange, MemoryEntry, MemoryPatch


def diff_values(old: Any, new: Any, *, prefix: str = "") -> list[FieldChange]:
    """Compute structural field-level diff between two JSON-like values.

    Uses RFC 6901 JSON Pointer paths. Handles dicts, lists, and scalars.
    """
    changes: list[FieldChange] = []

    if type(old) != type(new):
        changes.append(FieldChange(
            path=prefix or "/",
            operation="replace",
            old_value=old,
            new_value=new,
        ))
        return changes

    if isinstance(old, dict) and isinstance(new, dict):
        all_keys = set(old.keys()) | set(new.keys())
        for key in sorted(all_keys):
            child_path = f"{prefix}/{key}"
            if key not in old:
                changes.append(FieldChange(
                    path=child_path,
                    operation="add",
                    new_value=new[key],
                ))
            elif key not in new:
                changes.append(FieldChange(
                    path=child_path,
                    operation="remove",
                    old_value=old[key],
                ))
            elif old[key] != new[key]:
                changes.extend(diff_values(old[key], new[key], prefix=child_path))
        return changes

    if isinstance(old, list) and isinstance(new, list):
        max_len = max(len(old), len(new))
        for i in range(max_len):
            child_path = f"{prefix}/{i}"
            if i >= len(old):
                changes.append(FieldChange(
                    path=child_path,
                    operation="add",
                    new_value=new[i],
                ))
            elif i >= len(new):
                changes.append(FieldChange(
                    path=child_path,
                    operation="remove",
                    old_value=old[i],
                ))
            elif old[i] != new[i]:
                changes.extend(diff_values(old[i], new[i], prefix=child_path))
        return changes

    if old != new:
        changes.append(FieldChange(
            path=prefix or "/",
            operation="replace",
            old_value=old,
            new_value=new,
        ))

    return changes


def diff_entries(old_entry: MemoryEntry | None, new_entry: MemoryEntry | None) -> DiffEntry:
    """Compute a DiffEntry between two versions of the same key."""
    if old_entry is None and new_entry is not None:
        return DiffEntry(
            entity_path=new_entry.entity_path,
            key=new_entry.key,
            diff_type="added",
            branch_value=new_entry.value,
            branch_confidence=new_entry.confidence,
            branch_shared=new_entry.shared,
        )

    if old_entry is not None and new_entry is None:
        return DiffEntry(
            entity_path=old_entry.entity_path,
            key=old_entry.key,
            diff_type="deleted",
            parent_value=old_entry.value,
            parent_confidence=old_entry.confidence,
        )

    assert old_entry is not None and new_entry is not None
    field_changes = diff_values(old_entry.value, new_entry.value)

    return DiffEntry(
        entity_path=new_entry.entity_path,
        key=new_entry.key,
        diff_type="modified",
        branch_value=new_entry.value,
        parent_value=old_entry.value,
        branch_confidence=new_entry.confidence,
        parent_confidence=old_entry.confidence,
        branch_shared=new_entry.shared,
        field_changes=field_changes,
    )


def create_patch(old_entry: MemoryEntry, new_entry: MemoryEntry) -> MemoryPatch:
    """Create a serializable patch from two entry versions."""
    changes = diff_values(old_entry.value, new_entry.value)
    return MemoryPatch(
        entity_path=new_entry.entity_path,
        key=new_entry.key,
        changes=changes,
        source_version=old_entry.version,
        target_version=new_entry.version,
    )


def apply_patch(value: Any, patch: MemoryPatch) -> Any:
    """Apply a MemoryPatch to a value, returning the new value.

    Only handles top-level dict changes for now; nested paths use
    a recursive set/delete approach.
    """
    import copy
    result = copy.deepcopy(value)

    for change in patch.changes:
        parts = [p for p in change.path.split("/") if p]

        if change.operation == "add":
            _set_nested(result, parts, change.new_value)
        elif change.operation == "remove":
            _delete_nested(result, parts)
        elif change.operation == "replace":
            if not parts:
                result = change.new_value
            else:
                _set_nested(result, parts, change.new_value)

    return result


def _set_nested(obj: Any, path: list[str], value: Any) -> None:
    """Set a nested value by path parts."""
    for part in path[:-1]:
        if isinstance(obj, dict):
            obj = obj.setdefault(part, {})
        elif isinstance(obj, list):
            obj = obj[int(part)]
    last = path[-1]
    if isinstance(obj, dict):
        obj[last] = value
    elif isinstance(obj, list):
        idx = int(last)
        if idx >= len(obj):
            obj.append(value)
        else:
            obj[idx] = value


def _delete_nested(obj: Any, path: list[str]) -> None:
    """Delete a nested value by path parts."""
    for part in path[:-1]:
        if isinstance(obj, dict):
            obj = obj.get(part, {})
        elif isinstance(obj, list):
            obj = obj[int(part)]
    last = path[-1]
    if isinstance(obj, dict) and last in obj:
        del obj[last]
    elif isinstance(obj, list):
        idx = int(last)
        if idx < len(obj):
            obj.pop(idx)
