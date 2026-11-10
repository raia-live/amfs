"""Filesystem adapter handles entries whose keys contain slashes.

AMFS supports hierarchical keys containing slashes (e.g. ``active-negotiation/xyz``
or ``settings/timeout``). On the filesystem adapter, the key is mapped to nested
directories under the entity directory. PathLayout must properly discover, list,
and search slashed keys without confusing intermediate key segments with entity
paths or dropping them from listings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from amfs_core.models import MemoryEntry, Provenance, SearchQuery
from amfs_filesystem.adapter import FilesystemAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(entity_path: str, key: str, value: str = "test-val") -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        provenance=Provenance(agent_id="test-agent", session_id="test-sess", written_at=_now()),
        confidence=1.0,
    )


def test_filesystem_slashed_key_list(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    entry = _entry("sweep/abc", "active-negotiation/xyz", value="proposal-42")
    adapter.write(entry)

    # Direct read
    read_entry = adapter.read("sweep/abc", "active-negotiation/xyz")
    assert read_entry is not None
    assert read_entry.entity_path == "sweep/abc"
    assert read_entry.key == "active-negotiation/xyz"
    assert read_entry.value == "proposal-42"

    # Listing by entity path must find the slashed key
    listed = adapter.list("sweep/abc")
    assert len(listed) == 1
    assert listed[0].key == "active-negotiation/xyz"
    assert listed[0].entity_path == "sweep/abc"

    # Global listing must find the slashed key
    all_entries = adapter.list(None)
    assert len(all_entries) == 1
    assert all_entries[0].key == "active-negotiation/xyz"
    assert all_entries[0].entity_path == "sweep/abc"


def test_filesystem_slashed_key_search(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("sweep/abc", "active-negotiation/xyz", value="special-term"))

    # Search scoped to entity path
    res = adapter.search(SearchQuery(entity_path="sweep/abc"))
    assert len(res) == 1
    assert res[0].key == "active-negotiation/xyz"

    # Search matching key substring
    res_key = adapter.search(SearchQuery(query="active-negotiation"))
    assert len(res_key) == 1
    assert res_key[0].key == "active-negotiation/xyz"

    # Search matching value substring
    res_val = adapter.search(SearchQuery(query="special-term"))
    assert len(res_val) == 1
    assert res_val[0].key == "active-negotiation/xyz"


def test_filesystem_slashed_key_disambiguation_with_nested_entity(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    # Entity "myapp" with slashed key "settings/timeout"
    e1 = _entry("myapp", "settings/timeout", value="60s")
    # Entity "myapp/settings" with standard key "port"
    e2 = _entry("myapp/settings", "port", value="8080")
    adapter.write(e1)
    adapter.write(e2)

    # Discovered entity paths must be exactly the two declared entity paths
    entity_paths = adapter._layout.all_entity_paths()
    assert entity_paths == ["myapp", "myapp/settings"]

    # Keys for "myapp" must be only "settings/timeout"
    keys_myapp = adapter._layout.all_keys("myapp")
    assert keys_myapp == ["settings/timeout"]

    # Keys for "myapp/settings" must be only "port"
    keys_nested = adapter._layout.all_keys("myapp/settings")
    assert keys_nested == ["port"]

    # List for "myapp" returns only e1
    listed_myapp = adapter.list("myapp")
    assert len(listed_myapp) == 1
    assert listed_myapp[0].entity_path == "myapp"
    assert listed_myapp[0].key == "settings/timeout"
    assert listed_myapp[0].value == "60s"

    # List for "myapp/settings" returns only e2
    listed_nested = adapter.list("myapp/settings")
    assert len(listed_nested) == 1
    assert listed_nested[0].entity_path == "myapp/settings"
    assert listed_nested[0].key == "port"
    assert listed_nested[0].value == "8080"


def test_filesystem_slashed_key_agent_entries(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("repo/core", "subsystem/auth/policy", value="rbac"))

    entries = adapter.list_entries_for_agent("test-agent")
    assert len(entries) == 1
    assert entries[0].entity_path == "repo/core"
    assert entries[0].key == "subsystem/auth/policy"


def test_filesystem_multi_segment_slashed_key(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("a/b", "deep/nested/path/to/key", value="val"))

    assert adapter.read("a/b", "deep/nested/path/to/key") is not None
    listed = adapter.list("a/b")
    assert len(listed) == 1
    assert listed[0].key == "deep/nested/path/to/key"
