---
title: Custom Adapters
layout: default
parent: Adapters
nav_order: 3
description: "Build your own storage adapter for AMFS."
---

# Custom Adapters

You can build a custom adapter for any storage backend by implementing the `AdapterABC` abstract base class.

---

## The AdapterABC Interface

```python
from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, OutcomeRecord, SearchQuery, MemoryStats

class MyAdapter(AdapterABC):
    def read(self, entity_path: str, key: str) -> MemoryEntry | None:
        """Return the current version of a key, or None."""
        ...

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a new version. Must handle CoW (supersede old, write new)."""
        ...

    def list(
        self, entity_path: str | None = None, *, include_superseded: bool = False
    ) -> list[MemoryEntry]:
        """List entries, optionally filtered by entity and superseded status."""
        ...

    def watch(self, entity_path: str, callback) -> object:
        """Register a callback for changes. Return a handle with a cancel() method."""
        ...

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        """Apply an outcome to causal entries. Return updated entries."""
        ...

    def search(self, query: SearchQuery) -> list[MemoryEntry]:
        """Search entries with filters. Default implementation filters list()."""
        ...

    def stats(self) -> MemoryStats:
        """Return memory statistics. Default implementation counts list()."""
        ...
```

---

## Contract Tests

AMFS provides a shared contract test suite. Your adapter must pass all tests:

```python
from tests.integration.adapter_contract import AdapterContractMixin

class TestMyAdapter(AdapterContractMixin):
    def create_adapter(self):
        return MyAdapter(...)
```

This ensures your adapter behaves identically to the filesystem and Postgres adapters.

---

## Registering Your Adapter

Register your adapter with the factory so it can be used in YAML config:

```python
from amfs.factory import register_adapter

register_adapter("my-backend", MyAdapter)
```

Then in `amfs.yaml`:

```yaml
layers:
  primary:
    adapter: my-backend
    options:
      connection_string: "..."
```
