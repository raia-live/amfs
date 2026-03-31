"""PathLayout — maps entity_path/key/version to filesystem paths.

Directory structure::

    <root>/
    └── <namespace>/
        └── <entity_path>/
            └── <key>/
                ├── v001_current.json
                ├── v001_superseded.json   (after v002 is written)
                └── v002_current.json
"""

from __future__ import annotations

import re
from pathlib import Path


_VERSION_CURRENT_RE = re.compile(r"^v(\d+)_current\.json$")
_VERSION_ANY_RE = re.compile(r"^v(\d+)_(current|superseded)\.json$")


class PathLayout:
    """Translates AMFS logical paths to filesystem paths."""

    def __init__(self, root: Path, namespace: str) -> None:
        self.root = root
        self.namespace = namespace

    @property
    def namespace_dir(self) -> Path:
        return self.root / self.namespace

    def entity_dir(self, entity_path: str) -> Path:
        return self.namespace_dir / entity_path

    def key_dir(self, entity_path: str, key: str) -> Path:
        return self.entity_dir(entity_path) / key

    def version_file(self, entity_path: str, key: str, version: int, *, current: bool = True) -> Path:
        suffix = "current" if current else "superseded"
        return self.key_dir(entity_path, key) / f"v{version:03d}_{suffix}.json"

    def current_version_file(self, entity_path: str, key: str) -> Path | None:
        """Return the path to the current version file, or None if no versions exist."""
        key_d = self.key_dir(entity_path, key)
        if not key_d.exists():
            return None
        for f in sorted(key_d.iterdir(), reverse=True):
            if _VERSION_CURRENT_RE.match(f.name):
                return f
        return None

    def current_version_number(self, entity_path: str, key: str) -> int:
        """Return the current version number, or 0 if no versions exist."""
        f = self.current_version_file(entity_path, key)
        if f is None:
            return 0
        m = _VERSION_CURRENT_RE.match(f.name)
        return int(m.group(1)) if m else 0

    def all_version_files(
        self, entity_path: str, key: str, *, include_superseded: bool = False
    ) -> list[Path]:
        """Return all version files for a key, sorted by version ascending."""
        key_d = self.key_dir(entity_path, key)
        if not key_d.exists():
            return []
        result: list[Path] = []
        for f in sorted(key_d.iterdir()):
            m = _VERSION_ANY_RE.match(f.name)
            if m is None:
                continue
            if not include_superseded and m.group(2) == "superseded":
                continue
            result.append(f)
        return result

    def all_entity_paths(self) -> list[str]:
        """Return all entity paths in the namespace."""
        ns = self.namespace_dir
        if not ns.exists():
            return []
        paths: list[str] = []
        for entity_dir in sorted(ns.iterdir()):
            if entity_dir.is_dir():
                paths.append(entity_dir.name)
        return paths

    def all_keys(self, entity_path: str) -> list[str]:
        """Return all key names for an entity."""
        entity_d = self.entity_dir(entity_path)
        if not entity_d.exists():
            return []
        return sorted(d.name for d in entity_d.iterdir() if d.is_dir())

    def lock_path(self, entity_path: str, key: str) -> Path:
        return self.key_dir(entity_path, key) / ".lock"

    def tmp_file(self, entity_path: str, key: str, version: int) -> Path:
        return self.key_dir(entity_path, key) / f"v{version:03d}.tmp"
