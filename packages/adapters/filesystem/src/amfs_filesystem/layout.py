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

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

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

    def is_key_dir(self, directory: Path) -> bool:
        """Whether *directory* holds an entry's versions rather than more path.

        A key directory is the one that directly contains ``v001_current.json``
        and its siblings. That is the only structural marker separating the two
        kinds of directory, since both are just names on disk.
        """
        if not directory.is_dir():
            return False
        return any(_VERSION_ANY_RE.match(f.name) for f in directory.iterdir() if f.is_file())

    def _read_entry_entity_and_key(self, key_dir: Path) -> tuple[str, str] | None:
        """Read (entity_path, key) from the version file in key_dir, if present."""
        current_file: Path | None = None
        fallback_file: Path | None = None
        try:
            for f in key_dir.iterdir():
                if not f.is_file():
                    continue
                if _VERSION_CURRENT_RE.match(f.name):
                    current_file = f
                    break
                if fallback_file is None and _VERSION_ANY_RE.match(f.name):
                    fallback_file = f
            target = current_file or fallback_file
            if target is not None:
                data = json.loads(target.read_text(encoding="utf-8"))
                ep = data.get("entity_path")
                k = data.get("key")
                if ep is not None and k is not None:
                    return str(ep), str(k)
        except Exception:
            logger.debug("Failed to read entity/key from %s", key_dir, exc_info=True)
        return None

    def all_entity_paths(self) -> list[str]:
        """Return all entity paths in the namespace, at any depth.

        Entity paths are hierarchical and ``entity_dir`` maps them straight onto
        nested directories, so ``a/b`` is stored at ``<ns>/a/b``. Keys may also
        contain slashes (e.g. ``active-negotiation/xyz``), so a directory tree
        such as ``<ns>/sweep/abc/active-negotiation/xyz/`` holds entity
        ``sweep/abc`` with key ``active-negotiation/xyz``.

        To avoid misattributing intermediate key segments as entity paths (or
        mistaking nested entities for keys), this inspects the version file in
        each key directory to extract the true stored ``entity_path``.
        """
        ns = self.namespace_dir
        if not ns.exists():
            return []

        paths: set[str] = set()
        for root_dir, dirs, files in os.walk(ns):
            p = Path(root_dir)
            if self.is_key_dir(p):
                info = self._read_entry_entity_and_key(p)
                if info is not None:
                    paths.add(info[0])
                else:
                    rel = p.parent.relative_to(ns).as_posix()
                    if rel:
                        paths.add(rel)

        return sorted(paths)

    def all_keys(self, entity_path: str) -> list[str]:
        """Return all key names for an entity, including slashed keys."""
        entity_d = self.entity_dir(entity_path)
        if not entity_d.exists():
            return []

        keys: set[str] = set()
        for root_dir, dirs, files in os.walk(entity_d):
            p = Path(root_dir)
            if self.is_key_dir(p):
                info = self._read_entry_entity_and_key(p)
                if info is not None:
                    ep, k = info
                    if ep == entity_path:
                        keys.add(k)
                else:
                    rel = p.relative_to(entity_d).as_posix()
                    keys.add(rel)

        return sorted(keys)

    def lock_path(self, entity_path: str, key: str) -> Path:
        return self.key_dir(entity_path, key) / ".lock"

    def tmp_file(self, entity_path: str, key: str, version: int) -> Path:
        return self.key_dir(entity_path, key) / f"v{version:03d}.tmp"
