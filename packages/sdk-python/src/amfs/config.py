"""YAML configuration loader for AMFS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from amfs_core.models import AMFSConfig, LayerConfig

_DEFAULT_CONFIG_NAMES = ("amfs.yaml", "amfs.yml", ".amfs.yaml", ".amfs.yml")


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for an AMFS config file."""
    directory = (start or Path.cwd()).resolve()
    while True:
        for name in _DEFAULT_CONFIG_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def load_config(path: Path) -> AMFSConfig:
    """Load and validate an AMFSConfig from a YAML file."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    namespace = raw.get("namespace", "default")
    layers: dict[str, LayerConfig] = {}
    for name, layer_raw in raw.get("layers", {}).items():
        layers[name] = LayerConfig(
            adapter=layer_raw["adapter"],
            options=layer_raw.get("options", {}),
        )
    return AMFSConfig(namespace=namespace, layers=layers)


def load_config_or_default(path: Path | None = None) -> AMFSConfig:
    """Load config from *path*, auto-discover, or return a sensible default."""
    if path is not None:
        return load_config(path)
    found = find_config()
    if found is not None:
        return load_config(found)
    # Default: filesystem adapter at .amfs/ in cwd
    return AMFSConfig(
        namespace="default",
        layers={
            "primary": LayerConfig(
                adapter="filesystem",
                options={"root": ".amfs"},
            )
        },
    )
