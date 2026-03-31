"""Adapter factory — instantiates adapters from AMFSConfig."""

from __future__ import annotations

from pathlib import Path

from amfs_core.abc import AdapterABC
from amfs_core.models import AMFSConfig, LayerConfig


_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str, cls: type) -> None:
    """Register an adapter class by name."""
    _ADAPTER_REGISTRY[name] = cls


def _ensure_builtins() -> None:
    """Lazily register built-in adapters."""
    if "filesystem" not in _ADAPTER_REGISTRY:
        from amfs_filesystem.adapter import FilesystemAdapter

        register_adapter("filesystem", FilesystemAdapter)

    if "postgres" not in _ADAPTER_REGISTRY:
        try:
            from amfs_postgres.adapter import PostgresAdapter

            register_adapter("postgres", PostgresAdapter)
        except ImportError:
            pass  # psycopg not installed


def create_adapter(layer: LayerConfig, namespace: str) -> AdapterABC:
    """Create an adapter instance from a LayerConfig."""
    _ensure_builtins()
    cls = _ADAPTER_REGISTRY.get(layer.adapter)
    if cls is None:
        raise ValueError(
            f"Unknown adapter '{layer.adapter}'. "
            f"Available: {sorted(_ADAPTER_REGISTRY.keys())}"
        )
    options = dict(layer.options)
    # Normalise filesystem root to Path
    if layer.adapter == "filesystem" and "root" in options:
        options["root"] = Path(options["root"])
    return cls(namespace=namespace, **options)


def create_adapter_from_config(
    config: AMFSConfig, *, layer_name: str = "primary"
) -> AdapterABC:
    """Create an adapter from a full config, selecting the named layer."""
    layer = config.layers.get(layer_name)
    if layer is None:
        raise KeyError(
            f"Layer '{layer_name}' not found in config. "
            f"Available: {sorted(config.layers.keys())}"
        )
    return create_adapter(layer, config.namespace)
