"""Connector registry with automatic entry-point discovery.

Installed connectors register themselves via the ``amfs.connectors``
entry-point group in their ``pyproject.toml``::

    [project.entry-points."amfs.connectors"]
    pagerduty = "amfs_connector_pagerduty:PagerDutyConnector"

The registry auto-discovers all installed connectors at import time
and provides lookup by name.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from amfs_connectors.base import ConnectorABC, ConnectorConfig

logger = logging.getLogger(__name__)


def _load_entry_points() -> dict[str, str]:
    """Discover connectors registered via importlib.metadata entry points."""
    discovered: dict[str, str] = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        connector_eps = eps.get("amfs.connectors", []) if isinstance(eps, dict) else eps.select(group="amfs.connectors")
        for ep in connector_eps:
            discovered[ep.name] = ep.value
            logger.debug("Discovered connector entry point: %s = %s", ep.name, ep.value)
    except Exception:
        logger.debug("Entry point discovery failed", exc_info=True)
    return discovered


class ConnectorRegistry:
    """Registry of available connectors, populated from entry points and manual registration."""

    def __init__(self, *, auto_discover: bool = True) -> None:
        self._entry_points: dict[str, str] = {}
        self._instances: dict[str, ConnectorABC] = {}
        self._classes: dict[str, type[ConnectorABC]] = {}
        if auto_discover:
            self._entry_points = _load_entry_points()

    def register(self, name: str, connector_cls: type[ConnectorABC]) -> None:
        """Manually register a connector class."""
        self._classes[name] = connector_cls

    def get(self, name: str) -> ConnectorABC | None:
        """Get a connector instance by name, instantiating on first access."""
        if name in self._instances:
            return self._instances[name]

        cls = self._resolve_class(name)
        if cls is None:
            return None

        try:
            config = ConnectorConfig(
                name=name,
                connector_type=name,
                entity_path=name,
            )
            instance = cls(config)
            self._instances[name] = instance
            return instance
        except Exception:
            logger.error("Failed to instantiate connector %r", name, exc_info=True)
            return None

    def _resolve_class(self, name: str) -> type[ConnectorABC] | None:
        """Resolve a connector class from manual registrations or entry points."""
        if name in self._classes:
            return self._classes[name]

        ep_value = self._entry_points.get(name)
        if not ep_value:
            return None

        try:
            module_path, class_name = ep_value.rsplit(":", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._classes[name] = cls
            return cls
        except Exception:
            logger.error("Failed to load connector %r from %r", name, ep_value, exc_info=True)
            return None

    def list_available(self) -> list[str]:
        """List names of all available connectors (registered + entry points)."""
        return sorted(set(self._classes.keys()) | set(self._entry_points.keys()))

    def list_installed(self) -> list[dict[str, Any]]:
        """List installed connectors with metadata."""
        result = []
        for name in self.list_available():
            info: dict[str, Any] = {"name": name, "source": "manual"}
            if name in self._entry_points:
                info["source"] = "entry_point"
                info["entry_point"] = self._entry_points[name]
            result.append(info)
        return result
