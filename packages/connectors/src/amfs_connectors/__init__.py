"""AMFS connector framework: build, install, and manage external system connectors."""

from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult
from amfs_connectors.webhook import WebhookIngester, WebhookEvent, WebhookConfig
from amfs_connectors.manifest import ConnectorManifest, load_manifest, validate_manifest
from amfs_connectors.registry import ConnectorRegistry

__all__ = [
    "ConnectorABC",
    "ConnectorConfig",
    "ConnectorRegistry",
    "ConnectorManifest",
    "IngestionResult",
    "WebhookConfig",
    "WebhookEvent",
    "WebhookIngester",
    "load_manifest",
    "validate_manifest",
]
