"""Connector manifest (connector.yaml) schema, loader, and validator.

Every connector -- built-in or community -- is described by a standardized
``connector.yaml`` manifest that the CLI, registry, and Pro dashboard use
for discovery, installation, and configuration UI generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ConfigField(BaseModel):
    """Schema for a single connector configuration field."""

    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    enum: list[str] | None = None


class ConnectorManifest(BaseModel):
    """Parsed and validated connector.yaml manifest."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = "Apache-2.0"
    homepage: str = ""
    tags: list[str] = Field(default_factory=list)

    events: list[str] = Field(default_factory=list)

    outputs: dict[str, Any] = Field(default_factory=dict)

    entry_point: str = ""

    dependencies: list[str] = Field(default_factory=list)

    config_schema: dict[str, ConfigField] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid connector name: {v!r}")
        return v.lower().replace("_", "-")


def load_manifest(path: str | Path) -> ConnectorManifest:
    """Load and parse a connector.yaml file."""
    p = Path(path)
    if p.is_dir():
        p = p / "connector.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format in {p}")
    return ConnectorManifest(**data)


def validate_manifest(path: str | Path) -> list[str]:
    """Validate a connector.yaml and return a list of warnings (empty = valid)."""
    warnings: list[str] = []
    try:
        manifest = load_manifest(path)
    except Exception as e:
        return [f"Failed to load manifest: {e}"]

    if not manifest.entry_point:
        warnings.append("No entry_point specified -- connector won't be auto-discoverable")
    elif ":" not in manifest.entry_point:
        warnings.append(f"entry_point should be 'module:ClassName', got: {manifest.entry_point!r}")

    if not manifest.description:
        warnings.append("Missing description")

    if not manifest.events:
        warnings.append("No events listed -- consider documenting which events this connector handles")

    if not manifest.tags:
        warnings.append("No tags -- adding tags improves discoverability")

    return warnings
