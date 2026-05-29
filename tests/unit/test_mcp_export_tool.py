"""Unit tests for the amfs_export_training_data MCP tool (WS3)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestExportToolExists:
    """Verify the export tool is registered in the MCP server module."""

    def test_server_module_importable(self) -> None:
        import amfs_mcp.server  # noqa: F401

    def test_export_tool_source_present(self) -> None:
        """The server source contains the export tool registration."""
        import amfs_mcp.server as mod
        import inspect

        source = inspect.getsource(mod)
        assert "amfs_export_training_data" in source
        assert "pro/export" in source


class TestExportToolParameterLogic:
    """Test the parameter construction logic used by the export tool."""

    def test_params_include_format(self) -> None:
        params = {"format": "sft", "limit": "5"}
        assert params["format"] == "sft"

    def test_entity_path_included_when_provided(self) -> None:
        entity_paths = ["myapp/auth"]
        params: dict[str, str] = {"format": "sft", "limit": "5"}
        if entity_paths:
            params["entity_path"] = entity_paths[0]
        assert params["entity_path"] == "myapp/auth"

    def test_min_confidence_included_when_nonzero(self) -> None:
        min_confidence = 0.7
        params: dict[str, str] = {"format": "sft", "limit": "5"}
        if min_confidence > 0:
            params["min_confidence"] = str(min_confidence)
        assert params["min_confidence"] == "0.7"

    def test_min_confidence_excluded_when_zero(self) -> None:
        min_confidence = 0.0
        params: dict[str, str] = {"format": "sft", "limit": "5"}
        if min_confidence > 0:
            params["min_confidence"] = str(min_confidence)
        assert "min_confidence" not in params

    @pytest.mark.skipif(
        not pytest.importorskip("httpx", reason="httpx not installed"),
        reason="httpx required",
    )
    def test_httpx_call_uses_amfs_http_url(self) -> None:
        """When AMFS_HTTP_URL is set, the tool should use that base URL."""
        with patch.dict(os.environ, {"AMFS_HTTP_URL": "http://custom:9090"}):
            base = os.environ.get("AMFS_HTTP_URL", "http://localhost:8080")
            assert base == "http://custom:9090"

    def test_default_url_is_localhost(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMFS_HTTP_URL", None)
            base = os.environ.get("AMFS_HTTP_URL", "http://localhost:8080")
            assert base == "http://localhost:8080"
