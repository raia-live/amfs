"""Unit tests for the AMFS MCP server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from amfs.memory import AgentMemory
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_mcp.agent_id import detect_agent_id


# ---------------------------------------------------------------------------
# agent_id detection tests
# ---------------------------------------------------------------------------


class TestAgentIdDetection:
    def test_explicit_env_var(self) -> None:
        with patch.dict(os.environ, {"AMFS_AGENT_ID": "my-custom-agent"}, clear=False):
            assert detect_agent_id() == "my-custom-agent"

    def test_cursor_detection(self) -> None:
        env = {"CURSOR_SESSION_ID": "abc123", "USER": "alice"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AMFS_AGENT_ID", None)
            result = detect_agent_id()
            assert result == "cursor/alice"

    def test_vscode_pid_detection(self) -> None:
        env = {"VSCODE_PID": "12345", "USER": "bob"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AMFS_AGENT_ID", None)
            os.environ.pop("CURSOR_SESSION_ID", None)
            result = detect_agent_id()
            assert result == "cursor/bob"

    def test_claude_code_detection(self) -> None:
        env = {"CLAUDE_CODE_SESSION": "sess-xyz", "USER": "carol"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AMFS_AGENT_ID", None)
            os.environ.pop("CURSOR_SESSION_ID", None)
            os.environ.pop("VSCODE_PID", None)
            result = detect_agent_id()
            assert result == "claude-code/carol"

    def test_fallback_generic(self) -> None:
        with patch.dict(os.environ, {"USER": "dave"}, clear=False):
            os.environ.pop("AMFS_AGENT_ID", None)
            os.environ.pop("CURSOR_SESSION_ID", None)
            os.environ.pop("VSCODE_PID", None)
            os.environ.pop("CLAUDE_CODE_SESSION", None)
            result = detect_agent_id()
            assert result == "agent/dave"


# ---------------------------------------------------------------------------
# MCP tool function tests (direct invocation, no MCP transport)
# ---------------------------------------------------------------------------


class TestMCPTools:
    """Test MCP tool functions by calling them directly with a real adapter."""

    @pytest.fixture(autouse=True)
    def _setup_memory(self, tmp_amfs_root: Path) -> None:
        """Inject a test AgentMemory into the server module."""
        import amfs_mcp.server as srv

        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        mem = AgentMemory(agent_id="test-mcp-agent", adapter=adapter)
        srv._memory = mem
        yield
        mem.close()
        srv._memory = None

    def test_amfs_write_and_read(self) -> None:
        from amfs_mcp.server import amfs_read, amfs_write

        result = json.loads(amfs_write("svc", "retry", '{"max_retries": 3}'))
        assert result["entity_path"] == "svc"
        assert result["key"] == "retry"
        assert result["value"] == {"max_retries": 3}
        assert result["provenance"]["agent_id"] == "test-mcp-agent"
        assert result["version"] == 1

        read_result = json.loads(amfs_read("svc", "retry"))
        assert read_result["value"] == {"max_retries": 3}

    def test_amfs_read_not_found(self) -> None:
        from amfs_mcp.server import amfs_read

        result = json.loads(amfs_read("nonexistent", "nope"))
        assert result["status"] == "not_found"

    def test_amfs_write_plain_text(self) -> None:
        from amfs_mcp.server import amfs_write

        result = json.loads(amfs_write("svc", "note", "plain text value"))
        assert result["value"] == "plain text value"

    def test_amfs_write_with_confidence(self) -> None:
        from amfs_mcp.server import amfs_write

        result = json.loads(amfs_write("svc", "risky", "might fail", confidence=0.6))
        assert result["confidence"] == 0.6

    def test_amfs_write_with_pattern_refs(self) -> None:
        from amfs_mcp.server import amfs_write

        result = json.loads(
            amfs_write("svc", "pattern", "retry logic", pattern_refs=["timeout-handling"])
        )
        assert result["provenance"]["pattern_refs"] == ["timeout-handling"]

    def test_amfs_list_empty(self) -> None:
        from amfs_mcp.server import amfs_list

        result = json.loads(amfs_list())
        assert result == []

    def test_amfs_list_filtered(self) -> None:
        from amfs_mcp.server import amfs_list, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-a", "k2", "v2")
        amfs_write("svc-b", "k3", "v3")

        result = json.loads(amfs_list("svc-a"))
        assert len(result) == 2
        assert all(e["entity_path"] == "svc-a" for e in result)

    def test_amfs_list_all(self) -> None:
        from amfs_mcp.server import amfs_list, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_list())
        assert len(result) == 2

    def test_amfs_search_all(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_search())
        assert len(result) == 2

    def test_amfs_search_by_entity(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_search(entity_path="svc-a"))
        assert len(result) == 1
        assert result[0]["entity_path"] == "svc-a"

    def test_amfs_search_by_min_confidence(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "high", "v1", confidence=0.9)
        amfs_write("svc", "low", "v2", confidence=0.2)

        result = json.loads(amfs_search(min_confidence=0.5))
        assert len(result) == 1
        assert result[0]["key"] == "high"

    def test_amfs_search_with_text_query(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "retry-logic", "exponential backoff pattern")
        amfs_write("svc", "auth-flow", "JWT token validation")

        result = json.loads(amfs_search(query="retry"))
        assert len(result) == 1
        assert result[0]["key"] == "retry-logic"

    def test_amfs_search_text_query_in_value(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "k1", "contains the word backoff here")
        amfs_write("svc", "k2", "something else entirely")

        result = json.loads(amfs_search(query="backoff"))
        assert len(result) == 1
        assert result[0]["key"] == "k1"

    def test_amfs_stats_empty(self) -> None:
        from amfs_mcp.server import amfs_stats

        result = json.loads(amfs_stats())
        assert result["total_entries"] == 0
        assert result["total_entities"] == 0

    def test_amfs_stats_populated(self) -> None:
        from amfs_mcp.server import amfs_stats, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_stats())
        assert result["total_entries"] == 2
        assert result["total_entities"] == 2
        assert result["total_agents"] == 1

    def test_amfs_commit_outcome(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome, amfs_read, amfs_write

        amfs_write("svc", "key", "value")
        amfs_read("svc", "key")

        result = json.loads(amfs_commit_outcome("INC-001", "p1_incident"))
        assert result["outcome_ref"] == "INC-001"
        assert result["affected_entries"] == 1
        assert result["entries"][0]["confidence"] == 1.15

    def test_amfs_commit_outcome_invalid_type(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome

        result = json.loads(amfs_commit_outcome("REF-1", "invalid_type"))
        assert "error" in result
        assert "Invalid outcome_type" in result["error"]

    def test_amfs_commit_outcome_clean_deploy(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome, amfs_read, amfs_write

        amfs_write("svc", "key", "value")
        amfs_read("svc", "key")

        result = json.loads(amfs_commit_outcome("deploy-v1", "clean_deploy"))
        assert result["affected_entries"] == 1
        assert result["entries"][0]["confidence"] == pytest.approx(0.97)

    def test_embedding_stripped_from_output(self) -> None:
        from amfs_mcp.server import _serialize_entry
        from amfs_core.models import MemoryEntry, Provenance
        from datetime import datetime, timezone

        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            version=1,
            value="v",
            provenance=Provenance(
                agent_id="a",
                session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
            embedding=[0.1, 0.2, 0.3],
        )
        serialized = _serialize_entry(entry)
        assert "embedding" not in serialized


# ---------------------------------------------------------------------------
# Transport / CLI arg parsing tests
# ---------------------------------------------------------------------------


class TestTransportConfig:
    def test_parse_default_stdio(self) -> None:
        from amfs_mcp.server import _parse_args, _TRANSPORT_ALIASES
        import sys

        with patch.object(sys, "argv", ["amfs-mcp-server"]):
            args = _parse_args()
            assert args.transport is None

    def test_parse_http_transport(self) -> None:
        from amfs_mcp.server import _parse_args
        import sys

        with patch.object(sys, "argv", ["amfs-mcp-server", "--transport", "http"]):
            args = _parse_args()
            assert args.transport == "http"

    def test_parse_streamable_http_transport(self) -> None:
        from amfs_mcp.server import _parse_args
        import sys

        with patch.object(sys, "argv", ["amfs-mcp-server", "-t", "streamable-http"]):
            args = _parse_args()
            assert args.transport == "streamable-http"

    def test_parse_host_port_path(self) -> None:
        from amfs_mcp.server import _parse_args
        import sys

        argv = [
            "amfs-mcp-server",
            "--transport", "http",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--path", "/amfs",
        ]
        with patch.object(sys, "argv", argv):
            args = _parse_args()
            assert args.host == "127.0.0.1"
            assert args.port == 9000
            assert args.path == "/amfs"

    def test_transport_alias_mapping(self) -> None:
        from amfs_mcp.server import _TRANSPORT_ALIASES

        assert _TRANSPORT_ALIASES["stdio"] == "stdio"
        assert _TRANSPORT_ALIASES["http"] == "streamable-http"
        assert _TRANSPORT_ALIASES["streamable-http"] == "streamable-http"

    def test_main_stdio_calls_run(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        with (
            patch.object(sys, "argv", ["amfs-mcp-server"]),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(transport="stdio")

    def test_main_http_calls_run_with_defaults(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        with (
            patch.object(sys, "argv", ["amfs-mcp-server", "--transport", "http"]),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="0.0.0.0",
                port=8000,
                path="/mcp",
            )

    def test_main_http_custom_host_port_path(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        argv = [
            "amfs-mcp-server",
            "-t", "http",
            "--host", "127.0.0.1",
            "-p", "9000",
            "--path", "/amfs",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="127.0.0.1",
                port=9000,
                path="/amfs",
            )

    def test_main_env_transport_override(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        with (
            patch.object(sys, "argv", ["amfs-mcp-server"]),
            patch.dict(os.environ, {"AMFS_TRANSPORT": "http"}, clear=False),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="0.0.0.0",
                port=8000,
                path="/mcp",
            )

    def test_main_env_host_port_path(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        env = {
            "AMFS_TRANSPORT": "http",
            "AMFS_HOST": "10.0.0.1",
            "AMFS_PORT": "3000",
            "AMFS_PATH": "/memory",
        }
        with (
            patch.object(sys, "argv", ["amfs-mcp-server"]),
            patch.dict(os.environ, env, clear=False),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="10.0.0.1",
                port=3000,
                path="/memory",
            )

    def test_cli_flag_overrides_env(self) -> None:
        from amfs_mcp.server import main, mcp
        import sys

        with (
            patch.object(sys, "argv", ["amfs-mcp-server", "--transport", "stdio"]),
            patch.dict(os.environ, {"AMFS_TRANSPORT": "http"}, clear=False),
            patch.object(mcp, "run") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(transport="stdio")


# ---------------------------------------------------------------------------
# Config resolution tests
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_postgres_dsn_env(self) -> None:
        from amfs_mcp.server import _resolve_config

        with patch.dict(
            os.environ,
            {"AMFS_POSTGRES_DSN": "postgresql://user:pass@host:5432/amfs"},
            clear=False,
        ):
            config = _resolve_config()
            assert config.layers["primary"].adapter == "postgres"
            assert config.layers["primary"].options["dsn"] == "postgresql://user:pass@host:5432/amfs"

    def test_data_dir_env(self) -> None:
        from amfs_mcp.server import _resolve_config

        with patch.dict(os.environ, {"AMFS_DATA_DIR": "/tmp/custom-amfs"}, clear=False):
            os.environ.pop("AMFS_POSTGRES_DSN", None)
            config = _resolve_config()
            assert config.layers["primary"].adapter == "filesystem"
            assert config.layers["primary"].options["root"] == "/tmp/custom-amfs"

    def test_fallback_default(self) -> None:
        from amfs_mcp.server import _resolve_config

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMFS_POSTGRES_DSN", None)
            os.environ.pop("AMFS_DATA_DIR", None)
            config = _resolve_config()
            assert config.layers["primary"].adapter == "filesystem"
