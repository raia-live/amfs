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
        """Inject a test adapter and identity into the server module."""
        import amfs_mcp.server as srv

        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        srv._adapter = adapter
        srv._active_identity = "test-mcp-agent"
        srv._memories = {}
        srv._last_activity = 0.0
        yield
        for m in srv._memories.values():
            m.close()
        srv._memories = {}
        srv._adapter = None
        srv._active_identity = None

    def test_amfs_write_and_read(self) -> None:
        from amfs_mcp.server import amfs_read, amfs_write

        raw = json.loads(amfs_write("svc", "retry", '{"max_retries": 3}'))
        result = raw["entry"]
        assert result["entity_path"] == "svc"
        assert result["key"] == "retry"
        assert result["value"] == {"max_retries": 3}
        assert result["provenance"]["agent_id"] == "test-mcp-agent"
        assert result["version"] == 1
        assert "quality" in raw

        read_result = json.loads(amfs_read("svc", "retry"))
        assert read_result["value"] == {"max_retries": 3}

    def test_amfs_read_not_found(self) -> None:
        from amfs_mcp.server import amfs_read

        result = json.loads(amfs_read("nonexistent", "nope"))
        assert result["status"] == "not_found"

    def test_amfs_write_plain_text(self) -> None:
        from amfs_mcp.server import amfs_write

        raw = json.loads(amfs_write("svc", "note", "plain text value"))
        assert raw["entry"]["value"] == "plain text value"

    def test_amfs_write_with_confidence(self) -> None:
        from amfs_mcp.server import amfs_write

        raw = json.loads(amfs_write("svc", "risky", "might fail", confidence=0.6))
        assert raw["entry"]["confidence"] == 0.6

    def test_amfs_write_with_pattern_refs(self) -> None:
        from amfs_mcp.server import amfs_write

        raw = json.loads(
            amfs_write("svc", "pattern", "retry logic", pattern_refs=["timeout-handling"])
        )
        assert raw["entry"]["provenance"]["pattern_refs"] == ["timeout-handling"]

    def test_amfs_list_empty(self) -> None:
        from amfs_mcp.server import amfs_list

        result = json.loads(amfs_list())
        assert result["status"] == "empty"
        assert result["count"] == 0

    def test_amfs_list_filtered(self) -> None:
        from amfs_mcp.server import amfs_list, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-a", "k2", "v2")
        amfs_write("svc-b", "k3", "v3")

        result = json.loads(amfs_list("svc-a"))
        assert result["count"] == 2
        assert all(e["entity_path"] == "svc-a" for e in result["entries"])

    def test_amfs_list_all(self) -> None:
        from amfs_mcp.server import amfs_list, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_list())
        assert result["count"] == 2

    def test_amfs_search_all(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_search())
        assert result["count"] == 2

    def test_amfs_search_by_entity(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc-a", "k1", "v1")
        amfs_write("svc-b", "k2", "v2")

        result = json.loads(amfs_search(entity_path="svc-a"))
        assert result["count"] == 1
        assert result["entries"][0]["entity_path"] == "svc-a"

    def test_amfs_search_by_min_confidence(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "high", "v1", confidence=0.9)
        amfs_write("svc", "low", "v2", confidence=0.2)

        result = json.loads(amfs_search(min_confidence=0.5))
        assert result["count"] == 1
        assert result["entries"][0]["key"] == "high"

    def test_amfs_search_with_text_query(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "retry-logic", "exponential backoff pattern")
        amfs_write("svc", "auth-flow", "JWT token validation")

        result = json.loads(amfs_search(query="retry"))
        assert result["count"] == 1
        assert result["entries"][0]["key"] == "retry-logic"

    def test_amfs_search_text_query_in_value(self) -> None:
        from amfs_mcp.server import amfs_search, amfs_write

        amfs_write("svc", "k1", "contains the word backoff here")
        amfs_write("svc", "k2", "something else entirely")

        result = json.loads(amfs_search(query="backoff"))
        assert result["count"] == 1
        assert result["entries"][0]["key"] == "k1"

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

        result = json.loads(amfs_commit_outcome("INC-001", "critical_failure"))
        assert result["outcome_ref"] == "INC-001"
        assert result["affected_entries"] == 1
        # CRITICAL_FAILURE erodes confidence: 1.0 * 0.85 = 0.85
        assert result["entries"][0]["confidence"] == pytest.approx(0.85)

    def test_amfs_commit_outcome_invalid_type(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome

        result = json.loads(amfs_commit_outcome("REF-1", "invalid_type"))
        assert "error" in result
        assert "Invalid outcome_type" in result["error"]

    def test_amfs_commit_outcome_clean_deploy(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome, amfs_read, amfs_write

        amfs_write("svc", "key", "value")
        amfs_read("svc", "key")

        result = json.loads(amfs_commit_outcome("deploy-v1", "success"))
        assert result["affected_entries"] == 1
        # SUCCESS reinforces confidence: 1.0 * 1.03 = 1.03, clamped to 1.0
        assert result["entries"][0]["confidence"] == pytest.approx(1.0)

    def test_a_write_only_session_reports_zero_but_records_the_trace(self) -> None:
        """Back-propagation adjusts memories the outcome is evidence about,
        which is the set that was read. Writing alone leaves nothing to adjust,
        so zero here is the honest answer rather than a lost trace — and the
        response has to say so, because "0" on its own reads like a no-op."""
        from amfs_mcp.server import amfs_commit_outcome, amfs_write

        amfs_write("svc", "key", "value")

        result = json.loads(amfs_commit_outcome("task-42", "success"))
        assert result["affected_entries"] == 0
        assert result["entries_created"] == 1
        assert "read" in result["note"]

    def test_the_explanation_is_absent_once_something_was_read(self) -> None:
        from amfs_mcp.server import amfs_commit_outcome, amfs_read, amfs_write

        amfs_write("svc", "key", "value")
        amfs_read("svc", "key")

        result = json.loads(amfs_commit_outcome("task-43", "success"))
        assert result["affected_entries"] == 1
        assert "note" not in result

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

    # -- quality feedback tests --

    def test_amfs_write_quality_feedback_short_value(self) -> None:
        from amfs_mcp.server import amfs_write

        raw = json.loads(amfs_write("svc", "k", "short"))
        assert "quality" in raw
        assert raw["quality"]["score"] < 0.8
        types = [i["type"] for i in raw["quality"]["issues"]]
        assert "too_short" in types

    def test_amfs_write_quality_good_structured_value(self) -> None:
        from amfs_mcp.server import amfs_write

        raw = json.loads(amfs_write(
            "svc", "retry-config",
            '{"max_retries": 3, "backoff": "exponential", "timeout_ms": 5000}',
        ))
        assert "quality" in raw
        assert raw["quality"]["score"] >= 0.8
        assert raw["quality"]["action"] == "stored_ok"

    def test_amfs_write_quality_disabled(self) -> None:
        import amfs_mcp.server as srv

        srv._quality_evaluator = None
        with patch.dict(os.environ, {"AMFS_QUALITY_FEEDBACK": "0"}, clear=False):
            from amfs_mcp.server import amfs_write

            raw = json.loads(amfs_write("svc", "k", "short"))
            assert raw["quality"]["score"] == 1.0
            assert raw["quality"]["issues"] == []
        srv._quality_evaluator = None


# ---------------------------------------------------------------------------
# Identity conflict guard tests
# ---------------------------------------------------------------------------


class TestIdentityGuard:
    """Test that set_identity prevents cross-conversation clobbering."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_amfs_root: Path) -> None:
        import amfs_mcp.server as srv

        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        srv._adapter = adapter
        srv._active_identity = None
        srv._memories = {}
        srv._last_activity = 0.0
        srv._session_metadata = None
        yield
        for m in srv._memories.values():
            m.close()
        srv._memories = {}
        srv._adapter = None
        srv._active_identity = None
        srv._session_metadata = None

    def test_set_identity_creates_memory(self) -> None:
        import amfs_mcp.server as srv
        from amfs_mcp.server import amfs_set_identity

        result = json.loads(amfs_set_identity("agent-a", "doing stuff"))
        assert result["new_identity"] == "agent-a"
        assert "agent-a" in srv._memories
        assert srv._active_identity == "agent-a"

    def test_set_identity_idempotent(self) -> None:
        from amfs_mcp.server import amfs_set_identity

        amfs_set_identity("agent-a")
        result = json.loads(amfs_set_identity("agent-a"))
        assert result["status"] == "already_active"

    def test_set_identity_rejects_during_cooldown(self) -> None:
        import amfs_mcp.server as srv
        from amfs_mcp.server import amfs_set_identity

        amfs_set_identity("agent-a")
        srv._last_activity = srv.time.monotonic()

        result = json.loads(amfs_set_identity("agent-b"))
        assert result.get("error") == "identity_conflict"
        assert result["current_identity"] == "agent-a"
        assert result["requested_identity"] == "agent-b"
        assert srv._active_identity == "agent-a"

    def test_set_identity_allows_after_cooldown(self) -> None:
        import amfs_mcp.server as srv
        from amfs_mcp.server import amfs_set_identity

        amfs_set_identity("agent-a")
        srv._last_activity = srv.time.monotonic() - srv._IDENTITY_COOLDOWN_SECONDS - 1

        result = json.loads(amfs_set_identity("agent-b"))
        assert result["new_identity"] == "agent-b"
        assert srv._active_identity == "agent-b"

    def test_separate_memories_per_identity(self) -> None:
        import amfs_mcp.server as srv
        from amfs_mcp.server import amfs_set_identity, amfs_write

        amfs_set_identity("agent-a")
        amfs_write("svc", "key-a", '"from-a"')

        srv._last_activity = 0.0
        amfs_set_identity("agent-b")
        amfs_write("svc", "key-b", '"from-b"')

        assert "agent-a" in srv._memories
        assert "agent-b" in srv._memories
        assert srv._memories["agent-a"].agent_id == "agent-a"
        assert srv._memories["agent-b"].agent_id == "agent-b"

    def test_writes_use_active_identity(self) -> None:
        from amfs_mcp.server import amfs_set_identity, amfs_write

        amfs_set_identity("writer-agent")
        raw = json.loads(amfs_write("svc", "key", '"value"'))
        assert raw["entry"]["provenance"]["agent_id"] == "writer-agent"


class TestPerClientStickyIdentity:
    """Sticky identity is scoped per client so a work identity set in one
    client (Cursor) never leaks into another (Claude Desktop → "cli")."""

    @pytest.fixture(autouse=True)
    def _isolate_home(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        # Neutralise inherited IDE signals so each test picks its own platform.
        for var in ("CURSOR_SESSION_ID", "VSCODE_PID", "CLAUDE_CODE_SESSION"):
            monkeypatch.delenv(var, raising=False)

    def test_identity_file_scoped_by_platform(self, monkeypatch) -> None:
        import amfs_mcp.server as srv

        monkeypatch.setenv("CURSOR_SESSION_ID", "x")
        assert srv._identity_file().name == ".identity-cursor"

        monkeypatch.delenv("CURSOR_SESSION_ID", raising=False)
        # Claude Desktop / generic desktop detects as "cli".
        assert srv._identity_file().name == ".identity-cli"

    def test_no_cross_client_leak(self, monkeypatch) -> None:
        import amfs_mcp.server as srv

        # Cursor sets a work identity...
        monkeypatch.setenv("CURSOR_SESSION_ID", "x")
        srv._save_sticky_identity("value-metrics-agent")
        assert srv._load_sticky_identity() == "value-metrics-agent"

        # ...Claude Desktop (cli) must NOT inherit it.
        monkeypatch.delenv("CURSOR_SESSION_ID", raising=False)
        assert srv._load_sticky_identity() is None

    def test_save_retires_legacy_shared_file(self, monkeypatch) -> None:
        import amfs_mcp.server as srv

        monkeypatch.setenv("CURSOR_SESSION_ID", "x")
        legacy = srv._legacy_identity_file()
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text("old-shared-id", encoding="utf-8")

        srv._save_sticky_identity("new-id")
        assert not legacy.is_file()
        assert srv._load_sticky_identity() == "new-id"


class TestCallCompat:
    """amfs-mcp must not hard-fail against an older amfs SDK whose
    retrieve/search predate include_artifacts (version skew)."""

    def test_drops_include_artifacts_on_typeerror(self) -> None:
        from amfs_mcp.server import _call_compat

        seen: dict = {}

        def old_retrieve(**kwargs):
            if "include_artifacts" in kwargs:
                raise TypeError(
                    "retrieve() got an unexpected keyword argument 'include_artifacts'"
                )
            seen.update(kwargs)
            return "ok"

        assert _call_compat(old_retrieve, query="q", include_artifacts=True) == "ok"
        assert "include_artifacts" not in seen
        assert seen["query"] == "q"

    def test_passes_through_when_supported(self) -> None:
        from amfs_mcp.server import _call_compat

        result = _call_compat(lambda **kw: kw, query="q", include_artifacts=False)
        assert result["include_artifacts"] is False

    def test_reraises_unrelated_typeerror(self) -> None:
        from amfs_mcp.server import _call_compat

        def broken(**kwargs):
            raise TypeError("something else entirely")

        with pytest.raises(TypeError, match="something else"):
            _call_compat(broken, query="q", include_artifacts=True)


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
