"""What `install-mcp.sh` writes into a user's MCP config, for each invocation.

The script is published at a raw GitHub URL and piped straight into bash, so a
change to it reaches everyone who runs the one-liner on the next release — there
is no version anyone is pinned to. Until now nothing tested it at all.

The reason these exist is narrower than "the installer should work". A keyless
`--remote` mode was added, and the only way it could do damage is by changing
what the two invocations already in use produce: the bare `curl | bash`, and the
`--api-key` form the onboarding wizard hands out. Both are pinned below exactly,
so a future edit that reroutes them shows up here instead of in a support thread
about an agent that stopped remembering anything.

The functions are exercised by sourcing the script with a stubbed `main`, which
is what lets a config be built without touching a real client's files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "install-mcp.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is needed to run the installer"
)


#: The script's entry point, called on its own final line.
ENTRY_POINT = "main"


def _sourceable(tmp: Path) -> Path:
    """A copy of the script with its final `main` call removed.

    Stubbing `main` before sourcing does not work — the script defines its own
    further down, which replaces the stub, and then calls it. Rather than add a
    test-only escape hatch to a file people pipe from curl into bash, the entry
    point is stripped here. `test_the_entry_point_is_still_the_last_line` is what
    keeps that honest: if the script stops ending this way, that fails rather
    than every assertion below quietly testing a real install.
    """
    lines = SCRIPT.read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    assert lines[-1].strip() == ENTRY_POINT
    copy = tmp / "install-mcp-under-test.sh"
    copy.write_text("\n".join(lines[:-1]) + "\n")
    return copy


def _run(args: str, snippet: str, tmp_path: Path) -> str:
    """Source the script with the given flags and run one snippet against it.

    Parsing and the config builders run while nothing detects a client or writes
    a file. `--help` and `--uninstall` are the only paths that exit during
    parsing, and neither is used here.
    """
    script = _sourceable(tmp_path)
    program = f"""
set -euo pipefail
# shellcheck disable=SC1090
source {script!s} {args}
{snippet}
"""
    proc = subprocess.run(
        ["bash", "-c", program], capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return proc.stdout


@pytest.fixture()
def build_config(tmp_path: Path):
    """Build the MCP config block the script would write, for some flags."""

    def _build(args: str = "") -> dict:
        return json.loads(_run(args, "build_mcp_json", tmp_path))

    return _build


def test_the_entry_point_is_still_the_last_line() -> None:
    """Everything below depends on being able to strip it."""
    lines = [ln for ln in SCRIPT.read_text().splitlines() if ln.strip()]
    assert lines[-1].strip() == ENTRY_POINT


class TestTheDefaultInvocationIsUnchanged:
    """`curl -sSL … | bash`, with no flags. The most-run form of the script."""

    def test_it_still_installs_the_open_source_server_over_stdio(self, build_config) -> None:
        built = build_config()
        assert built["args"] == ["--refresh", "amfs-mcp-server"]
        assert built["command"].endswith("uvx")

    def test_it_carries_no_credentials_and_no_url(self, build_config) -> None:
        """Local filesystem storage: there is nothing to authenticate to."""
        built = build_config()
        assert built["env"] == {}
        assert "url" not in built
        assert "type" not in built


class TestThePastedKeyInvocationIsUnchanged:
    """`--api-key …`, which is what the onboarding wizard hands out."""

    def test_it_installs_the_pro_server_with_the_key_in_the_environment(self, build_config) -> None:
        built = build_config("--api-key amfs_test_key_123")
        assert built["args"] == ["--refresh", "amfs-mcp-server-pro"]
        assert built["env"]["AMFS_API_KEY"] == "amfs_test_key_123"
        assert built["env"]["AMFS_HTTP_URL"] == "https://amfs-login.sense-lab.ai"

    def test_api_url_still_overrides_the_default_host(self, build_config) -> None:
        built = build_config("--api-key k --api-url https://staging.example.com")
        assert built["env"]["AMFS_HTTP_URL"] == "https://staging.example.com"

    def test_it_is_still_a_local_process_rather_than_a_url(self, build_config) -> None:
        built = build_config("--api-key k")
        assert "url" not in built


class TestTheRemoteMode:
    def test_it_writes_a_url_and_no_command(self, build_config) -> None:
        """The point of it: nothing runs locally, so nothing needs installing."""
        built = build_config("--remote")
        assert built == {"type": "http", "url": "https://mcp.sense-lab.ai/mcp"}

    def test_it_carries_no_key_because_there_is_nothing_to_carry(self, build_config) -> None:
        built = build_config("--remote")
        assert "env" not in built
        assert not any("key" in str(v).lower() for v in built.values())

    def test_the_endpoint_can_be_pointed_elsewhere(self, build_config) -> None:
        """So a dev install does not reach the production endpoint."""
        built = build_config("--mcp-url https://mcp.dev.example.com/mcp")
        assert built["url"] == "https://mcp.dev.example.com/mcp"

    def test_joining_a_room_implies_it(self, build_config) -> None:
        """An invite admits a person, and only the browser sign-in has one."""
        built = build_config("--join tok-abc")
        assert built["type"] == "http"


class TestContradictoryFlagsAreRefused:
    """Both would otherwise resolve silently, and to the surprising answer.

    `--remote` beat `--api-key` in the order the branches were written, so
    passing both produced a keyless config and dropped the key without a word.
    """

    def _fails(self, args: str, tmp_path: Path) -> str:
        proc = subprocess.run(
            ["bash", "-c", f"source {_sourceable(tmp_path)!s} {args}"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode != 0, f"expected a refusal, got {proc.stdout!r}"
        return proc.stderr

    def test_remote_and_api_key_together(self, tmp_path: Path) -> None:
        assert "pass one" in self._fails("--remote --api-key k", tmp_path)

    def test_join_and_uninstall_together(self, tmp_path: Path) -> None:
        assert "opposite" in self._fails("--join tok --uninstall", tmp_path)

    def test_an_api_key_with_a_join_keeps_the_key_path(self, build_config) -> None:
        """Not a contradiction: a key already identifies the user who made it,
        and that user is the one the invite would admit."""
        built = build_config("--api-key amfs_k --join tok-abc")
        assert built["args"] == ["--refresh", "amfs-mcp-server-pro"]
        assert built["env"]["AMFS_API_KEY"] == "amfs_k"


class TestClientsThatCannotHoldAUrl:
    """Claude Desktop and Windsurf, in remote mode.

    Claude Desktop's config file takes stdio servers only, and Windsurf calls the
    field `serverUrl`. Writing `{"type": "http", "url": …}` into either is
    ignored, so the script reporting "Configured" would be a lie that also hides
    the one place the person could have fixed it.
    """

    def _configure(self, client: str, args: str, tmp_path: Path) -> str:
        home = tmp_path / "home"
        home.mkdir()
        out = _run(
            args,
            f'HOME={home!s} configure_client {client} 2>&1 || true',
            tmp_path,
        )
        return out

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_remote_mode_tells_the_user_instead_of_writing(
        self, client: str, tmp_path: Path
    ) -> None:
        out = self._configure(client, "--remote", tmp_path)
        assert "cannot be configured from here" in out
        assert "https://mcp.sense-lab.ai/mcp" in out
        assert "Configured" not in out

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_remote_mode_leaves_the_config_file_alone(
        self, client: str, tmp_path: Path
    ) -> None:
        """An ignored entry is not the only cost — the file is shared."""
        self._configure(client, "--remote", tmp_path)
        assert not list((tmp_path / "home").rglob("*.json"))

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_the_stdio_path_still_writes_the_file(
        self, client: str, tmp_path: Path
    ) -> None:
        """The regression this could have caused, stated as a test."""
        out = self._configure(client, "", tmp_path)
        assert "Configured" in out
        written = list((tmp_path / "home").rglob("*.json"))
        assert len(written) == 1
        assert "amfs-mcp-server" in written[0].read_text()

    def test_cursor_is_configured_by_file_in_remote_mode(
        self, tmp_path: Path
    ) -> None:
        """It understands the URL form, so it is not in the told-by-hand set."""
        out = self._configure("cursor", "--remote", tmp_path)
        assert "Configured Cursor" in out
        written = list((tmp_path / "home").rglob("mcp.json"))
        assert len(written) == 1
        assert "https://mcp.sense-lab.ai/mcp" in written[0].read_text()


class TestTheJoinTokenIsNotLeakedByTheScript:
    def test_it_does_not_appear_in_the_help_text(self) -> None:
        proc = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0
        assert "--join" in proc.stdout
        # The flag is documented; no example carries a token shaped like a real
        # one, so copying the help text cannot hand anyone a live invite.
        assert "<link|token>" in proc.stdout
