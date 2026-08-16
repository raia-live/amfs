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

    #: Where each client's config lives under a fake HOME, on this platform.
    #:
    #: Only macOS and Linux differ, and only for Claude Desktop; the script picks
    #: between them itself, so the test asks it rather than guessing.
    def _config_path(self, client: str, home: Path, tmp_path: Path) -> Path:
        fn = {
            "claude-desktop": "claude_desktop_config_path",
            "windsurf": "windsurf_config_path",
            "cursor": "cursor_config_path",
        }[client]
        return Path(_run("", f"HOME={home!s} {fn}", tmp_path).strip())

    def _existing_stdio_config(self, client: str, home: Path) -> Path:
        """A config file as a previous stdio install would have left it."""
        home.mkdir(exist_ok=True)
        path = self._config_path(client, home, home.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mcpServers": {
                "senselab": {
                    "command": "/opt/homebrew/bin/uvx",
                    "args": ["--refresh", "amfs-mcp-server-pro"],
                    "env": {"AMFS_API_KEY": "amfs_old_key"},
                },
                "somebody-else": {"command": "node", "args": ["other.js"]},
            }
        }, indent=4))
        return path

    def _configure(
        self, client: str, args: str, tmp_path: Path, home: Path | None = None
    ) -> str:
        if home is None:
            home = tmp_path / "home"
            home.mkdir(exist_ok=True)
        return _run(
            args,
            f'HOME={home!s} configure_client {client} 2>&1 || true',
            tmp_path,
        )

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_remote_mode_tells_the_user_instead_of_writing(
        self, client: str, tmp_path: Path
    ) -> None:
        out = self._configure(client, "--remote", tmp_path)
        assert "cannot be configured from here" in out
        assert "https://mcp.sense-lab.ai/mcp" in out
        assert "Configured" not in out

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_remote_mode_writes_no_new_config_file(
        self, client: str, tmp_path: Path
    ) -> None:
        """An ignored entry is not the only cost — the file is shared."""
        self._configure(client, "--remote", tmp_path)
        assert not list((tmp_path / "home").rglob("*.json"))

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_it_takes_out_an_existing_local_entry(
        self, client: str, tmp_path: Path
    ) -> None:
        """Someone switching an existing install over to the hosted endpoint.

        Not writing was only half of it. Left in place, the old block keeps the
        client launching the local server — so either they add the connector and
        have two, or they do not and the migration silently did nothing, while
        the summary says they are on the hosted endpoint.
        """
        home = tmp_path / "home"
        path = self._existing_stdio_config(client, home)

        out = self._configure(client, "--remote", tmp_path, home=home)

        assert "Removed the old local" in out
        assert "senselab" not in path.read_text()

    @pytest.mark.parametrize("client", ["claude-desktop", "windsurf"])
    def test_it_leaves_other_servers_in_that_file_alone(
        self, client: str, tmp_path: Path
    ) -> None:
        """The file is shared with every other MCP server the person uses."""
        home = tmp_path / "home"
        path = self._existing_stdio_config(client, home)

        self._configure(client, "--remote", tmp_path, home=home)

        assert json.loads(path.read_text())["mcpServers"]["somebody-else"]

    def test_there_is_nothing_to_say_when_there_was_no_old_entry(
        self, tmp_path: Path
    ) -> None:
        """A first-time install should not be told something was removed."""
        out = self._configure("claude-desktop", "--remote", tmp_path)
        assert "Removed the old local" not in out

    def test_a_file_it_cannot_parse_is_reported_not_claimed(
        self, tmp_path: Path
    ) -> None:
        """The case a grep could not distinguish.

        `remove_mcp_config` leaves an unparseable file untouched and exits 0
        either way, so reporting from the attempt rather than the result claimed
        a removal that did not happen — putting back the stale local server the
        removal exists to prevent, with a success line over the top of it.
        """
        home = tmp_path / "home"
        home.mkdir()
        path = self._config_path("claude-desktop", home, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"mcpServers": {"senselab": {"command": "uvx"},,,')

        out = self._configure("claude-desktop", "--remote", tmp_path, home=home)

        assert "Removed the old local" not in out
        assert "could not be read" in out
        assert "remove it by hand" in out

    def test_a_check_that_cannot_run_at_all_warns_rather_than_going_quiet(
        self, tmp_path: Path
    ) -> None:
        """No python3 on PATH, which is the shape of every unexpected failure.

        The state used to arrive as an exit status, which had to carry both the
        answer and whether asking worked. A missing interpreter exits 127, and as
        a number that fell outside the three-way branch — so nothing was removed,
        nothing was said, and the summary still reported the hosted endpoint.
        """
        home = tmp_path / "home"
        path = self._existing_stdio_config("claude-desktop", home)

        out = _run(
            "--remote",
            f"python3() {{ return 127; }}\n"
            f"HOME={home!s} configure_client claude-desktop 2>&1 || true",
            tmp_path,
        )

        assert "Removed the old local" not in out
        assert "could not be read" in out
        assert "senselab" in path.read_text()

    def test_a_config_file_it_cannot_write_does_not_abort_the_install(
        self, tmp_path: Path
    ) -> None:
        """`set -e` is on, so a bare removal call exiting non-zero took the whole
        script with it — before printing the connector instructions, and before
        the warning that the old entry is still in place."""
        home = tmp_path / "home"
        path = self._existing_stdio_config("claude-desktop", home)

        out = _run(
            "--remote",
            f"remove_mcp_config() {{ return 1; }}\n"
            f"HOME={home!s} configure_client claude-desktop 2>&1 || true",
            tmp_path,
        )

        assert "Could not confirm" in out
        assert "Add it once, by hand" in out
        assert "senselab" in path.read_text()

    def test_a_removal_that_cannot_be_confirmed_is_not_called_a_removal(
        self, tmp_path: Path
    ) -> None:
        """Only "the entry is absent" is evidence of a removal.

        The check after removing was a plain if/else at first, so the two states
        that are not a removal — still present, and unreadable — collapsed into
        the success branch between them. Stubbing the removal to do nothing is the
        cheapest way to hold that open: if any non-absent answer ever reads as
        success again, this fails.
        """
        home = tmp_path / "home"
        path = self._existing_stdio_config("claude-desktop", home)

        out = _run(
            "--remote",
            f"remove_mcp_config() {{ :; }}\n"
            f"HOME={home!s} configure_client claude-desktop 2>&1 || true",
            tmp_path,
        )

        assert "Removed the old local" not in out
        assert "Could not confirm" in out
        assert "senselab" in path.read_text()

    def test_the_word_senselab_elsewhere_is_not_an_entry(
        self, tmp_path: Path
    ) -> None:
        """A path or another server's arguments can carry the name."""
        home = tmp_path / "home"
        home.mkdir()
        path = self._config_path("claude-desktop", home, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mcpServers": {
                "other": {"command": "node", "args": ["/opt/senselab/x.js"]}
            }
        }))

        out = self._configure("claude-desktop", "--remote", tmp_path, home=home)

        assert "Removed the old local" not in out
        assert "Could not remove" not in out
        assert "not readable as JSON" not in out
        assert json.loads(path.read_text())["mcpServers"]["other"]

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


class TestTheKeyOnDiskSurvivesBeingReplaced:
    """The `senselab` block is the only copy of the user's API key.

    SenseLab stores keys hashed, so no part of the product can hand the same one
    back — a key this script overwrites or deletes is gone, and every client
    still pointing at it is unrepairable without minting a new one and visiting
    them all.

    That is not hypothetical. It happened here: an `--uninstall` and then a run
    carrying a six-character placeholder left a real key unrecoverable. Nothing
    warned, nothing kept a copy, and the file had been rewritten by the time
    anyone looked. The three paths that reach it are all normal use — removing
    an install, migrating one to `--remote`, and re-running the one-liner, which
    without `--api-key` quietly downgrades a keyed install to the free server.

    So the rule these hold: if a key is on disk and is about to stop being on
    disk, a copy is made first and its location is said out loud. When nothing
    is at risk, no copy appears — a backup on every idempotent re-run would
    train people to ignore them.
    """

    #: A key long enough to be recognisably real rather than a placeholder.
    REAL_KEY = "amfs_xZnxjg8Jhl1Sd0liWX2uga_g9qdqgo18xSKDBbMtJaU"

    def _config_path(self, client: str, home: Path, tmp_path: Path) -> Path:
        fn = {
            "cursor": "cursor_config_path",
            "claude-desktop": "claude_desktop_config_path",
        }[client]
        return Path(_run("", f"HOME={home!s} {fn}", tmp_path).strip())

    def _keyed_config(
        self, home: Path, tmp_path: Path, client: str = "cursor"
    ) -> Path:
        """A config as a working `--api-key` install would have left it."""
        home.mkdir(exist_ok=True)
        path = self._config_path(client, home, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mcpServers": {
                "senselab": {
                    "command": "/opt/homebrew/bin/uvx",
                    "args": ["--refresh", "amfs-mcp-server-pro"],
                    "env": {
                        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
                        "AMFS_API_KEY": self.REAL_KEY,
                    },
                },
                "somebody-else": {"command": "node", "args": ["other.js"]},
            }
        }, indent=4))
        return path

    def _configure(
        self, client: str, args: str, tmp_path: Path, home: Path, extra: str = ""
    ) -> str:
        return _run(
            args,
            f"{extra}HOME={home!s} configure_client {client} 2>&1 || true",
            tmp_path,
        )

    def _backups(self, path: Path) -> list[Path]:
        return sorted(path.parent.glob(f"{path.name}.senselab-backup-*"))

    def _the_one_backup(self, path: Path) -> Path:
        found = self._backups(path)
        assert len(found) == 1, f"expected one copy, found {found}"
        return found[0]

    def test_an_uninstall_keeps_the_key_it_takes_away(self, tmp_path: Path) -> None:
        """Anyone uninstalling to reinstall cleanly is about to need this."""
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        out = self._configure("cursor", "--uninstall", tmp_path, home)

        assert self.REAL_KEY in self._the_one_backup(path).read_text()
        assert "senselab" not in path.read_text()
        assert "Kept a copy" in out

    def test_a_keyless_rerun_keeps_the_key_it_downgrades(
        self, tmp_path: Path
    ) -> None:
        """The incident, exactly: `curl | bash` over a working keyed install.

        No flags means the free local server and no `env` block at all, so the
        key is dropped by a run whose output otherwise reads like a success.
        """
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        self._configure("cursor", "", tmp_path, home)

        assert self.REAL_KEY in self._the_one_backup(path).read_text()
        assert self.REAL_KEY not in path.read_text()

    def test_a_rerun_with_a_different_key_keeps_the_old_one(
        self, tmp_path: Path
    ) -> None:
        """A placeholder pasted out of a doc, or a second account's key."""
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        self._configure("cursor", "--api-key amfs_k", tmp_path, home)

        assert self.REAL_KEY in self._the_one_backup(path).read_text()
        assert "amfs_k" in json.loads(path.read_text())[
            "mcpServers"]["senselab"]["env"]["AMFS_API_KEY"]

    def test_the_remote_migration_keeps_the_key_it_clears(
        self, tmp_path: Path
    ) -> None:
        """`--remote` deletes the stdio block as a migration, key and all.

        The user asked to switch endpoints, not to give up a credential, and
        going back afterwards is what needs it.
        """
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path, client="claude-desktop")

        out = self._configure("claude-desktop", "--remote", tmp_path, home)

        assert self.REAL_KEY in self._the_one_backup(path).read_text()
        assert "Removed the old local" in out

    def test_a_rerun_with_the_same_key_leaves_nothing_behind(
        self, tmp_path: Path
    ) -> None:
        """Re-running the wizard's own line is how people repair a config.

        Nothing stops existing, so a copy would be litter — and a directory of
        backups nobody needed is how the one that matters gets ignored.
        """
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        out = self._configure(
            "cursor", f"--api-key {self.REAL_KEY}", tmp_path, home
        )

        assert self._backups(path) == []
        assert "Kept a copy" not in out

    def test_a_first_install_leaves_nothing_behind(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()

        out = self._configure("cursor", f"--api-key {self.REAL_KEY}", tmp_path, home)

        path = self._config_path("cursor", home, tmp_path)
        assert self._backups(path) == []
        assert "Kept a copy" not in out

    def test_a_config_holding_no_key_is_not_copied(self, tmp_path: Path) -> None:
        """The free local server has no `env`, so there is nothing to lose."""
        home = tmp_path / "home"
        home.mkdir()
        path = self._config_path("cursor", home, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mcpServers": {"senselab": {"command": "uvx", "args": ["amfs-mcp-server"]}}
        }))

        out = self._configure("cursor", "--uninstall", tmp_path, home)

        assert self._backups(path) == []
        assert "Kept a copy" not in out

    def test_the_copy_is_not_left_readable_by_everyone(
        self, tmp_path: Path
    ) -> None:
        """It is a credential in a directory that may not be private."""
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        self._configure("cursor", "--uninstall", tmp_path, home)

        assert self._the_one_backup(path).stat().st_mode & 0o777 == 0o600

    def test_it_says_where_the_copy_is_and_why_it_matters(
        self, tmp_path: Path
    ) -> None:
        """A backup nobody is told about is not a rescue.

        The path has to be in the output because the name carries a timestamp,
        and the reason has to be there because "just make another one" is the
        obvious assumption and it is wrong — the same key never comes back.
        """
        home = tmp_path / "home"
        path = self._keyed_config(home, tmp_path)

        out = self._configure("cursor", "--uninstall", tmp_path, home)

        assert str(self._the_one_backup(path)) in out
        assert "cannot reissue the same one" in out

    def test_a_file_it_cannot_read_is_not_claimed_as_saved(
        self, tmp_path: Path
    ) -> None:
        """An unparseable config is left alone by the removal, so nothing is
        lost and nothing needs keeping — but claiming a copy that does not exist
        would be worse than saying nothing."""
        home = tmp_path / "home"
        home.mkdir()
        path = self._config_path("cursor", home, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"mcpServers": {"senselab": {"env": {,,,')

        out = self._configure("cursor", "--uninstall", tmp_path, home)

        assert self._backups(path) == []
        assert "Kept a copy" not in out

    def test_a_copy_that_fails_says_so_instead_of_going_quiet(
        self, tmp_path: Path
    ) -> None:
        """The write that follows will probably fail too, but not certainly —
        and a silent failure here is the exact shape of the original loss."""
        home = tmp_path / "home"
        self._keyed_config(home, tmp_path)

        out = self._configure(
            "cursor", "--uninstall", tmp_path, home, extra="cp() { return 1; }\n"
        )

        assert "Could not copy" in out
        assert "cannot" in out and "hand the same one back" in out

    def test_the_uninstall_prompt_says_it_before_asking(
        self, tmp_path: Path
    ) -> None:
        """Order is the whole point: after the answer, the key is already gone."""
        home = tmp_path / "home"
        home.mkdir()

        out = _run(
            "--uninstall",
            "detect_clients() { DETECTED_CLIENTS=(cursor); }\n"
            "configure_client() { :; }\n"
            f"HOME={home!s} prompt_clients 2>&1 || true",
            tmp_path,
        )

        assert "takes your API key with it" in out
        assert out.index("takes your API key") < out.index("Remove AMFS from all")

    def test_a_cli_managed_store_is_kept_when_it_holds_the_key(
        self, tmp_path: Path
    ) -> None:
        """Claude Code is configured through its own CLI, not by writing files.

        There is no block of ours to compare, so the marker is the test: finding
        AMFS_API_KEY in the file that CLI keeps means a copy really does preserve
        the key, and not finding it means saying nothing.
        """
        home = tmp_path / "home"
        home.mkdir()
        store = home / ".claude.json"
        store.write_text(json.dumps({
            "mcpServers": {"senselab": {"env": {"AMFS_API_KEY": self.REAL_KEY}}}
        }))

        out = self._configure(
            "claude-code", "--uninstall", tmp_path, home,
            extra="command() { :; }\nclaude() { :; }\n",
        )

        assert self.REAL_KEY in self._the_one_backup(store).read_text()
        assert "Kept a copy" in out

    def _cli_store_decision(
        self, stored: str, incoming: str, tmp_path: Path, quote: str = '"'
    ) -> tuple[str, list[Path]]:
        """Ask `backup_cli_store` directly, for one stored/incoming pair.

        The two CLIs own these files, so the shape that matters is a line
        carrying the name and the quoted value — JSON uses `:`, TOML uses `=`,
        and either may quote with `'`. Calling the function rather than a whole
        client path is what makes the quoting the only variable.
        """
        store = tmp_path / "cli-store"
        store.write_text(
            f"  AMFS_API_KEY = {quote}{stored}{quote}\n"
            '  command = "uvx"\n  args = ["--refresh", "amfs-mcp-server-pro"]\n'
        )
        out = _run("", f'backup_cli_store {store!s} "{incoming}" 2>&1 || true', tmp_path)
        return out, self._backups(store)

    def test_a_shorter_key_is_not_mistaken_for_the_one_stored(
        self, tmp_path: Path
    ) -> None:
        """The bug this function exists to prevent, reintroduced inside it.

        Matching the incoming key as a bare substring meant a placeholder like
        `amfs_k` counted as already present in any real key beginning with those
        characters. The copy was skipped, `codex mcp remove` deleted the entry,
        and the key was gone — which is the original incident exactly, and short
        keys are the ones people paste by mistake.
        """
        stored = "amfs_kREALKEYcontinuesWellPastThePlaceholder"

        out, backups = self._cli_store_decision(stored, "amfs_k", tmp_path)

        assert len(backups) == 1, "a prefix must not count as the stored key"
        assert stored in backups[0].read_text()
        assert "Kept a copy" in out

    def test_a_different_key_entirely_is_kept(self, tmp_path: Path) -> None:
        out, backups = self._cli_store_decision(
            self.REAL_KEY, "amfs_someOtherAccountsKey", tmp_path
        )
        assert len(backups) == 1
        assert self.REAL_KEY in backups[0].read_text()
        assert "Kept a copy" in out

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_re_adding_the_same_key_keeps_nothing(
        self, quote: str, tmp_path: Path
    ) -> None:
        """Nothing stops existing, in either quoting style those CLIs use."""
        out, backups = self._cli_store_decision(
            self.REAL_KEY, self.REAL_KEY, tmp_path, quote=quote
        )
        assert backups == []
        assert "Kept a copy" not in out

    def test_a_longer_key_containing_the_stored_one_is_kept(
        self, tmp_path: Path
    ) -> None:
        """The mirror image, in case the comparison is ever loosened again."""
        stored = "amfs_short"

        _, backups = self._cli_store_decision(stored, f"{stored}AndThenSome", tmp_path)

        assert len(backups) == 1

    def test_a_cli_managed_store_without_the_key_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        """Where a given version of those tools keeps it is not ours to assume."""
        home = tmp_path / "home"
        home.mkdir()
        store = home / ".claude.json"
        store.write_text(json.dumps({"projects": {"/tmp/x": {"allowedTools": []}}}))

        out = self._configure(
            "claude-code", "--uninstall", tmp_path, home,
            extra="command() { :; }\nclaude() { :; }\n",
        )

        assert self._backups(store) == []
        assert "Kept a copy" not in out


class TestTheClosingSummaryDoesNotOverclaim:
    """A full run, end to end, for what the last lines say happened.

    The summary used to be written from the flags rather than from the outcome:
    `--remote` printed "Configured the hosted SenseLab endpoint" and "restart
    your IDE/app" whether or not a single file had been written. A remote run
    that found only Claude Desktop wrote nothing and still reported success,
    which sends someone to look for a connection instead of at the one
    instruction that would have worked.
    """

    def _run_main(self, args: str, tmp_path: Path) -> str:
        home = tmp_path / "home"
        home.mkdir()
        proc = subprocess.run(
            ["bash", str(SCRIPT), *args.split(), "--yes"],
            capture_output=True,
            text=True,
            timeout=120,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        return proc.stdout + proc.stderr

    def test_a_client_it_cannot_write_to_is_not_called_configured(
        self, tmp_path: Path
    ) -> None:
        out = self._run_main("--remote --client claude-desktop", tmp_path)
        assert "Nothing was configured automatically" in out
        assert "Restart your IDE" not in out

    def test_it_still_names_the_endpoint_that_needs_pasting(
        self, tmp_path: Path
    ) -> None:
        """The run is not useless — it is the instruction that is the product."""
        out = self._run_main("--remote --client claude-desktop", tmp_path)
        assert "https://mcp.sense-lab.ai/mcp" in out
        assert "Add custom connector" in out

    def test_a_client_it_did_write_to_is_told_to_restart(
        self, tmp_path: Path
    ) -> None:
        out = self._run_main("--remote --client cursor", tmp_path)
        assert "Restart your IDE" in out
        assert "Configured the hosted SenseLab endpoint" in out


class TestTheJoinCopyMatchesHowTheUserWillSignIn:
    """`--join` says what to do next, and next is not the same in both modes.

    In remote mode there is a browser sign-in to wait for. On the key path there
    is not — the key already names its owner — so "once your client has signed
    in" describes a step that never happens and reads as a stuck install. The
    browser fallback is worse than merely wrong there: a browser admits whoever
    is logged into it, which need not be the account the key belongs to, so
    following it joins the room as one account while the connected agent belongs
    to another.
    """

    def _steps(self, args: str, tmp_path: Path) -> str:
        return _run(args, "join_next_steps", tmp_path)

    def test_remote_mode_waits_for_the_sign_in(self, tmp_path: Path) -> None:
        assert "Once your client has signed in" in self._steps(
            "--join tok-abc", tmp_path
        )

    def test_the_key_path_does_not_wait_for_one(self, tmp_path: Path) -> None:
        out = self._steps("--api-key amfs_k --join tok-abc", tmp_path)
        assert "Once your client has signed in" not in out
        assert "Ask your agent:" in out

    def test_the_key_path_warns_that_a_browser_may_be_someone_else(
        self, tmp_path: Path
    ) -> None:
        out = self._steps("--api-key amfs_k --join tok-abc", tmp_path)
        assert "whoever is signed in there" in out

    def test_only_remote_calls_the_browser_the_same_thing(
        self, tmp_path: Path
    ) -> None:
        assert "does the same thing" in self._steps("--join tok", tmp_path)
        assert "does the same thing" not in self._steps(
            "--api-key amfs_k --join tok", tmp_path
        )

    @pytest.mark.parametrize(
        "args", ["--join tok-abc", "--api-key amfs_k --join tok-abc"]
    )
    def test_both_modes_name_the_tool_and_the_token(
        self, args: str, tmp_path: Path
    ) -> None:
        out = self._steps(args, tmp_path)
        assert "amfs_room_redeem, invite tok-abc" in out

    def test_it_says_nothing_at_all_without_an_invite(self, tmp_path: Path) -> None:
        assert self._steps("--remote", tmp_path).strip() == ""


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
