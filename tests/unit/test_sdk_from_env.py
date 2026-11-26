"""``AgentMemory.from_env`` connects to the hosted API or stops; the constructor
keeps its local default and says so once.

The constructor does not read ``AMFS_HTTP_URL`` / ``AMFS_API_KEY``, and several
guides said it did. With only the variables set, ``AgentMemory(agent_id=...)``
wrote to a local ``.amfs/`` folder without a word while the hosted dashboard
stayed empty. Making the constructor read them would fix that and break the
opposite case — a shell that still has them exported from other work starts
writing to a hosted account, which has happened — so the default stays local,
the mismatch is logged, and reaching the hosted API from the environment is
something a caller asks for by name.
"""

from __future__ import annotations

import builtins
import logging

import amfs.memory as memory_module
import pytest
from amfs import AgentMemory

URL = "https://amfs.example.test"
KEY = "amfs_sk_test"


@pytest.fixture(autouse=True)
def _fresh_warning(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Each test starts with the once-per-process warning unspent, and in a
    directory of its own so a default (filesystem) memory writes nowhere real."""
    monkeypatch.setattr(memory_module, "_env_ignored_warned", False)
    monkeypatch.chdir(tmp_path)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == "amfs.memory" and r.levelno == logging.WARNING
    ]


class TestFromEnv:
    def test_builds_an_http_adapter_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        mem = AgentMemory.from_env("agent-a")
        adapter = mem.adapter
        assert type(adapter).__name__ == "HttpAdapter"
        assert adapter._base == URL
        assert adapter._api_key == KEY
        assert mem.agent_id == "agent-a"

    def test_arguments_override_the_environment(self, monkeypatch) -> None:
        """A second key for a second agent, on the same URL."""
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        mem = AgentMemory.from_env("agent-b", api_key="amfs_sk_other")
        assert mem.adapter._base == URL
        assert mem.adapter._api_key == "amfs_sk_other"

    def test_works_from_arguments_alone(self) -> None:
        mem = AgentMemory.from_env("agent-a", base_url=URL + "/", api_key=KEY)
        assert mem.adapter._base == URL

    def test_other_arguments_reach_the_constructor(self, monkeypatch) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        mem = AgentMemory.from_env("agent-a", session_id="sess-9", branch="experiment")
        assert mem.session_id == "sess-9"
        assert mem.branch == "experiment"

    def test_surrounding_whitespace_is_not_part_of_the_value(self, monkeypatch) -> None:
        """A value pasted into a ``.env`` with a trailing newline or space."""
        monkeypatch.setenv("AMFS_HTTP_URL", f"  {URL}\n")
        monkeypatch.setenv("AMFS_API_KEY", f"{KEY} ")
        mem = AgentMemory.from_env("agent-a")
        assert mem.adapter._base == URL
        assert mem.adapter._api_key == KEY

    @pytest.mark.parametrize(
        ("env", "named", "not_named"),
        [
            ({}, ["AMFS_HTTP_URL", "AMFS_API_KEY"], []),
            ({"AMFS_HTTP_URL": URL}, ["AMFS_API_KEY"], ["AMFS_HTTP_URL"]),
            ({"AMFS_API_KEY": KEY}, ["AMFS_HTTP_URL"], ["AMFS_API_KEY"]),
            ({"AMFS_HTTP_URL": "   ", "AMFS_API_KEY": KEY}, ["AMFS_HTTP_URL"], ["AMFS_API_KEY"]),
        ],
    )
    def test_a_missing_value_raises_and_names_it(
        self, monkeypatch, tmp_path, env, named, not_named,
    ) -> None:
        """Never the local fallback: nothing is built and nothing is written."""
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(ValueError) as raised:
            AgentMemory.from_env("agent-a")
        message = str(raised.value)
        for name in named:
            assert name in message
        for name in not_named:
            assert name not in message
        assert not (tmp_path / ".amfs").exists()

    def test_a_missing_adapter_package_raises_and_says_what_to_install(
        self, monkeypatch, tmp_path,
    ) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        real_import = builtins.__import__

        def _no_http_adapter(name, *args, **kwargs):
            if name == "amfs_adapter_http":
                raise ImportError("No module named 'amfs_adapter_http'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_http_adapter)
        with pytest.raises(ImportError) as raised:
            AgentMemory.from_env("agent-a")
        assert "pip install amfs-adapter-http" in str(raised.value)
        assert not (tmp_path / ".amfs").exists()

    def test_an_adapter_argument_is_refused(self) -> None:
        with pytest.raises(TypeError):
            AgentMemory.from_env("agent-a", base_url=URL, api_key=KEY, adapter=object())

    def test_it_does_not_warn(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory.from_env("agent-a")
        assert _warnings(caplog) == []


class TestConstructorKeepsItsDefault:
    def test_the_environment_does_not_change_the_adapter(self, monkeypatch) -> None:
        """The guarantee the warning exists to keep: an inherited
        ``AMFS_HTTP_URL`` never moves a bare ``AgentMemory`` onto the hosted API."""
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        mem = AgentMemory("agent-a")
        assert type(mem.adapter).__name__ == "FilesystemAdapter"

    def test_it_warns_once_and_names_the_way_to_connect(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory("agent-a")
            AgentMemory("agent-b")
        messages = _warnings(caplog)
        assert len(messages) == 1
        assert "AMFS_HTTP_URL" in messages[0]
        assert "from_env" in messages[0]
        assert "HttpAdapter" in messages[0]

    def test_the_warning_never_carries_the_url_or_the_key(self, monkeypatch, caplog) -> None:
        """Log lines travel further than environments do."""
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        monkeypatch.setenv("AMFS_API_KEY", KEY)
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory("agent-a")
        (message,) = _warnings(caplog)
        assert URL not in message
        assert KEY not in message

    def test_no_warning_without_the_variable(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory("agent-a")
        assert _warnings(caplog) == []

    def test_no_warning_when_the_variable_is_blank(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("AMFS_HTTP_URL", "  ")
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory("agent-a")
        assert _warnings(caplog) == []

    def test_no_warning_when_an_adapter_is_passed(self, monkeypatch, caplog, tmp_path) -> None:
        from amfs_filesystem import FilesystemAdapter

        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            AgentMemory("agent-a", adapter=FilesystemAdapter(root=tmp_path, namespace="default"))
        assert _warnings(caplog) == []

    def test_no_warning_when_a_config_file_is_named(self, monkeypatch, caplog, tmp_path) -> None:
        """Naming a config is choosing local storage on purpose."""
        config = tmp_path / "amfs.yaml"
        config.write_text(
            "namespace: default\n"
            "layers:\n"
            "  primary:\n"
            "    adapter: filesystem\n"
            "    options:\n"
            f"      root: {tmp_path / 'store'}\n"
        )
        monkeypatch.setenv("AMFS_HTTP_URL", URL)
        with caplog.at_level(logging.WARNING, logger="amfs.memory"):
            mem = AgentMemory("agent-a", config_path=config)
        assert type(mem.adapter).__name__ == "FilesystemAdapter"
        assert _warnings(caplog) == []
