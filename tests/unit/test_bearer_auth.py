"""The REST API accepts its key from X-AMFS-API-Key *or* Authorization: Bearer.

X-AMFS-API-Key is the canonical header and every existing SDK/HttpAdapter client
keeps working unchanged. Bearer is an additive fallback: credential-brokering
gateways (e.g. Fly.io Sprites connectors) store the key once and inject it as
``Authorization: Bearer <amfs_key>``, and cannot set a custom header. These tests
pin that both sources authenticate, that the custom header wins when both are
present, and that a bad/absent credential still 401s.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import HTTPException  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials  # noqa: E402

from amfs_http.auth import verify_api_key  # noqa: E402

KEY = "amfs_testkey_123"


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _call(**kwargs) -> str | None:
    return asyncio.run(verify_api_key(**kwargs))


@pytest.fixture()
def keyed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enforce a known env key and no DB, so only the header logic is exercised."""
    monkeypatch.setenv("AMFS_API_KEYS", KEY)
    monkeypatch.delenv("AMFS_POSTGRES_DSN", raising=False)


class TestBearerFallback:
    def test_custom_header_still_authenticates(self, keyed_env) -> None:
        assert _call(api_key=KEY, bearer=None) == KEY

    def test_bearer_authenticates(self, keyed_env) -> None:
        assert _call(api_key=None, bearer=_bearer(KEY)) == KEY

    def test_custom_header_wins_over_bearer(self, keyed_env) -> None:
        # A valid X-AMFS-API-Key is used even if a (wrong) bearer is also present.
        assert _call(api_key=KEY, bearer=_bearer("wrong")) == KEY

    def test_bad_bearer_is_rejected(self, keyed_env) -> None:
        with pytest.raises(HTTPException) as exc:
            _call(api_key=None, bearer=_bearer("amfs_not_a_real_key"))
        assert exc.value.status_code == 401

    def test_no_credential_is_rejected(self, keyed_env) -> None:
        with pytest.raises(HTTPException) as exc:
            _call(api_key=None, bearer=None)
        assert exc.value.status_code == 401


class TestOpenModeUnaffected:
    def test_bearer_does_not_break_open_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No env keys and no DB DSN => open mode returns None regardless of headers.
        monkeypatch.delenv("AMFS_API_KEYS", raising=False)
        monkeypatch.delenv("AMFS_POSTGRES_DSN", raising=False)
        assert _call(api_key=None, bearer=_bearer(KEY)) is None
        assert _call(api_key=None, bearer=None) is None
