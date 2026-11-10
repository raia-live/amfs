"""The dashboard proxy secret is compared in constant time.

``apply_tenant_headers_from_request`` decides whether a request may act as a
dashboard account. It compared the shared secret with ``!=``, which returns as
soon as two strings differ, so the time it takes leaks how much of a guess
was right. ``hmac.compare_digest`` takes the same time whatever the guess.
"""

from __future__ import annotations

import hmac
from uuid import uuid4

import pytest
from amfs_http import tenant_middleware
from starlette.requests import Request

SECRET = "dashboard-proxy-secret"


def _request(secret: str | None) -> Request:
    headers = [(b"x-amfs-dashboard-account-id", str(uuid4()).encode())]
    if secret is not None:
        headers.append((b"x-amfs-dashboard-secret", secret.encode()))
    return Request({"type": "http", "headers": headers, "state": {}})


@pytest.fixture
def secret_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMFS_DASHBOARD_PROXY_SECRET", SECRET)


@pytest.fixture(autouse=True)
def _clear_tls() -> None:
    yield
    tenant_middleware.clear_tenant_headers()


def test_the_right_secret_is_accepted(secret_set: None) -> None:
    assert tenant_middleware.apply_tenant_headers_from_request(_request(SECRET)) is True


@pytest.mark.parametrize("provided", ["wrong-secret", "dashboard-proxy-secre", ""])
def test_a_wrong_secret_is_refused(secret_set: None, provided: str) -> None:
    assert tenant_middleware.apply_tenant_headers_from_request(_request(provided)) is False


def test_a_missing_secret_header_is_refused(secret_set: None) -> None:
    assert tenant_middleware.apply_tenant_headers_from_request(_request(None)) is False


def test_the_secret_is_compared_with_compare_digest(
    secret_set: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)

    tenant_middleware.apply_tenant_headers_from_request(_request("guess"))

    assert calls == [(b"guess", SECRET.encode())]
