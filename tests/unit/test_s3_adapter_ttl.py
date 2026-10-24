"""An S3 entry past its ``ttl_at`` reads as missing, as on the other adapters.

The bucket is an in-memory stand-in for the client calls the adapter makes
on a read and a write, so these run without network access or credentials.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from amfs_core.engine import CausalTagger
from amfs_core.models import MemoryEntry
from amfs_s3 import adapter as s3_adapter
from amfs_s3.adapter import S3Adapter


class _FakeBucket:
    """Just enough of a boto3 S3 client for S3Adapter."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_paginator(self, name: str) -> Any:
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket: str, Prefix: str) -> Any:  # noqa: N803
                keys = sorted(k for k in objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]} if keys else {}

        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:  # noqa: N803
        self.objects[Key] = Body

    def copy_object(self, *, Bucket: str, CopySource: dict[str, str], Key: str) -> None:  # noqa: N803
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects.pop(Key, None)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> S3Adapter:
    bucket = _FakeBucket()
    monkeypatch.setattr(s3_adapter.boto3, "client", lambda *a, **kw: bucket)
    return S3Adapter("amfs-test", namespace="test")


def _entry(ttl_at: datetime) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/checkout",
        key="retry",
        value={"max": 3},
        provenance=CausalTagger("agent-a").tag(),
        ttl_at=ttl_at,
    )


def test_an_expired_entry_reads_as_missing(adapter: S3Adapter) -> None:
    adapter.write(_entry(datetime.now(timezone.utc) - timedelta(hours=1)))

    assert adapter.read("svc/checkout", "retry") is None


def test_an_entry_whose_ttl_has_not_elapsed_still_reads(adapter: S3Adapter) -> None:
    adapter.write(_entry(datetime.now(timezone.utc) + timedelta(hours=1)))

    entry = adapter.read("svc/checkout", "retry")
    assert entry is not None
    assert entry.value == {"max": 3}
