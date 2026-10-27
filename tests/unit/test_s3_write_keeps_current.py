"""A failed S3 write must not take the current version of a key with it.

``write`` deleted the current object before it put the new one. A put that
failed after the delete left the key with no current object: the previous
value was still in the bucket as a superseded copy, but ``read`` returned
``None``. The new version is now put first, and the old one is superseded
only once it is safely in place.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_s3 import adapter as s3_adapter
from amfs_s3.adapter import S3Adapter


class _FakeBucket:
    """Just enough of a boto3 S3 client for S3Adapter, with a put that can fail."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_puts = False

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
        if self.fail_puts:
            raise OSError("connection reset (simulated)")
        self.objects[Key] = Body

    def copy_object(self, *, Bucket: str, CopySource: dict[str, str], Key: str) -> None:  # noqa: N803
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects.pop(Key, None)


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch) -> _FakeBucket:
    fake = _FakeBucket()
    monkeypatch.setattr(s3_adapter.boto3, "client", lambda *a, **kw: fake)
    return fake


@pytest.fixture
def adapter(bucket: _FakeBucket) -> S3Adapter:
    return S3Adapter("amfs-test", namespace="test")


def _entry(value: str) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/checkout",
        key="retry",
        value=value,
        provenance=Provenance(
            agent_id="ag", session_id="s", written_at=datetime.now(UTC)
        ),
        confidence=1.0,
    )


def test_a_failed_put_leaves_the_previous_version_current(
    adapter: S3Adapter, bucket: _FakeBucket
) -> None:
    adapter.write(_entry("old"))
    bucket.fail_puts = True

    with pytest.raises(OSError):
        adapter.write(_entry("new"))

    current = adapter.read("svc/checkout", "retry")
    assert current is not None
    assert current.value == "old"


def test_a_good_write_leaves_one_current_object(adapter: S3Adapter) -> None:
    adapter.write(_entry("old"))
    adapter.write(_entry("new"))

    assert [(e.version, e.value) for e in adapter.list("svc/checkout")] == [(2, "new")]


def test_a_write_clears_a_current_object_left_by_an_interrupted_write(
    adapter: S3Adapter, bucket: _FakeBucket
) -> None:
    adapter.write(_entry("old"))
    # What a write that stopped after putting v002 but before superseding v001
    # leaves in the bucket: two current objects for the key.
    bucket.objects["test/svc/checkout/retry/v002_current.json"] = bucket.objects[
        "test/svc/checkout/retry/v001_current.json"
    ]

    adapter.write(_entry("newest"))

    assert [(e.version, e.value) for e in adapter.list("svc/checkout")] == [(3, "newest")]
