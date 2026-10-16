"""The S3 adapter accepts the calls the engine actually makes.

CoWEngine passes ``branch=`` to ``adapter.read`` and ``adapter.list`` on every
read, write, list and history call. S3Adapter's signatures predate that
keyword, so each of those raised TypeError before S3 was reached, and an
``AgentMemory`` on S3 could not write a single entry. Nothing noticed because
no test ever drove the adapter through the engine.

The bucket is an in-memory stand-in for the five client calls the adapter
makes, so these run without network access or credentials.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from amfs import AgentMemory
from amfs_core.engine import CausalTagger, CoWEngine
from amfs_s3 import adapter as s3_adapter
from amfs_s3.adapter import S3Adapter


class _FakeBucket:
    """Just enough of a boto3 S3 client for S3Adapter."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2"
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


def test_the_engine_can_write_and_read_back(adapter: S3Adapter) -> None:
    engine = CoWEngine(adapter, CausalTagger("agent-a"))

    engine.write("svc/checkout", "retry", {"max": 3})
    engine.write("svc/checkout", "retry", {"max": 5})

    entry = engine.read("svc/checkout", "retry")
    assert entry is not None
    assert entry.version == 2
    assert entry.value == {"max": 5}


def test_list_and_history_go_through_the_engine(adapter: S3Adapter) -> None:
    engine = CoWEngine(adapter, CausalTagger("agent-a"))
    engine.write("svc/checkout", "retry", {"max": 3})
    engine.write("svc/checkout", "retry", {"max": 5})

    assert [(e.key, e.version) for e in engine.list("svc/checkout")] == [("retry", 2)]
    assert [e.value for e in engine.history("svc/checkout", "retry")] == [
        {"max": 3},
        {"max": 5},
    ]


def test_an_agent_memory_on_s3_round_trips(adapter: S3Adapter) -> None:
    mem = AgentMemory(agent_id="agent-a", adapter=adapter)

    mem.write("svc/checkout", "retry", {"max": 3})
    entry = mem.read("svc/checkout", "retry")

    assert entry is not None
    assert entry.value == {"max": 3}
    assert [e.key for e in mem.list("svc/checkout")] == ["retry"]


def test_branch_is_accepted_and_reads_the_one_line_of_history(adapter: S3Adapter) -> None:
    CoWEngine(adapter, CausalTagger("agent-a")).write("svc/checkout", "retry", {"max": 3})

    entry = adapter.read("svc/checkout", "retry", branch="main")
    assert entry is not None
    assert entry.value == {"max": 3}
    assert [e.key for e in adapter.list("svc/checkout", branch="main")] == ["retry"]
