"""S3Adapter — AMFS adapter backed by S3-compatible object storage.

Object layout mirrors the filesystem adapter:
    s3://{bucket}/{prefix}/{namespace}/{entity_path}/{key}/v{NNN}_current.json
    s3://{bucket}/{prefix}/{namespace}/{entity_path}/{key}/v{NNN}_superseded.json

CoW semantics:
    1. List objects in the key prefix to find current version
    2. Copy current to superseded
    3. PUT new version as current
    4. DELETE old current

Supports any S3-compatible provider: AWS S3, ACS, MinIO, Cloudflare R2, etc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

import boto3
from botocore.exceptions import ClientError

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.exceptions import VersionConflictError
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
    clamp_confidence,
)

logger = logging.getLogger(__name__)

VERSION_RE = re.compile(r"v(\d+)_(current|superseded)\.json$")


class S3Adapter(AdapterABC):
    """Store AMFS entries as versioned JSON objects in S3-compatible storage.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    prefix:
        Optional key prefix (e.g. "amfs/").
    namespace:
        Logical namespace (default: "default").
    endpoint_url:
        Custom S3 endpoint (for ACS, MinIO, R2, etc.). None for AWS S3.
    region_name:
        AWS region name.
    aws_access_key_id:
        Access key (can also use env vars or IAM roles).
    aws_secret_access_key:
        Secret key.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        namespace: str = "default",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._namespace = namespace

        client_kwargs: dict[str, Any] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if region_name:
            client_kwargs["region_name"] = region_name
        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._s3 = boto3.client("s3", **client_kwargs)
        self._watchers: dict[str, list[Callable[[MemoryEntry], None]]] = {}

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key_prefix(self, entity_path: str, key: str) -> str:
        """S3 key prefix for a given entity_path/key."""
        return f"{self._prefix}{self._namespace}/{entity_path}/{key}/"

    def _ns_prefix(self) -> str:
        """S3 key prefix for the namespace."""
        return f"{self._prefix}{self._namespace}/"

    def _entity_prefix(self, entity_path: str) -> str:
        """S3 key prefix for an entity."""
        return f"{self._prefix}{self._namespace}/{entity_path}/"

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List all objects under a prefix."""
        objects: list[dict[str, Any]] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            objects.extend(page.get("Contents", []))
        return objects

    def _find_current_version(self, entity_path: str, key: str) -> tuple[int, str | None]:
        """Find the current version number and S3 key for an entry."""
        prefix = self._key_prefix(entity_path, key)
        objects = self._list_objects(prefix)

        current_version = 0
        current_key = None
        for obj in objects:
            s3_key = obj["Key"]
            filename = s3_key.rsplit("/", 1)[-1]
            match = VERSION_RE.match(filename)
            if match and match.group(2) == "current":
                ver = int(match.group(1))
                if ver > current_version:
                    current_version = ver
                    current_key = s3_key
        return current_version, current_key

    def _read_object(self, s3_key: str) -> MemoryEntry:
        """Read and parse a MemoryEntry from an S3 object."""
        response = self._s3.get_object(Bucket=self._bucket, Key=s3_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return MemoryEntry.model_validate(data)

    # ------------------------------------------------------------------
    # AdapterABC implementation
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        _, current_key = self._find_current_version(entity_path, key)
        if current_key is None:
            return None
        try:
            entry = self._read_object(current_key)
        except Exception:
            logger.warning("Failed to read %s", current_key)
            return None
        if entry.confidence < min_confidence:
            return None
        return entry

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        current_version, current_key = self._find_current_version(
            entry.entity_path, entry.key
        )
        new_version = current_version + 1

        if entry.version > 1 and entry.version != new_version:
            raise VersionConflictError(
                entry.entity_path, entry.key, entry.version, current_version
            )

        entry = entry.model_copy(update={"version": new_version})

        # Supersede old current by copying to superseded then deleting
        if current_key is not None:
            superseded_key = current_key.replace("_current.json", "_superseded.json")
            self._s3.copy_object(
                Bucket=self._bucket,
                CopySource={"Bucket": self._bucket, "Key": current_key},
                Key=superseded_key,
            )
            self._s3.delete_object(Bucket=self._bucket, Key=current_key)

        # Write new current version
        prefix = self._key_prefix(entry.entity_path, entry.key)
        new_key = f"{prefix}v{new_version:03d}_current.json"
        data = entry.model_dump(mode="json")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=new_key,
            Body=json.dumps(data, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )

        for watch_ep, callbacks in self._watchers.items():
            if entry.entity_path == watch_ep or entry.entity_path.startswith(watch_ep + "/"):
                for cb in callbacks:
                    try:
                        cb(entry)
                    except Exception:
                        logger.debug("Watcher callback error", exc_info=True)

        return entry

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        prefix = self._entity_prefix(entity_path) if entity_path else self._ns_prefix()
        objects = self._list_objects(prefix)

        entries: list[MemoryEntry] = []
        for obj in objects:
            s3_key = obj["Key"]
            filename = s3_key.rsplit("/", 1)[-1]
            match = VERSION_RE.match(filename)
            if not match:
                continue
            status = match.group(2)
            if not include_superseded and status == "superseded":
                continue
            try:
                entries.append(self._read_object(s3_key))
            except Exception:
                logger.warning("Skipping unreadable object: %s", s3_key)
        return entries

    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        if entity_path not in self._watchers:
            self._watchers[entity_path] = []
        self._watchers[entity_path].append(callback)

        def cancel() -> None:
            cbs = self._watchers.get(entity_path, [])
            if callback in cbs:
                cbs.remove(callback)

        return WatchHandle(cancel)

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        multiplier = OUTCOME_MULTIPLIERS[record.outcome_type]
        updated: list[MemoryEntry] = []

        for entry_key_spec in record.causal_entry_keys:
            parts = entry_key_spec.rsplit("/", 1)
            if len(parts) != 2:
                logger.warning("Invalid causal_entry_key format: %s", entry_key_spec)
                continue
            entity_path, key = parts

            current = self.read(entity_path, key)
            if current is None:
                logger.warning("Causal entry not found: %s", entry_key_spec)
                continue

            new_confidence = clamp_confidence(
                current.confidence * multiplier * record.causal_confidence
            )
            new_entry = current.model_copy(
                update={
                    "version": 1,
                    "confidence": new_confidence,
                    "outcome_count": current.outcome_count + 1,
                }
            )
            written = self.write(new_entry)
            updated.append(written)

        return updated

    def close(self) -> None:
        """Clean up resources."""
        self._watchers.clear()
