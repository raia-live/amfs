"""Reference bounded multi-artifact runtime behind authenticated ingress.

AMFS_DECISION_ARTIFACTS names an operator-written registry. This service never accepts
customer-supplied paths, downloads adapters, or executes recommended tools.
"""
from __future__ import annotations

import os
import json
import threading
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .model import DecisionScorer, artifact_digest


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_key: str = Field(min_length=1, max_length=256)
    state: str = Field(max_length=128_000)
    questions: dict[str, list[str]] = Field(min_length=1, max_length=16)


def warmup_keys(value: str, registry: dict, capacity: int) -> list[str]:
    """Operator-only JSON configuration, never supplied by a scoring request."""
    keys = json.loads(value)
    if not isinstance(keys, list) or any(not isinstance(key, str) or not 1 <= len(key) <= 256 for key in keys):
        raise ValueError("AMFS_DECISION_WARMUP_KEYS must be a JSON array of runtime keys")
    if len(keys) > capacity:
        raise ValueError("startup warmup exceeds the configured model cache capacity")
    if len(set(keys)) != len(keys):
        raise ValueError("startup warmup contains duplicate runtime keys")
    if any(key not in registry for key in keys):
        raise ValueError("startup warmup contains an unregistered runtime key")
    return keys


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A prior lifespan cannot leave a stale ready flag after restart failure.
    if hasattr(app.state, "pool"):
        del app.state.pool
    pool = ArtifactPool(json.loads(Path(os.environ["AMFS_DECISION_ARTIFACTS"]).read_text()),
                        int(os.environ.get("AMFS_DECISION_CACHE_MODELS", "2")))
    try:
        keys = warmup_keys(os.environ.get("AMFS_DECISION_WARMUP_KEYS", "[]"), pool.registry, pool.capacity)
        # Construct privately and publish only after EVERY requested digest,
        # version and device load succeeds. Failure aborts startup; no partial
        # model set is advertised ready. No warmup is performed by default.
        await run_in_threadpool(pool.warmup, keys)
        app.state.pool = pool
        yield
    finally:
        if getattr(app.state, "pool", None) is pool:
            del app.state.pool
        pool.close()


class ArtifactPool:
    """Bounded LRU; never swaps adapter state on an in-flight request.

    This reference serializes GPU inference and cache changes per instance.
    Scale replicas for throughput; production batching can replace this lock.
    Registry entries pin a complete artifact digest and immutable version. Requests supply
    an opaque registered key, never a filesystem or cloud storage path.
    """
    def __init__(self, registry: dict, capacity: int):
        if capacity < 1 or not registry:
            raise ValueError("artifact pool requires registered artifacts and positive capacity")
        self.registry, self.capacity = registry, capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def warmup(self, keys: list[str]):
        # Validate before the first allocation, including direct operator use.
        keys = warmup_keys(json.dumps(keys), self.registry, self.capacity)
        with self.lock, torch.inference_mode():
            for key in keys:
                self.get(key)

    def close(self):
        with self.lock:
            self.cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get(self, key: str):
        entry = self.registry.get(key)
        if entry is None:
            raise HTTPException(404, "artifact not registered")
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        directory = Path(entry["directory"])
        if artifact_digest(directory) != entry["sha256"]:
            raise HTTPException(503, "artifact integrity check failed")
        if len(self.cache) >= self.capacity:
            _, old = self.cache.popitem(last=False)
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model = DecisionScorer.load(directory)
        if model.config.version != entry["version"]:
            raise HTTPException(503, "artifact version mismatch")
        if torch.cuda.is_available():
            model.to("cuda")
        self.cache[key] = model
        return model


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": hasattr(app.state, "pool")}


def score_batch(body: ScoreRequest):
    sizes = [len(v) for v in body.questions.values()]
    if any(n < 2 or n > 256 for n in sizes) or sum(sizes) > 1024:
        raise HTTPException(422, "candidate count exceeds runtime limits")
    queries = [q for values in body.questions.values() for q in values]
    if any(len(q) > 8192 for q in queries):
        raise HTTPException(422, "candidate query exceeds runtime limit")
    try:
        with app.state.pool.lock, torch.inference_mode():
            scorer = app.state.pool.get(body.runtime_key)
            logits = scorer(body.state, queries)["logits"]
            result, start = {}, 0
            for name, size in zip(body.questions, sizes):
                result[name] = logits[start:start + size].softmax(-1).cpu().tolist()
                start += size
            version = scorer.config.version
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"distributions": result, "model_version": version, "runtime_key": body.runtime_key,
            "artifact_sha256": app.state.pool.registry[body.runtime_key]["sha256"]}


@app.post("/score")
async def score(body: ScoreRequest):
    return await run_in_threadpool(score_batch, body)
