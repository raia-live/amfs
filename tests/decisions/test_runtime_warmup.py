import json

import pytest

torch = pytest.importorskip("torch")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from amfs_decision_runtime import DecisionScorer, ScorerConfig
from amfs_decision_runtime.model import artifact_digest
from amfs_decision_runtime import server


def setup(tmp_path, monkeypatch, capacity=2):
    registry = {}
    for name in ("tenant:model-a:v1", "tenant:model-b:v1"):
        directory = tmp_path / name.replace(":", "-")
        DecisionScorer(ScorerConfig(hidden_size=16, max_state_tokens=32, max_candidate_tokens=32)).save(directory)
        registry[name] = {"directory": str(directory), "sha256": artifact_digest(directory), "version": "experimental-v1"}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    monkeypatch.setenv("AMFS_DECISION_ARTIFACTS", str(path))
    monkeypatch.setenv("AMFS_DECISION_CACHE_MODELS", str(capacity))
    app = FastAPI(lifespan=server.lifespan)
    return registry, app


def test_default_remains_lazy_and_explicit_warmup_is_ready_as_complete_set(tmp_path, monkeypatch):
    registry, app = setup(tmp_path, monkeypatch)
    monkeypatch.delenv("AMFS_DECISION_WARMUP_KEYS", raising=False)
    with TestClient(app):
        assert list(app.state.pool.cache) == []
    assert not hasattr(app.state, "pool")
    monkeypatch.setenv("AMFS_DECISION_WARMUP_KEYS", json.dumps(list(registry)))
    with TestClient(app):
        assert list(app.state.pool.cache) == list(registry)
        for key in registry:
            assert app.state.pool.get(key) is app.state.pool.cache[key]
    assert not hasattr(app.state, "pool")


@pytest.mark.parametrize("keys", ["not-json", "{}", "null", '[1]', '[""]', '["unknown"]',
    '["tenant:model-a:v1","tenant:model-a:v1"]',
    '["tenant:model-a:v1","tenant:model-b:v1","other"]'])
def test_invalid_operator_list_fails_before_loading_any_model(tmp_path, monkeypatch, keys):
    _, app = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(server.DecisionScorer, "load", lambda path: calls.append(path))
    monkeypatch.setenv("AMFS_DECISION_WARMUP_KEYS", keys)
    with pytest.raises(ValueError):
        with TestClient(app):
            pytest.fail("startup must not become ready")
    assert calls == []
    assert not hasattr(app.state, "pool")


def test_over_capacity_with_registered_keys_is_rejected(tmp_path, monkeypatch):
    registry, app = setup(tmp_path, monkeypatch, capacity=1)
    monkeypatch.setenv("AMFS_DECISION_WARMUP_KEYS", json.dumps(list(registry)))
    with pytest.raises(ValueError, match="capacity"):
        with TestClient(app):
            pytest.fail("startup must fail")
    assert not hasattr(app.state, "pool")


def test_corrupt_second_artifact_never_publishes_partial_ready_pool(tmp_path, monkeypatch):
    registry, app = setup(tmp_path, monkeypatch)
    keys = list(registry)
    config = tmp_path / keys[1].replace(":", "-") / "config.json"
    config.write_text(config.read_text()+"\n")
    monkeypatch.setenv("AMFS_DECISION_WARMUP_KEYS", json.dumps(keys))
    loaded = []
    pools = []
    original_get = server.ArtifactPool.get
    def observed_get(pool, key):
        # First model is loaded successfully; app readiness remains unpublished
        # even while the second artifact is being checked.
        assert not hasattr(app.state, "pool")
        pools.append(pool)
        result = original_get(pool, key)
        loaded.append(key)
        return result
    monkeypatch.setattr(server.ArtifactPool, "get", observed_get)
    app.state.pool = object()  # stale state from a prior lifecycle
    with pytest.raises(HTTPException) as error:
        with TestClient(app):
            pytest.fail("startup must fail on corrupt artifact")
    assert error.value.status_code == 503
    assert loaded == [keys[0]]
    assert not hasattr(app.state, "pool")
    assert all(not pool.cache for pool in pools)


def test_device_allocation_failure_aborts_readiness_and_clears_cache(tmp_path, monkeypatch):
    registry, app = setup(tmp_path, monkeypatch)
    monkeypatch.setenv("AMFS_DECISION_WARMUP_KEYS", json.dumps(list(registry)))
    calls = []
    def load_failure(pool, key):
        calls.append(key)
        if len(calls) == 1:
            pool.cache[key] = object()
            return pool.cache[key]
        raise RuntimeError("simulated device out of memory")
    seen = []
    original_close = server.ArtifactPool.close
    def close(pool):
        original_close(pool)
        seen.append(len(pool.cache))
    monkeypatch.setattr(server.ArtifactPool, "get", load_failure)
    monkeypatch.setattr(server.ArtifactPool, "close", close)
    with pytest.raises(RuntimeError, match="out of memory"):
        with TestClient(app):
            pytest.fail("startup must fail")
    assert seen == [0]
    assert not hasattr(app.state, "pool")
