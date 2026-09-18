"""An empty shared pool is healthy infrastructure, never a scoring fallback."""
import json
import pytest
pytest.importorskip('torch')
from fastapi.testclient import TestClient
from amfs_decision_runtime import server


def test_empty_registry_starts_without_loading_and_unknown_keys_stay_404(tmp_path, monkeypatch):
    path = tmp_path / 'registry.json'
    path.write_text('{}')
    monkeypatch.setenv('AMFS_DECISION_ARTIFACTS', str(path))
    monkeypatch.setenv('AMFS_DECISION_CACHE_MODELS', '2')
    monkeypatch.delenv('AMFS_DECISION_WARMUP_KEYS', raising=False)
    def forbidden(*args):
        pytest.fail('empty registry must not load or digest any artifact')
    monkeypatch.setattr(server.DecisionScorer, 'load', forbidden)
    monkeypatch.setattr(server, 'artifact_digest', forbidden)
    with TestClient(server.app) as client:
        assert client.get('/health').json() == {'ready': True, 'registered_models': 0, 'loaded_models': 0}
        for key in ('tenant:model:v1', '/tmp/customer-path', 'gs://untrusted/model'):
            response = client.post('/score', json={'runtime_key': key, 'state': 'failure',
                                                   'questions': {'action': ['retry', 'defer']}})
            assert response.status_code == 404
            assert response.json()['detail'] == 'artifact not registered'
    assert server.health()['ready'] is False


@pytest.mark.parametrize('registry', [None, [], '', False, {'key': None}, {'key': {}},
    {'key': {'directory': '/tmp/artifact', 'version': 'v1', 'sha256': 'bad'}}])
def test_malformed_registry_never_becomes_ready(tmp_path, monkeypatch, registry):
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(registry))
    monkeypatch.setenv('AMFS_DECISION_ARTIFACTS', str(path))
    monkeypatch.delenv('AMFS_DECISION_WARMUP_KEYS', raising=False)
    with pytest.raises(ValueError):
        with TestClient(server.app):
            pytest.fail('malformed registry became ready')
    assert server.health()['ready'] is False


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5, '2'])
def test_empty_registry_requires_positive_integer_capacity(capacity):
    with pytest.raises(ValueError, match='capacity'):
        server.ArtifactPool({}, capacity)


def test_empty_registry_does_not_allow_unregistered_warmup():
    pool = server.ArtifactPool({}, 2)
    with pytest.raises(ValueError, match='unregistered'):
        pool.warmup(['unregistered'])
    assert not pool.cache
