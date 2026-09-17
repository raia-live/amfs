import json

import httpx
import pytest

from amfs import DecisionClient


def test_training_retries_and_activation_are_explicit():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(202, json={"state": "queued"})
    with DecisionClient("https://example.test", "test", transport=httpx.MockTransport(handler)) as client:
        for _ in range(2):
            client.enqueue_training("test:model", version="v1", dataset_id="approved", idempotency_key="same-key")
        client.activate_version("test:model", "v1", expected_active_version=None)
        client.set_mode("test:model", "observe")
        with pytest.raises(TypeError):
            client.activate_version("test:model", "v1")
    assert seen[0].headers["Idempotency-Key"] == seen[1].headers["Idempotency-Key"] == "same-key"
    assert seen[0].content == seen[1].content
    assert json.loads(seen[2].content) == {"expected_active_version": None}
    assert seen[3].method == "PATCH"


def test_history_keeps_opaque_cursor_and_propagates_conflict():
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path.endswith("activate"):
            return httpx.Response(409, json={"detail": "stale version"})
        return httpx.Response(200, json={"decisions": [], "next_cursor": None})
    with DecisionClient("https://example.test", "test", transport=httpx.MockTransport(handler)) as client:
        client.history("example", cursor="opaque+/=&x", limit=10)
        client.usage("example", days=7)
        with pytest.raises(httpx.HTTPStatusError) as error:
            client.activate_version("example", "v2", expected_active_version="v1")
        assert error.value.response.status_code == 409
        with pytest.raises(ValueError):
            client.history("example", limit=101)
        with pytest.raises(ValueError):
            client.enqueue_training("example", version="v1", dataset_id="approved", idempotency_key="")
    assert seen[0].url.params["cursor"] == "opaque+/=&x"
    assert seen[0].url.params["limit"] == "10"
    assert seen[1].url.params["days"] == "7"
