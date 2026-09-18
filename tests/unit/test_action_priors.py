"""Action-level learning: priors from the outcome record and the recommendation.

Pure-function tests over ``amfs_core.actions``, plus the derivation the SDK
performs at commit and the server's ``/api/v1/retrieve`` trailing ``_meta``
element over an adapter stubbed to return outcome rows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from amfs import AgentMemory
from amfs_core import actions as act
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter

# ── action_key ────────────────────────────────────────────────────────────


def test_action_key_prefers_the_action_argument() -> None:
    assert act.action_key("resolve", {"reply": "hello there", "action": "resend_email"}) == "resolve:resend_email"


def test_action_key_falls_back_to_the_first_short_argument_by_name() -> None:
    assert act.action_key("lookup", {"zone": "eu-west", "account": "acc_1"}) == "lookup:acc_1"


def test_action_key_never_uses_free_text() -> None:
    assert act.action_key("reply", {"body": "a long free text with spaces in it"}) == "reply"
    assert act.action_key("reply", {"body": "x" * 41}) == "reply"


def test_action_key_explicit_wins() -> None:
    assert act.action_key("resolve", {"action": "a"}, explicit="resolve:b") == "resolve:b"


# ── actions_taken ─────────────────────────────────────────────────────────

_CALLS = [
    {"tool_name": "lookup_account", "arguments": {"email": "x@y"}},
    {"tool_name": "resolve", "arguments": {"action": "update_payment_method"}},
    {"tool_name": "lookup_account", "arguments": {"email": "x@y"}},
    {"tool_name": "resolve", "arguments": {"action": "resend_email"}},
]


def test_actions_taken_labels_the_decisive_action_of_each_attempt_and_the_terminal_one() -> None:
    rows = act.actions_taken(_CALLS, [{"attempt": 1, "action_indices": [0, 1]}], 3, "success")
    assert [(r["action_key"], r["success"], r["attempt"]) for r in rows] == [
        ("resolve:update_payment_method", False, 1),
        ("resolve:resend_email", True, None),
    ]
    # The lookups are observations, not decisions: not labeled.
    assert all(r["tool_name"] == "resolve" for r in rows)


def test_actions_taken_terminal_failure_is_a_loss() -> None:
    rows = act.actions_taken(_CALLS, [], 1, "failure")
    assert rows == [{"index": 1, "action_key": "resolve:update_payment_method",
                     "tool_name": "resolve", "success": False, "attempt": None}]


def test_actions_taken_is_defensive_about_indices() -> None:
    assert act.actions_taken(_CALLS, [{"attempt": 1, "action_indices": [99]}], 42, "success") == []
    assert act.actions_taken([], [], None, "success") == []


def test_entity_paths_of_dedupes_and_keeps_the_explicit_first() -> None:
    assert act.entity_paths_of(["acme/support/fix-a", "acme/support/fix-b", "acme/billing/x"], "acme/ops") == [
        "acme/ops", "acme/support", "acme/billing",
    ]


# ── aggregate_priors ───────────────────────────────────────────────────────


def _row(actions, *, agent="a1", days_ago=0, sim=1.0):
    return {
        "actions_taken": [{"action_key": k, "success": ok} for k, ok in actions],
        "committed_at": datetime.now(UTC) - timedelta(days=days_ago),
        "agent_id": agent,
        "similarity": sim,
    }


def test_priors_count_wins_and_losses_per_action_and_name_the_untried() -> None:
    rows = [
        _row([("resolve:a", False), ("resolve:b", True)], agent="a1"),
        _row([("resolve:a", False), ("resolve:b", True)], agent="a2"),
        _row([("resolve:a", False)], agent="a3"),
    ]
    pr = act.aggregate_priors(rows, candidate_actions=["resolve:a", "resolve:b", "resolve:c"])
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["resolve:b"]["won"] == 2 and by["resolve:b"]["n"] == 2 and by["resolve:b"]["agents"] == 2
    assert by["resolve:a"]["lost"] == 3 and by["resolve:a"]["agents"] == 3
    assert pr["tried"][0]["action_key"] == "resolve:b"  # winners first
    assert pr["untried"] == ["resolve:c"]
    assert pr["n_outcomes"] == 3


def test_priors_decay_old_outcomes_but_still_count_them() -> None:
    fresh = act.aggregate_priors([_row([("x", True)], days_ago=0)])["tried"][0]
    old = act.aggregate_priors([_row([("x", True)], days_ago=60)])["tried"][0]
    assert fresh["won"] == old["won"] == 1
    assert fresh["p"] > old["p"]  # the old win carries less evidence mass


# ── recommend ─────────────────────────────────────────────────────────────

_LOSERS = {
    "tried": [
        {"action_key": "resolve:a", "won": 0, "lost": 4, "p": 0.17, "n": 4, "agents": 2},
        {"action_key": "resolve:b", "won": 0, "lost": 3, "p": 0.2, "n": 3, "agents": 1},
    ],
    "untried": ["resolve:c", "resolve:d", "resolve:e"],
}


def test_recommend_acts_on_a_winning_action() -> None:
    pr = {"tried": [{"action_key": "resolve:b", "won": 3, "lost": 0, "p": 0.8, "n": 3, "agents": 1}], "untried": []}
    rec = act.recommend(pr, agent_id="x")
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:b"


def test_recommend_explores_when_everything_tried_failed_and_spreads_agents() -> None:
    picks = {a: act.recommend(_LOSERS, agent_id=a)["suggested_action"] for a in (f"agent-{i}" for i in range(12))}
    assert all(p in _LOSERS["untried"] for p in picks.values())
    assert len(set(picks.values())) > 1, "a fleet must not all explore the same action"
    # Stable: same agent, same pick, on every process.
    assert act.recommend(_LOSERS, agent_id="agent-1")["suggested_action"] == picks["agent-1"]
    assert act.stable_bucket("agent-1", 3) == act.stable_bucket("agent-1", 3)


def test_recommend_escalates_only_when_candidates_were_given() -> None:
    exhausted = {"tried": _LOSERS["tried"], "untried": []}
    assert act.recommend(exhausted, agent_id="x") is None
    rec = act.recommend(exhausted, agent_id="x", candidate_actions=["resolve:a", "resolve:b"])
    assert rec["mode"] == "escalate"


def test_recommend_acts_on_a_validated_top_hit_and_not_on_a_shifted_one() -> None:
    assert act.recommend(None, top_hit_status="validated")["mode"] == "act"
    assert act.recommend(None, top_hit_status="validated", top_hit_recent_failure=True) is None
    rec = act.recommend({"tried": [], "untried": ["resolve:c"]}, top_hit_status="validated", regime_shift=True)
    assert rec["mode"] == "explore" and rec["suggested_action"] == "resolve:c"


def test_recommend_is_silent_with_nothing_to_say() -> None:
    assert act.recommend(None) is None
    assert act.recommend({"tried": [], "untried": []}, top_hit_status="untested") is None


# ── SDK derivation at commit ───────────────────────────────────────────────


@pytest.fixture
def mem(tmp_path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    return AgentMemory(agent_id="support-agent", adapter=adapter)


def test_commit_outcome_derives_actions_taken_and_entity_paths(mem: AgentMemory, monkeypatch) -> None:
    captured = {}
    original = mem._adapter.commit_outcome

    def spy(record):
        captured["record"] = record
        return original(record)

    monkeypatch.setattr(mem._adapter, "commit_outcome", spy)
    mem.write("acme/support", "fix-a", "try updating the payment method", confidence=0.8)
    mem._read_tracker.clear()
    mem.read("acme/support", "fix-a")
    mem.record_action("resolve", {"action": "update_payment_method"}, success=False)
    mem.record_attempt(summary="payment method update did nothing")
    mem.record_action("resolve", {"action": "resend_email"}, action_key="resolve:resend_email")
    mem.commit_outcome("t-1", OutcomeType.SUCCESS, entity_path="acme/tickets", situation="card declined")

    rec = captured["record"]
    assert [(a["action_key"], a["success"]) for a in rec.actions_taken] == [
        ("resolve:update_payment_method", False),
        ("resolve:resend_email", True),
    ]
    assert rec.entity_paths == ["acme/tickets", "acme/support"]
    assert rec.situation == "card declined"


def test_commit_outcome_scans_the_action_key_for_secrets(mem: AgentMemory, monkeypatch) -> None:
    """When the gate blocks the derived key, the action keeps only its tool name."""
    import amfs.memory as memmod

    captured = {}
    original = mem._adapter.commit_outcome
    monkeypatch.setattr(mem._adapter, "commit_outcome", lambda r: (captured.setdefault("r", r), original(r))[1])
    real_scan = memmod.scan_captured_text

    def gate(text, **kw):
        if text and "AKIA" in text:
            return None  # what the Pro SafetyGate does with a credential
        return real_scan(text, **kw)

    monkeypatch.setattr(memmod, "scan_captured_text", gate)
    mem.record_action("rotate", {"token": "AKIAIOSFODNN7EXAMPLE"})
    mem.commit_outcome("t-2", OutcomeType.SUCCESS)
    assert captured["r"].actions_taken[0]["action_key"] == "rotate"


# ── server: priors ride on retrieve ────────────────────────────────────────


@pytest.fixture
def server_mem(monkeypatch, tmp_path) -> AgentMemory:
    from amfs_http import server

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="http-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: None)
    monkeypatch.setattr(server, "_retrieval_reranker", None)
    monkeypatch.setattr(server, "_retrieval_query_rewriter", None)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
    handle.write("acme/support", "fix-a", "card declined: resend the receipt email", confidence=0.8)
    handle.write("acme/support", "fix-b", "card declined: update the payment method", confidence=0.8)
    handle._read_tracker.clear()
    return handle


@pytest.fixture
def client(server_mem):
    from amfs_http import server
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _stub_stats(adapter, rows):
    adapter.action_stats = lambda entity_path, **kw: list(rows)  # type: ignore[attr-defined]


def test_retrieve_without_include_priors_is_unchanged(client, server_mem) -> None:
    _stub_stats(server_mem._adapter, [_row([("resolve:a", False)])])
    resp = client.post("/api/v1/retrieve", json={"query": "card declined", "entity_path": "acme/support"})
    assert resp.status_code == 200
    assert all(not e.get("_meta") for e in resp.json())
    assert "posterior" in resp.json()[0] and resp.json()[0]["posterior"] == {"p": 0.5, "n": 0}


def test_retrieve_with_priors_appends_one_meta_element(client, server_mem) -> None:
    _stub_stats(server_mem._adapter, [
        _row([("resolve:update_payment_method", False)], agent="a1"),
        _row([("resolve:update_payment_method", False)], agent="a2"),
        _row([("resolve:resend_email", True)], agent="a1"),
        _row([("resolve:resend_email", True)], agent="a3"),
    ])
    resp = client.post("/api/v1/retrieve", json={
        "query": "card declined", "entity_path": "acme/support", "include_priors": True,
        "agent_id": "a9", "candidate_actions": ["resolve:resend_email", "resolve:update_payment_method", "resolve:refund"],
    })
    assert resp.status_code == 200, resp.text
    metas = [e for e in resp.json() if e.get("_meta")]
    assert len(metas) == 1 and resp.json()[-1] is not None and resp.json()[-1].get("_meta")
    meta = metas[0]
    tried = {t["action_key"]: t for t in meta["priors"]["tried"]}
    assert tried["resolve:resend_email"]["won"] == 2 and tried["resolve:resend_email"]["agents"] == 2
    assert tried["resolve:update_payment_method"]["lost"] == 2
    assert meta["priors"]["untried"] == ["resolve:refund"]
    assert meta["recommendation"]["mode"] == "act"
    assert meta["recommendation"]["suggested_action"] == "resolve:resend_email"


def test_retrieve_priors_explore_when_all_tried_failed(client, server_mem) -> None:
    _stub_stats(server_mem._adapter, [_row([("resolve:a", False)], agent=f"a{i}") for i in range(3)])
    resp = client.post("/api/v1/retrieve", json={
        "query": "card declined", "entity_path": "acme/support", "include_priors": True,
        "agent_id": "a1", "candidate_actions": ["resolve:a", "resolve:b"],
    })
    meta = resp.json()[-1]
    assert meta["recommendation"]["mode"] == "explore"
    assert meta["recommendation"]["suggested_action"] == "resolve:b"


def test_retrieve_priors_need_an_entity_path(client, server_mem) -> None:
    _stub_stats(server_mem._adapter, [_row([("resolve:a", False)])])
    resp = client.post("/api/v1/retrieve", json={"query": "card declined", "include_priors": True})
    assert all(not e.get("_meta") for e in resp.json())


def test_http_adapter_lifts_the_meta_element_out(monkeypatch) -> None:
    from amfs_adapter_http import adapter as http_adapter

    a = http_adapter.HttpAdapter.__new__(http_adapter.HttpAdapter)
    payload = [
        {"entity_path": "acme/support", "key": "fix-a", "version": 1, "value": "v", "confidence": 0.8,
         "provenance": {"agent_id": "x", "session_id": "s", "written_at": datetime.now(UTC).isoformat()},
         "_score": 0.5, "_breakdown": {}},
        {"_meta": True, "priors": {"tried": [], "untried": ["resolve:b"]}, "recommendation": None},
    ]
    monkeypatch.setattr(a, "_post", lambda path, body: payload)
    monkeypatch.setattr(a, "_capture_reuse_value", lambda: None)
    rows = a.retrieve("card declined", entity_path="acme/support", include_priors=True)
    assert [e.key for e, _, _ in rows] == ["fix-a"]
    assert a._last_retrieve_meta == {"priors": {"tried": [], "untried": ["resolve:b"]}, "recommendation": None}


def test_compact_retrieve_keeps_what_an_agent_acts_on(client, server_mem) -> None:
    full = client.post("/api/v1/retrieve", json={"query": "card declined", "entity_path": "acme/support"}).json()[0]
    compact = client.post("/api/v1/retrieve", json={"query": "card declined", "entity_path": "acme/support", "compact": True}).json()[0]
    for k in ("entity_path", "key", "value", "confidence", "evidence_status", "posterior", "validators", "_score"):
        assert k in compact, k
    for k in ("importance_dimensions", "integrity_chain", "content_hash", "tier", "ttl_at", "recall_count"):
        assert k not in compact, k
    assert len(str(compact)) < len(str(full)) * 0.6
    # Still a MemoryEntry for the client.
    from amfs_core.models import MemoryEntry

    MemoryEntry.model_validate({k: v for k, v in compact.items() if not k.startswith("_")})


def test_compact_retrieve_carries_the_first_two_hits_whole_and_the_rest_as_previews(client, server_mem) -> None:
    from amfs_http import server

    long = "card declined: " + ("resend the receipt email and ask the customer to retry; " * 12)
    for i in range(6):
        server_mem.write("acme/support", f"long-{i}", long + f" note {i}", confidence=0.6)
    rows = client.post("/api/v1/retrieve", json={
        "query": "card declined", "entity_path": "acme/support", "compact": True, "limit": 7,
    }).json()
    rows = [r for r in rows if not r.get("_meta")]
    assert len(rows) >= 4
    for r in rows[: server._COMPACT_FULL_HITS]:
        assert not r.get("value_truncated"), r["key"]
    assert any(r["key"].startswith("long-") for r in rows[server._COMPACT_FULL_HITS:])
    for r in rows[server._COMPACT_FULL_HITS:]:
        assert r.get("value_truncated") is True, r["key"]
        assert len(r["value"]) <= server._COMPACT_TAIL_CHARS + len(" …[truncated]")
    full = client.post("/api/v1/retrieve", json={"query": "card declined", "entity_path": "acme/support", "limit": 7}).json()
    assert len(str(rows)) <= 0.55 * len(str(full))
