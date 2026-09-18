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


def _outcomes(mem: AgentMemory, key: str, *outcomes: OutcomeType) -> None:
    for i, outcome in enumerate(outcomes):
        mem.read("acme/support", key)
        mem.commit_outcome(f"{key}-{i}", outcome)
    mem._read_tracker.clear()


def _priors_meta(client, **extra):
    resp = client.post("/api/v1/retrieve", json={
        "query": "card declined", "entity_path": "acme/support", "include_priors": True,
        "agent_id": "a1", "candidate_actions": ["resolve:a", "resolve:b"], **extra,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body, body[-1]


def test_a_first_strike_on_a_long_validated_rule_is_not_a_regime_shift(client, server_mem) -> None:
    """One failure against eight successes stays ``validated`` under first-strike
    tolerance, so it must not flag a regime shift either — the flag overrides a
    winning prior and forces ``explore``, which would skip the action that has
    been winning here on the same failure the label forgives."""
    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE)

    body, meta = _priors_meta(client)
    hit = next(e for e in body if e.get("key") == "fix-a")
    assert hit["evidence_status"] == "validated"
    assert meta["regime_shift"] is False
    assert meta["recommendation"]["mode"] == "act"
    assert meta["recommendation"]["suggested_action"] == "resolve:a"


def test_a_second_failure_flags_the_shift_even_after_the_rule_leaves_the_head(client, server_mem) -> None:
    """Two failures in a row discredit the rule, which drops it out of the
    ranked list. The flag reads the discredited entries kept aside as well, so
    the clearest case of a regime shift — a rule validated eight times and then
    discredited — is the one that fires it, and the winner is skipped for an
    untried action."""
    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    body, meta = _priors_meta(client)
    assert all(e.get("key") != "fix-a" for e in body if not e.get("_meta")), "discredited rule left the head"
    assert meta["regime_shift"] is True
    assert meta["recommendation"]["mode"] == "explore"
    assert meta["recommendation"]["suggested_action"] == "resolve:b"
    assert "regime shift" in meta["recommendation"]["why"]


def test_the_shift_still_fires_when_the_confidence_gate_hid_the_discredited_rule(client, server_mem) -> None:
    """The candidate fetch honours min_confidence; a discredited rule sits below
    the discredit threshold, so a retrieve gated there (the benchmark's setting)
    never saw it — and the two-failure signal went with it. The flag reads a
    separate, ungated fetch of the query-matched discredited rows."""
    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    entry = server_mem._adapter.read("acme/support", "fix-a")
    assert entry.discredited_at is not None and entry.confidence < 0.5

    body, meta = _priors_meta(client, min_confidence=0.5)
    assert all(e.get("key") != "fix-a" for e in body if not e.get("_meta"))
    assert meta["regime_shift"] is True
    assert meta["recommendation"]["mode"] == "explore"


def test_the_below_gate_read_is_entity_wide_not_a_rerun_of_the_query(client, server_mem) -> None:
    """The rule that stopped working need not share a word with this query, and
    a rule validated over months is old by write time. The below-gate read is
    the entity's discredited rows, whatever they say and whenever they were
    written — the same scope priors and the briefing use."""
    server_mem.write("acme/support", "fix-old", "rotate the ingest worker on a stuck queue", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-old", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    assert server_mem._adapter.read("acme/support", "fix-old").discredited_at is not None

    _, meta = _priors_meta(client, min_confidence=0.5)  # query: "card declined" — no overlap with fix-old
    assert meta["regime_shift"] is True
    assert meta["recommendation"]["mode"] == "explore"


def test_the_off_query_rule_fires_the_shift_at_the_default_gate_too(client, server_mem) -> None:
    """``min_confidence`` defaults to 0, so the documented include_priors call
    has no gate to lift — the entity-wide read must still run, or a rule that
    stopped working but shares no words with the query never flags."""
    server_mem.write("acme/support", "fix-old", "rotate the ingest worker on a stuck queue", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-old", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    _, meta = _priors_meta(client)  # default min_confidence=0.0
    assert meta["regime_shift"] is True
    assert meta["recommendation"]["mode"] == "explore"


def test_the_below_gate_read_respects_the_callers_visibility(client, server_mem, monkeypatch) -> None:
    """Rows the caller cannot see must not steer the recommendation either. The
    rows are not returned, but the policy decision they drive is."""
    from amfs_http import server as http_server

    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    class _HideFixA:
        def should_filter(self) -> bool:
            return True

        def filter_entries(self, entries):
            return [e for e in entries if e.key != "fix-a"]

    monkeypatch.setattr(http_server, "_get_visibility_filter", lambda request: _HideFixA())
    _, meta = _priors_meta(client, min_confidence=0.5)
    assert meta["regime_shift"] is False
    assert meta["recommendation"]["mode"] == "act"


def test_the_shift_clears_after_the_window(client, server_mem, monkeypatch) -> None:
    """A shift is an event. A rule discredited two weeks ago is history the
    discredited section covers, not a standing reason to skip the action that
    has been winning here since."""
    from amfs_http import server as http_server

    _stub_stats(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    from amfs_core.evidence import regime_shifted

    # The server takes its clock from a function-local import, so move the
    # predicate's clock instead: the same call the server makes, two weeks on.
    later = datetime.now(UTC) + timedelta(days=14)
    monkeypatch.setattr(
        http_server, "_regime_shifted",
        lambda e, now=None, **kw: regime_shifted(e, now=later, **kw),
    )
    _, meta = _priors_meta(client)
    assert meta["regime_shift"] is False
    assert meta["recommendation"]["mode"] == "act"
    assert meta["recommendation"]["suggested_action"] == "resolve:a"


def test_a_winner_that_has_won_since_the_shift_is_still_acted_on() -> None:
    """The shift skips winners because their wins may predate it. One whose
    latest take came after the shift and won is the replacement the shift called
    for; exploring past it would re-learn what the record already knows."""
    shift_at = datetime.now(UTC) - timedelta(hours=6)
    stale = {"action_key": "resolve:old", "won": 5, "lost": 0, "p": 0.86, "n": 5, "agents": 2,
             "last_3": ["won", "won", "won"], "last_at": (shift_at - timedelta(days=2)).isoformat()}
    fresh = {"action_key": "resolve:new", "won": 3, "lost": 0, "p": 0.8, "n": 3, "agents": 1,
             "last_3": ["won", "won", "won"], "last_at": (shift_at + timedelta(hours=1)).isoformat()}

    rec = act.recommend({"tried": [stale, fresh], "untried": ["resolve:c"]}, agent_id="a1",
                        regime_shift=True, regime_shift_at=shift_at)
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:new"
    assert "since" in rec["why"]

    # Only stale winners: the shift still sends the agent exploring.
    rec = act.recommend({"tried": [stale], "untried": ["resolve:c"]}, agent_id="a1",
                        regime_shift=True, regime_shift_at=shift_at)
    assert rec["mode"] == "explore" and rec["suggested_action"] == "resolve:c"

    # A fresh take that lost is not a replacement.
    lost = dict(fresh, last_3=["lost", "won", "won"])
    rec = act.recommend({"tried": [stale, lost], "untried": ["resolve:c"]}, agent_id="a1",
                        regime_shift=True, regime_shift_at=shift_at)
    assert rec["mode"] == "explore"

    # Without a moment to be after, every win is read as pre-shift.
    rec = act.recommend({"tried": [fresh], "untried": ["resolve:c"]}, agent_id="a1", regime_shift=True)
    assert rec["mode"] == "explore"


def test_regime_shifted_reads_the_label_not_a_failure_ratio() -> None:
    """The predicate has to agree with first-strike tolerance, and a ratio over
    the evidence masses cannot: after eight successes one failure already
    carries about half the success mass (severity 2 against a decayed run)."""
    from types import SimpleNamespace

    from amfs_core.evidence import regime_shifted

    def entry(**kw):
        base = dict(success_count=0, failure_count=0, last_outcome=None,
                    evidence_success=0.0, evidence_failure=0.0, discredited_at=None,
                    outcome_count=0, confidence=0.8, last_outcome_at=datetime.now(UTC))
        base.update(kw)
        return SimpleNamespace(**base)

    # 8 successes then 1 failure, with the masses the evidence model produces.
    first_strike = entry(success_count=8, failure_count=1, last_outcome="failure",
                         evidence_success=3.73, evidence_failure=2.0, outcome_count=9)
    assert first_strike.evidence_failure >= first_strike.evidence_success * 0.5, "a ratio rule would fire here"
    assert regime_shifted(first_strike) is False
    # The second failure discredits it.
    second = entry(success_count=8, failure_count=2, last_outcome="failure",
                   evidence_success=2.98, evidence_failure=4.93, outcome_count=10,
                   discredited_at=datetime.now(UTC))
    assert regime_shifted(second) is True
    # A failure buried under later successes is history, not a shift.
    recovered = entry(success_count=9, failure_count=2, last_outcome="success",
                      evidence_success=3.94, evidence_failure=3.94, outcome_count=11)
    assert regime_shifted(recovered) is False
    # Never validated enough to have a regime to shift from.
    young = entry(success_count=1, failure_count=1, last_outcome="failure",
                  evidence_success=1.3, evidence_failure=2.0, outcome_count=2)
    assert regime_shifted(young) is False
    # An entry that carries its own label is read through it.
    labelled = entry(success_count=5, failure_count=3, last_outcome="failure", evidence_status="contested")
    assert regime_shifted(labelled) is True
    # The window: the same discredited rule, two weeks on, is history.
    old = entry(success_count=8, failure_count=2, last_outcome="failure",
                evidence_success=2.98, evidence_failure=4.93, outcome_count=10,
                discredited_at=datetime.now(UTC) - timedelta(days=14),
                last_outcome_at=datetime.now(UTC) - timedelta(days=14))
    assert regime_shifted(old) is False
    assert regime_shifted(old, window_days=None) is True
    # No timestamp cannot be called recent.
    assert regime_shifted(entry(success_count=8, failure_count=2, last_outcome="failure",
                                evidence_status="discredited", last_outcome_at=None)) is False


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
