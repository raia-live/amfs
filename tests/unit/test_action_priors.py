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
    cands = ["resolve:a", "resolve:c"]
    assert act.recommend(None, top_hit_status="validated", candidate_actions=cands)["mode"] == "act"
    assert act.recommend(None, top_hit_status="validated", top_hit_recent_failure=True,
                         candidate_actions=cands) is None
    # A shift elsewhere on the entity does not override the hit's own record...
    mixed = {"tried": [{"action_key": "resolve:a", "won": 1, "lost": 1, "p": 0.5, "n": 2,
                        "agents": 1, "last_3": ["lost", "won"], "last_at": None}],
             "untried": ["resolve:c"]}
    rec = act.recommend(mixed, top_hit_status="validated", regime_shift=True, candidate_actions=cands)
    assert rec["mode"] == "act" and rec["suggested_action"] is None
    assert "elsewhere" in rec["why"]
    # ...but the hit being the rule that shifted does.
    rec = act.recommend(mixed, top_hit_status="validated", regime_shift=True, top_hit_shifted=True,
                        candidate_actions=cands)
    assert rec["mode"] == "explore" and rec["suggested_action"] == "resolve:c"


def test_a_validated_top_hit_alone_is_not_an_act_without_candidate_actions() -> None:
    """No candidates, no action to act with. A caller whose task is not a choice
    among a fixed set of actions gets no ``act`` off a bare validated hit — the
    hit's own ``evidence_status`` already says it is validated, and grid v5
    measured what the extra ``act`` costs such a caller (42% vs 6% failures)."""
    assert act.recommend(None, top_hit_status="validated") is None
    assert act.recommend(None, top_hit_status="validated", candidate_actions=[]) is None
    # A real winner among candidates still acts, with or without a validated hit.
    won = {"tried": [{"action_key": "resolve:a", "won": 3, "lost": 0, "p": 1.0, "n": 3,
                      "agents": 1, "last_3": ["won", "won", "won"], "last_at": None}],
           "untried": []}
    assert act.recommend(won, top_hit_status="untested")["suggested_action"] == "resolve:a"


def _prior(key: str, won: int, n: int, last_3: list[str], p: float | None = None) -> dict:
    return {"action_key": key, "won": won, "lost": n - won, "n": n, "agents": 1,
            "p": (won / n) if p is None else p, "last_3": last_3, "last_at": None}


def test_a_winner_that_lost_its_last_three_is_not_acted_on() -> None:
    """Grid v5, ci-fix after the change: rerun_job had won every flaky-integration
    task, then lost every one since. Its lifetime ratio kept it a winner for a
    dozen more episodes; the record's newest takes say the rule has turned."""
    cands = ["fix:rerun_job", "fix:fix_code", "fix:edit_generated_file", "fix:add_audit_exception"]
    stale = {"tried": [_prior("fix:rerun_job", 8, 11, ["lost", "lost", "lost"]),
                       _prior("fix:fix_code", 0, 4, ["lost", "lost", "lost"])],
             "untried": ["fix:edit_generated_file", "fix:add_audit_exception"]}
    rec = act.recommend(stale, agent_id="a", candidate_actions=cands)
    assert rec["mode"] == "explore"
    assert rec["suggested_action"] in stale["untried"]
    assert rec["stopped_working"] == ["fix:rerun_job"]
    assert "stopped working" in rec["why"] and "8/11" in rec["why"]
    # One loss against a run of wins is forgiven: the lifetime record still stands.
    fresh = {"tried": [_prior("fix:rerun_job", 8, 9, ["lost", "won", "won"])], "untried": ["fix:fix_code"]}
    assert act.recommend(fresh, candidate_actions=cands)["mode"] == "act"
    # The streak shows in the rendered line, so the agent reads it without the recommendation.
    assert "fix:rerun_job 8/11, lost last 3" in act.render_priors(stale, None)
    assert "lost last" not in act.render_priors(fresh, None)


def test_a_validated_winner_that_lost_its_last_two_has_turned() -> None:
    """The action-level regime rule, the same line ``regime_shifted`` draws for
    entries: one failure against a run of wins is forgiven, the second in a row
    is not. Ops-queue CI demo, 2026-09-22: after the change ``rerun_job`` 3/4
    with its newest take lost was still ``act`` on the next flaky task, and the
    one after; three CI runs a task until the third loss. The entry-level rule
    never fired because the agent had rewritten its lesson ("no fix known") and
    the rewrite opened a claim with an empty record; the situation's action
    record keeps every take."""
    cands = ["fix:rerun_job", "fix:fix_code", "fix:edit_generated_file"]
    two = {"tried": [_prior("fix:rerun_job", 8, 10, ["lost", "lost", "won"])], "untried": ["fix:fix_code", "fix:edit_generated_file"]}
    rec = act.recommend(two, agent_id="a", candidate_actions=cands)
    assert act.turned(two["tried"][0])
    assert rec["mode"] == "explore"
    assert rec["stopped_working"] == ["fix:rerun_job"]
    assert "lost its last 2" in rec["why"]
    assert "fix:rerun_job 8/10, lost last 2" in act.render_priors(two, None)
    # Not a *validated* action: a 1/3 that lost twice is a poor record, not a turn.
    thin = {"tried": [_prior("fix:rerun_job", 1, 3, ["lost", "lost", "won"])], "untried": ["fix:fix_code"]}
    assert not act.turned(thin["tried"][0])
    # The strength label reads the same rule: a turned winner is not strong.
    assert act.guidance_strength(two, []) == "thin"


def test_an_action_tried_once_and_lost_does_not_block_explore() -> None:
    """The flaky-integration record at the end of grid v5: six actions tried, none
    winning, one of them 0/1 — and no recommendation, because a single loss
    was not a firm failure. The agent kept cycling the same three actions and
    never reached the two untried ones."""
    cands = ["fix:" + a for a in ("run_formatter", "fix_code", "rerun_job", "update_snapshots",
                                  "regen_migrations", "bump_dependency", "add_audit_exception",
                                  "edit_generated_file")]
    record = {"tried": [_prior("fix:update_snapshots", 0, 1, ["lost"]),
                        _prior("fix:run_formatter", 5, 8, ["won", "won", "lost"], p=0.37),
                        _prior("fix:regen_migrations", 0, 2, ["lost", "lost"]),
                        _prior("fix:rerun_job", 3, 13, ["lost", "lost", "lost"]),
                        _prior("fix:bump_dependency", 0, 6, ["lost", "lost", "lost"]),
                        _prior("fix:fix_code", 0, 12, ["lost", "lost", "lost"])],
              "untried": ["fix:add_audit_exception", "fix:edit_generated_file"]}
    rec = act.recommend(record, agent_id="ci-agent-3", candidate_actions=cands)
    assert rec["mode"] == "explore" and rec["suggested_action"] in record["untried"]
    # ...but a 0/1 is not firm enough to *escalate* on when nothing is untried.
    exhausted = {"tried": [_prior("fix:a", 0, 1, ["lost"]), _prior("fix:b", 0, 3, ["lost", "lost", "lost"])],
                 "untried": []}
    assert act.recommend(exhausted, candidate_actions=["fix:a", "fix:b"]) is None
    firm = {"tried": [_prior("fix:a", 0, 2, ["lost", "lost"]), _prior("fix:b", 0, 3, ["lost", "lost", "lost"])],
            "untried": []}
    assert act.recommend(firm, candidate_actions=["fix:a", "fix:b"])["mode"] == "escalate"


def test_explore_needs_something_tried_to_explore_from() -> None:
    """With nothing tried, "untried" is every candidate and the pick is a hash
    of the agent's name: no information, so no advice."""
    assert act.recommend({"tried": [], "untried": ["resolve:c"]}, regime_shift=True) is None
    assert act.recommend({"tried": [], "untried": ["resolve:c"]}, top_hit_status="contested",
                         regime_shift=True) is None


def test_a_shift_over_entity_wide_priors_does_not_explore() -> None:
    """The action_stats fallback is every outcome on the entity. A winner there
    is acted on; a shift read over it is not a reason to explore."""
    rec = act.recommend(_LOSERS, agent_id="x", candidate_actions=["resolve:a", "resolve:b", "resolve:c"],
                        regime_shift=True, priors_are_local=False)
    # all tried failed still explores — that claim holds for the whole entity.
    assert rec["mode"] == "explore"
    mixed = {"tried": [{"action_key": "resolve:a", "won": 2, "lost": 1, "p": 0.6, "n": 3,
                        "agents": 1, "last_3": ["won", "won", "lost"], "last_at": None}],
             "untried": ["resolve:c"]}
    assert act.recommend(mixed, regime_shift=True, priors_are_local=False, agent_id="x") is None
    assert act.recommend(mixed, regime_shift=True, priors_are_local=True, agent_id="x")["mode"] == "explore"


def test_neighbourhood_weights_are_relative_to_the_best_match() -> None:
    rows = [_row([("resolve:a", True)], sim=0.92), _row([("resolve:a", True)], sim=0.90),
            _row([("resolve:b", False)], sim=0.86), _row([("resolve:b", False)], sim=0.80)]
    out = act.neighbourhood_weights(rows)
    weights = {round(r["task_similarity"], 2): round(r["similarity"], 3) for r in out}
    assert weights[0.92] == 1.0
    assert 0.4 < weights[0.90] < 0.6          # 0.02 behind: exp(-2/3)
    assert 0.1 < weights[0.86] < 0.2          # 0.06 behind: exp(-2)
    assert 0.80 not in weights, "0.12 behind is dropped, not counted"
    priors = act.aggregate_priors(out)
    tried = {t["action_key"]: t for t in priors["tried"]}
    assert tried["resolve:a"]["n"] == 2 and tried["resolve:b"]["n"] == 1
    assert tried["resolve:a"]["p"] > 0.6, "the near wins carry their weight"
    assert tried["resolve:b"]["p"] > 0.4, "one far loss at a fifth of a weight is not a loser"


def test_neighbourhood_weights_leave_rows_without_similarity_alone() -> None:
    rows = [{"actions_taken": [], "committed_at": None, "agent_id": "a"}]
    assert act.neighbourhood_weights(rows) == rows
    assert act.neighbourhood_weights([]) == []


def test_recommend_is_silent_with_nothing_to_say() -> None:
    assert act.recommend(None) is None
    assert act.recommend({"tried": [], "untried": []}, top_hit_status="untested") is None


def test_recommend_abstains_only_when_asked_and_only_on_weak_evidence() -> None:
    """Opt-in: the silent case above must stay silent for callers that did not
    ask. With ``abstain=True``, no priors and nothing validated in the hits is
    said out loud; one validated hit or any prior turns it off."""
    empty = {"tried": [], "untried": []}
    rec = act.recommend(empty, top_hit_status="untested", abstain=True)
    assert rec["mode"] == "abstain" and rec["suggested_action"] is None
    assert "1 untested" in rec["why"]
    rec = act.recommend(empty, abstain=True, hit_statuses=["untested", "contested", "discredited"])
    assert rec["mode"] == "abstain"
    # Nothing to rate at all: still silent (no hits, no priors).
    assert act.recommend(empty, abstain=True) is None
    # A validated hit never abstains. With candidate actions it acts; without
    # them (#429) there is no action to act *with*, so it stays silent — the
    # hit's own evidence_status carries the message — but it is not "weak".
    assert act.recommend(
        empty, top_hit_status="validated", abstain=True, candidate_actions=["resolve:a"],
    )["mode"] == "act"
    assert act.recommend(empty, top_hit_status="validated", abstain=True) is None
    # A winning prior acts.
    pr = {"tried": [{"action_key": "resolve:b", "won": 3, "lost": 0, "p": 0.8, "n": 3, "agents": 1}], "untried": []}
    assert act.recommend(pr, top_hit_status="untested", abstain=True)["mode"] == "act"
    # A thin prior (n=1) is still a prior: not abstain, not act — silent.
    thin = {"tried": [{"action_key": "resolve:b", "won": 1, "lost": 0, "p": 0.67, "n": 1, "agents": 1}], "untried": []}
    assert act.recommend(thin, top_hit_status="untested", abstain=True) is None


def test_guidance_strength() -> None:
    win = {"tried": [{"action_key": "a", "won": 3, "lost": 0, "p": 0.8, "n": 3}], "untried": []}
    thin = {"tried": [{"action_key": "a", "won": 1, "lost": 0, "p": 0.67, "n": 1}], "untried": []}
    assert act.guidance_strength(None, None) == "none"
    assert act.guidance_strength(None, ["untested", "discredited"]) == "none"
    assert act.guidance_strength(None, ["validated"]) == "strong"
    assert act.guidance_strength(win, ["untested"]) == "strong"
    assert act.guidance_strength(thin, ["untested"]) == "thin"
    assert act.guidance_strength(None, ["validated"], regime_shift=True) == "thin"
    assert act.guidance_strength(None, [], regime_shift=True) == "none"


def test_aggregate_priors_down_weights_another_environment() -> None:
    """A win under python3.9 counts for a python3.12 run, at half weight; a row
    that reports no environment counts in full; no ``environment`` given
    leaves every weight as it was."""
    now = datetime.now(UTC)
    rows = [
        {"actions_taken": [{"action_key": "fix:pin", "success": True}], "committed_at": now,
         "agent_id": "a", "session_metadata": {"runtime": "python3.9"}},
        {"actions_taken": [{"action_key": "fix:pin", "success": False}], "committed_at": now,
         "agent_id": "b", "session_metadata": {"runtime": "python3.12"}},
        {"actions_taken": [{"action_key": "fix:pin", "success": True}], "committed_at": now,
         "agent_id": "c"},
    ]
    plain = act.aggregate_priors(rows)["tried"][0]
    scoped = act.aggregate_priors(rows, environment={"runtime": "python3.12"})["tried"][0]
    # Counts are counts either way; the posterior moves with the weights.
    assert plain["won"] == scoped["won"] == 2 and plain["lost"] == scoped["lost"] == 1
    assert plain["p"] == pytest.approx((2 + 1) / (3 + 2), abs=1e-3)
    assert scoped["p"] == pytest.approx((1.5 + 1) / (2.5 + 2), abs=1e-3)
    assert scoped["p"] < plain["p"]
    # Same runtime as the run: full weight, identical to plain.
    same = act.aggregate_priors(rows, environment={"runtime": "python3.9"})["tried"][0]
    assert same["p"] == pytest.approx((2 + 1) / (2.5 + 2), abs=1e-3)


def test_recorded_environment_reads_every_place_a_producer_puts_it() -> None:
    """``Run.begin`` and ``set_session_attributes`` stamp agent_version /
    runtime into ``session_metadata.attributes``; ``model`` sits at the top of
    the metadata; the store may return an ``environment`` block of its own.
    Priors must see all of them, first source to name a key winning."""
    row = {
        "session_metadata": {
            "model": "gpt-5",
            "attributes": {"agent_version": "2.1.0", "runtime": "python3.12", "noise": "x"},
        },
    }
    assert act.recorded_environment(row) == {
        "model": "gpt-5", "agent_version": "2.1.0", "runtime": "python3.12",
    }
    # A stored environment block is authoritative over the metadata.
    assert act.recorded_environment({**row, "environment": {"runtime": "node20"}})["runtime"] == "node20"
    # A trace's attribute bag counts too; blanks and non-strings are skipped.
    assert act.recorded_environment({"attributes": {"runtime": "  ", "model": 3}}) == {}
    assert act.recorded_environment({}) == {}
    # And the weights follow: an attribute-stamped runtime mismatch is down-weighted.
    now = datetime.now(UTC)
    rows = [
        {"actions_taken": [{"action_key": "fix:pin", "success": True}], "committed_at": now,
         "agent_id": "a", "session_metadata": {"attributes": {"runtime": "python3.9"}}},
    ]
    plain = act.aggregate_priors(rows)["tried"][0]
    scoped = act.aggregate_priors(rows, environment={"runtime": "python3.12"})["tried"][0]
    assert scoped["p"] < plain["p"]
# ── contrasts: one fail-then-succeed outcome ───────────────────────────────


def _contrast_row(failed, resolved, *, sim=1.0, days_ago=0, agent="a1", ref="o1"):
    """An outcome in which attempt 1 ended on *failed* and lost, and the terminal
    action *resolved* won — the shape ``actions_taken`` produces."""
    row = _row([], sim=sim, days_ago=days_ago, agent=agent)
    row["actions_taken"] = [
        {"action_key": failed, "success": False, "attempt": 1},
        {"action_key": resolved, "success": True, "attempt": None},
    ]
    row["outcome_ref"] = ref
    return row


def test_aggregate_priors_extracts_contrast_pairs_and_keeps_the_per_action_counts() -> None:
    rows = act.neighbourhood_weights([
        _contrast_row("resolve:a", "resolve:b", sim=0.95, ref="near"),
        _contrast_row("resolve:a", "resolve:c", sim=0.90, ref="far"),
        _row([("resolve:a", True)], sim=0.95),           # a plain win, no contrast
        _row([("resolve:a", False)], sim=0.95),          # a plain loss, no contrast
    ])
    pr = act.aggregate_priors(rows)
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["resolve:a"]["lost"] == 3 and by["resolve:a"]["won"] == 1
    assert by["resolve:b"]["won"] == 1
    assert [c["outcome_ref"] for c in pr["contrasts"]] == ["near", "far"], "heaviest first"
    near = pr["contrasts"][0]
    assert near["failed"] == ["resolve:a"] and near["resolved_with"] == "resolve:b"
    assert near["weight"] == 1.0 and near["task_similarity"] == 0.95
    assert pr["contrasts"][1]["weight"] < 0.25, "0.05 behind the best is not near-identical"


def test_a_retry_with_the_same_action_is_not_a_contrast() -> None:
    row = _row([], sim=1.0)
    row["actions_taken"] = [
        {"action_key": "resolve:a", "success": False, "attempt": 1},
        {"action_key": "resolve:a", "success": True, "attempt": None},
    ]
    assert act.aggregate_priors([row])["contrasts"] == []


def test_recommend_acts_on_the_resolving_action_of_a_near_identical_contrast() -> None:
    """One exposure is enough when the outcome holds both halves: A failed an
    attempt on this task and B resolved it. ACT_MIN_N would wait for a second
    win; grid v4 measured the wait as the same quirk failing 4-7 more times."""
    pr = act.aggregate_priors(act.neighbourhood_weights([_contrast_row("resolve:a", "resolve:b", sim=0.95)]))
    rec = act.recommend(pr, agent_id="x")
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:b"
    assert rec["contrast"]["failed"] == ["resolve:a"]
    assert "resolve:a failed and resolve:b resolved it" in rec["why"]


def test_a_contrast_displaces_a_winner_only_when_the_winner_is_what_failed() -> None:
    # A has a long record and just failed on this task; B resolved it.
    rows = act.neighbourhood_weights(
        [_row([("resolve:a", True)], sim=0.95, days_ago=d) for d in (3, 4, 5, 6, 7)]
        + [_contrast_row("resolve:a", "resolve:b", sim=0.95)]
    )
    pr = act.aggregate_priors(rows)
    a = next(t for t in pr["tried"] if t["action_key"] == "resolve:a")
    assert a["p"] >= act.ACT_MIN_P and a["n"] >= act.ACT_MIN_N, "A still qualifies as a winner"
    rec = act.recommend(pr, agent_id="x")
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:b"

    # An established C the contrast says nothing about keeps its recommendation.
    rows = act.neighbourhood_weights(
        [_row([("resolve:c", True)], sim=0.95, days_ago=d) for d in (3, 4, 5)]
        + [_contrast_row("resolve:a", "resolve:b", sim=0.95)]
    )
    rec = act.recommend(act.aggregate_priors(rows), agent_id="x")
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:c"


def test_a_contrast_is_ignored_when_far_when_b_lost_since_and_when_not_local() -> None:
    base = _row([("resolve:c", True)], sim=0.95)
    # Far: the pair is 0.06 behind the best match.
    pr = act.aggregate_priors(act.neighbourhood_weights([base, _contrast_row("resolve:a", "resolve:b", sim=0.89)]))
    assert pr["contrasts"] and pr["contrasts"][0]["weight"] < act.CONTRAST_MIN_W
    assert act.recommend(pr, agent_id="x") is None, "one far pair and one plain win say nothing"
    # B lost more recently than it resolved this task.
    pr = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("resolve:a", "resolve:b", sim=0.95, days_ago=2),
        _row([("resolve:b", False)], sim=0.95, days_ago=1),
    ]))
    rec = act.recommend(pr, agent_id="x")
    assert rec is None or rec["suggested_action"] != "resolve:b"
    # Entity-wide priors (action_stats fallback) are not known to be about this task.
    pr = act.aggregate_priors([_contrast_row("resolve:a", "resolve:b")])
    assert act.recommend(pr, agent_id="x", priors_are_local=False) is None


def test_under_a_regime_shift_the_contrast_must_postdate_it() -> None:
    pr = act.aggregate_priors(act.neighbourhood_weights([_contrast_row("resolve:a", "resolve:b", sim=0.95, days_ago=3)]))
    shift_after = datetime.now(UTC) - timedelta(days=1)
    shift_before = datetime.now(UTC) - timedelta(days=5)
    assert act.recommend(pr, agent_id="x", regime_shift=True, regime_shift_at=shift_after) is None
    rec = act.recommend(pr, agent_id="x", regime_shift=True, regime_shift_at=shift_before)
    assert rec["mode"] == "act" and rec["suggested_action"] == "resolve:b"
    assert act.recommend(pr, agent_id="x", regime_shift=True, regime_shift_at=None) is None


def _pooled_rows():
    """The ops-queue CI demo's snapshot neighbourhood: a backend PR (fix_code
    resolves what update_snapshots failed) and a UI PR (the reverse) share a
    description, and the queue alternates between them."""
    return act.neighbourhood_weights([
        _contrast_row("fix:update_snapshots", "fix:fix_code", sim=0.96, days_ago=3, ref="nonui-1"),
        _contrast_row("fix:fix_code", "fix:update_snapshots", sim=0.95, days_ago=2, ref="ui-1"),
        _contrast_row("fix:update_snapshots", "fix:fix_code", sim=0.96, days_ago=1, ref="nonui-2"),
    ])


def test_pooled_classes_reads_an_alternating_contradiction_and_not_a_single_flip() -> None:
    pr = act.aggregate_priors(_pooled_rows())
    pooled = act.pooled_classes(pr["contrasts"])
    assert pooled is not None
    assert sorted(pooled["actions"]) == ["fix:fix_code", "fix:update_snapshots"]
    assert pooled["sequence"] == ["fix:fix_code", "fix:update_snapshots", "fix:fix_code"]
    assert pooled["reversals"] == 2
    # One reversal is the shape of a rule that flipped, not of two classes:
    # every old outcome says A, every new one says B. Left to newest-wins.
    flip = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=4, ref="old-1"),
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=3, ref="old-2"),
        _contrast_row("fix:a", "fix:b", sim=0.96, days_ago=1, ref="new-1"),
    ]))
    assert act.pooled_classes(flip["contrasts"]) is None
    rec = act.recommend(flip, agent_id="x", abstain=True)
    assert rec["mode"] == "act" and rec["suggested_action"] == "fix:b", "the newest contrast wins a flip"
    # Two contrasts that merely name different resolvers without contradicting
    # each other (A over C, B over D) are not a pooled record either.
    apart = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("fix:c", "fix:a", sim=0.96, days_ago=3),
        _contrast_row("fix:d", "fix:b", sim=0.96, days_ago=2),
        _contrast_row("fix:c", "fix:a", sim=0.96, days_ago=1),
    ]))
    assert act.pooled_classes(apart["contrasts"]) is None
    # A far contrast does not take part.
    far = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=3),
        _contrast_row("fix:a", "fix:b", sim=0.88, days_ago=2),
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=1),
    ]))
    assert act.pooled_classes(far["contrasts"]) is None
    # Only rows that pit A against B are on the timeline. One flip (b then a)
    # plus a later win of A over some third action C is still one reversal.
    flip_plus_c = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("fix:a", "fix:b", sim=0.96, days_ago=4, ref="old"),
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=2, ref="new"),
        _contrast_row("fix:c", "fix:a", sim=0.96, days_ago=1, ref="a-over-c"),
    ]))
    assert act.pooled_classes(flip_plus_c["contrasts"]) is None
    # And with the wins over C on the other side of the flip, still one reversal.
    flip_c_first = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("fix:c", "fix:b", sim=0.96, days_ago=5, ref="b-over-c"),
        _contrast_row("fix:a", "fix:b", sim=0.96, days_ago=4, ref="old"),
        _contrast_row("fix:c", "fix:a", sim=0.96, days_ago=3, ref="a-over-c"),
        _contrast_row("fix:b", "fix:a", sim=0.96, days_ago=2, ref="new"),
    ]))
    assert act.pooled_classes(flip_c_first["contrasts"]) is None


def test_pooling_is_read_only_over_local_priors() -> None:
    """The ``action_stats`` fallback records every outcome on the entity at
    similarity 1.0, so mixed kinds of task there always look near-identical
    and always contradict. It can name a winner; it cannot say the record is
    pooled, thin the guidance, or warn the agent off the action record."""
    pr = act.aggregate_priors(_pooled_rows())
    entity_wide = {**pr, "source": "action_stats"}
    assert act.priors_local(pr) and not act.priors_local(entity_wide)
    # Strength: the winner stands over an entity-wide record …
    assert act.guidance_strength(entity_wide, ["untested"]) == "strong"
    assert act.guidance_strength(pr, ["untested"], priors_are_local=False) == "strong"
    # … and a validated hit is strong whatever the priors look like: its
    # record is about the hit, not about the neighbourhood.
    assert act.guidance_strength(pr, ["validated"]) == "strong"
    # Render: no contrast lines and no warning off an entity-wide record.
    text = act.render_priors(entity_wide, act.recommend(entity_wide, agent_id="x", priors_are_local=False))
    assert "near-identical" not in text and "alternating over time" not in text
    assert "Recommendation: act -> fix:fix_code" in text


def test_recommend_abstains_over_a_pooled_record_instead_of_naming_either_class_fix() -> None:
    pr = act.aggregate_priors(_pooled_rows())
    by = {t["action_key"]: t for t in pr["tried"]}
    # Without the rule, fix_code (2/3) is a winner and the newest contrast
    # resolves with it: ``act -> fix_code`` on a UI PR, the wrong class's fix.
    assert by["fix:fix_code"]["won"] == 2 and by["fix:fix_code"]["p"] >= act.ACT_MIN_P
    rec = act.recommend(pr, agent_id="x", abstain=True)
    assert rec["mode"] == "abstain" and rec["suggested_action"] is None
    assert rec["pooled"]["reversals"] == 2
    assert "fix:fix_code resolved what fix:update_snapshots failed" in rec["why"]
    assert "two kinds of task share this description" in rec["why"]
    # Without abstain the caller gets silence, not the other class's fix.
    assert act.recommend(pr, agent_id="x") is None
    # A pooled record is not strong guidance, whatever its winners' ratios.
    assert act.guidance_strength(pr, ["untested"]) == "thin"


def _two_classes_apart():
    """The same neighbourhood as measured on the live demo (2026-09-22): the
    backend PRs sit at weight ~1.0 against a backend query, the UI PRs — same
    jest check, a UI file changed — at 0.25-0.38. The UI class is a neighbour
    the query resembles, not a second class under the same description."""
    return act.neighbourhood_weights([
        _contrast_row("fix:update_snapshots", "fix:fix_code", sim=0.96, days_ago=4, ref="nonui-1"),
        _contrast_row("fix:fix_code", "fix:update_snapshots", sim=0.93, days_ago=3, ref="ui-1"),
        _contrast_row("fix:update_snapshots", "fix:fix_code", sim=0.96, days_ago=2, ref="nonui-2"),
        _contrast_row("fix:fix_code", "fix:update_snapshots", sim=0.93, days_ago=1, ref="ui-2"),
    ])


def test_a_neighbouring_class_at_lower_weight_is_not_a_pooled_contradiction() -> None:
    pr = act.aggregate_priors(_two_classes_apart())
    ws = sorted({round(c["weight"], 2) for c in pr["contrasts"]})
    assert ws[0] < act.POOLED_WEIGHT_RATIO * ws[-1], "the fixture must keep the classes apart in weight"
    assert act.pooled_classes(pr["contrasts"]) is None
    # The neighbourhood-weighted record names this class's fix: fix_code's two
    # wins weigh 1.0 each, its two losses on the UI PRs ~0.37 each.
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["fix:fix_code"]["p"] >= act.ACT_MIN_P > by["fix:update_snapshots"]["p"]
    assert by["fix:fix_code"]["lost_w"] < by["fix:fix_code"]["won_w"]
    rec = act.recommend(pr, agent_id="x", abstain=True)
    assert rec["mode"] == "act" and rec["suggested_action"] == "fix:fix_code"
    assert act.guidance_strength(pr, []) == "strong"
    # And the UI PR's newest loss for fix_code does not turn the backend
    # contrast: a loss weighing 0.37 against a pair weighing 1.0.
    text = act.render_priors(pr, rec)
    assert "alternating over time" not in text
    assert "fix:update_snapshots failed and fix:fix_code resolved it" in text


def test_a_loss_as_near_as_the_contrast_still_turns_it() -> None:
    """The flip case the weight rule must not undo: B resolved a task, then lost
    on an equally near one. Newest-wins stands."""
    pr = act.aggregate_priors(act.neighbourhood_weights([
        _contrast_row("resolve:a", "resolve:b", sim=0.95, days_ago=2),
        _row([("resolve:b", False)], sim=0.95, days_ago=1),
    ]))
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["resolve:b"]["lost_w"] >= act.POOLED_WEIGHT_RATIO * pr["contrasts"][0]["weight"]
    rec = act.recommend(pr, agent_id="x")
    assert rec is None or rec["suggested_action"] != "resolve:b"


def test_a_lone_winner_is_acted_on_when_everything_else_tried_has_failed() -> None:
    """Ops-queue CI demo, 2026-09-22, task 29: the changed class had been solved
    once (edit_generated_file 1/1) and every other action was 0/n or on a
    streak. With ``ACT_MIN_N`` alone there was nothing to say, and the task was
    brute-forced again and missed."""
    cands = ["fix:rerun_job", "fix:fix_code", "fix:run_formatter", "fix:edit_generated_file",
             "fix:regen_migrations", "fix:add_audit_exception"]
    pr = {"tried": [_prior("fix:edit_generated_file", 1, 1, ["won"], p=0.66),
                    _prior("fix:rerun_job", 3, 6, ["lost", "lost", "lost"], p=0.5),
                    _prior("fix:run_formatter", 0, 1, ["lost"], p=0.34),
                    _prior("fix:fix_code", 0, 3, ["lost", "lost", "lost"], p=0.2)],
          "untried": ["fix:regen_migrations", "fix:add_audit_exception"]}
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    assert rec["mode"] == "act" and rec["suggested_action"] == "fix:edit_generated_file"
    assert "only action that has worked" in rec["why"] and "fix:fix_code 0/3" in rec["why"]
    assert rec["plan"][0] == "fix:edit_generated_file"
    # One win with nothing else tried is a hint, not the best of the known options.
    alone = {"tried": [_prior("fix:edit_generated_file", 1, 1, ["won"], p=0.66)], "untried": cands[:2]}
    assert act.recommend(alone, agent_id="a", candidate_actions=cands) is None
    # A lone win whose newest take lost is not a winner either.
    turned = dict(pr, tried=[_prior("fix:edit_generated_file", 1, 2, ["lost", "won"], p=0.5)] + pr["tried"][1:])
    rec = act.recommend(turned, agent_id="a", candidate_actions=cands)
    assert rec is None or rec["suggested_action"] != "fix:edit_generated_file"
    # Not read over the entity-wide fallback: one win there is not known to be about this task.
    assert act.recommend(pr, agent_id="a", candidate_actions=cands, priors_are_local=False) is None


def test_the_plan_covers_the_budget_and_never_repeats_what_failed_here() -> None:
    """Task 23 of the same run: ``act -> rerun_job`` (3/4, newest lost — one loss,
    forgiven), then the agent's own second and third choices were fix_code
    (already 0/2 on this situation) and a guess. The plan puts the untried
    actions second and third and the known failures last."""
    cands = ["fix:rerun_job", "fix:fix_code", "fix:run_formatter", "fix:update_snapshots",
             "fix:regen_migrations", "fix:bump_dependency", "fix:add_audit_exception", "fix:edit_generated_file"]
    pr = {"tried": [_prior("fix:rerun_job", 3, 4, ["lost", "won", "won"]),
                    _prior("fix:fix_code", 0, 2, ["lost", "lost"]),
                    _prior("fix:run_formatter", 0, 1, ["lost"])],
          "untried": ["fix:update_snapshots", "fix:regen_migrations", "fix:bump_dependency",
                      "fix:add_audit_exception", "fix:edit_generated_file"]}
    rec = act.recommend(pr, agent_id="ci-agent-a", candidate_actions=cands)
    assert rec["mode"] == "act" and rec["suggested_action"] == "fix:rerun_job"
    plan = rec["plan"]
    assert plan[0] == "fix:rerun_job"
    assert len(plan) == len(set(plan)) <= act.PLAN_MAX
    assert set(plan[1:]) <= set(pr["untried"]), "second and third choices are untried actions, not known failures"
    assert "fix:fix_code" not in plan
    # Two agents on the same problem start their exploration at different actions.
    other = act.recommend(pr, agent_id="ci-agent-b", candidate_actions=cands)["plan"]
    assert other[0] == plan[0] and other[1:] != plan[1:]
    # Restricted to the candidates the caller still has (a retry after rerun_job
    # failed): the plan is over those, and the recommendation's action leads.
    left = [a for a in cands if a != "fix:rerun_job"]
    rec = act.recommend(pr, agent_id="ci-agent-a", candidate_actions=left)
    assert "fix:rerun_job" not in rec["plan"] and rec["plan"]
    # Under a turned winner the explore pick leads and the turned action is not
    # ahead of the untried ones.
    two = dict(pr, tried=[_prior("fix:rerun_job", 3, 5, ["lost", "lost", "won"])] + pr["tried"][1:])
    rec = act.recommend(two, agent_id="ci-agent-a", candidate_actions=cands)
    assert rec["mode"] == "explore" and rec["plan"][0] == rec["suggested_action"]
    assert rec["plan"].index("fix:rerun_job") > 2 if "fix:rerun_job" in rec["plan"] else True
    # No priors, no plan; a bare untried list still yields one.
    assert act.plan_actions(None) == []
    assert act.plan_actions({"tried": [], "untried": ["a", "b"]}, agent_id="x") == ["a", "b"] or \
        act.plan_actions({"tried": [], "untried": ["a", "b"]}, agent_id="x") == ["b", "a"]


def test_render_priors_gives_the_order_and_what_not_to_retry() -> None:
    cands = ["fix:rerun_job", "fix:fix_code", "fix:edit_generated_file", "fix:regen_migrations"]
    pr = {"tried": [_prior("fix:rerun_job", 3, 4, ["lost", "won", "won"]),
                    _prior("fix:fix_code", 0, 2, ["lost", "lost"])],
          "untried": ["fix:edit_generated_file", "fix:regen_migrations"]}
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    text = act.render_priors(pr, rec)
    assert "Do not spend an attempt on: fix:fix_code (0/2)" in text
    # The winner is the order; the untried tail is a list for the agent's
    # judgement, in the candidates' order, not the plan's rotation.
    assert "Try fix:rerun_job first. Do not repeat an action that failed on this task." in text
    assert "Untried on tasks like this: fix:edit_generated_file, fix:regen_migrations — the record cannot order these" in text
    assert text.index("Recommendation:") < text.index("Try fix:rerun_job first")
    # A turned winner is named as such in the avoid line.
    two = dict(pr, tried=[_prior("fix:rerun_job", 3, 5, ["lost", "lost", "won"])] + pr["tried"][1:])
    text = act.render_priors(two, act.recommend(two, agent_id="a", candidate_actions=cands))
    assert "fix:rerun_job (3/5, stopped working)" in text
    # Escalate: nothing to try, no order given.
    firm = {"tried": [_prior("fix:a", 0, 3, ["lost"] * 3), _prior("fix:b", 0, 3, ["lost"] * 3)], "untried": []}
    rec = act.recommend(firm, agent_id="a", candidate_actions=["fix:a", "fix:b"])
    assert rec["mode"] == "escalate"
    assert "Do not spend" not in act.render_priors(firm, rec)
    # The order names candidates only, whether computed here or sent by the server.
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    text = act.render_priors(pr, rec, candidate_actions=["fix:edit_generated_file", "fix:regen_migrations"])
    assert "Try " not in text and "fix:rerun_job" not in text.split("Untried on tasks like this:")[1]
    sent = dict(rec, plan=["fix:rerun_job", "fix:edit_generated_file", "fix:regen_migrations"])
    text = act.render_priors(pr, sent, candidate_actions=["fix:edit_generated_file", "fix:regen_migrations"])
    assert "Untried on tasks like this: fix:edit_generated_file, fix:regen_migrations —" in text
    # The recommendation line does not name a barred action either, and the
    # untried list is drawn from the candidates: a retry's text names nothing
    # it has ruled out.
    assert rec["suggested_action"] == "fix:rerun_job"
    assert "Recommendation: act." in text and "-> fix:rerun_job" not in text
    with_untried = dict(pr, untried=["fix:rerun_job", "fix:regen_migrations"])
    text = act.render_priors(with_untried, rec, candidate_actions=["fix:edit_generated_file", "fix:regen_migrations"])
    assert "Untried on tasks like this: fix:regen_migrations —" in text
    # No recommendation, or an abstain: the record is shown, no order is given.
    assert "Try " not in act.render_priors(pr, None) and "Untried" not in act.render_priors(pr, None)
    assert "Do not spend" not in act.render_priors(pr, None)
    assert "Try " not in act.render_priors(pr, {"mode": "abstain", "why": "pooled"})
    assert "Tried on similar tasks here" in act.render_priors(pr, None)


def test_render_priors_states_the_contradiction_once() -> None:
    pr = act.aggregate_priors(_pooled_rows())
    text = act.render_priors(pr, act.recommend(pr, agent_id="x", abstain=True))
    assert "On a near-identical task here" not in text, "two contradicting lines read as a coin toss"
    assert text.count("alternating over time") == 1
    assert "Recommendation: abstain." in text
    # No recommendation given (caller without abstain): the warning still shows, once.
    text = act.render_priors(pr, None)
    assert text.count("alternating over time") == 1 and "Recommendation" not in text


def test_render_priors_names_the_near_identical_contrast() -> None:
    pr = act.aggregate_priors(act.neighbourhood_weights([_contrast_row("resolve:a", "resolve:b", sim=0.95)]))
    text = act.render_priors(pr, act.recommend(pr, agent_id="x"))
    assert "On a near-identical task here resolve:a failed and resolve:b resolved it." in text
    assert "Recommendation: act -> resolve:b" in text
    far = act.aggregate_priors(act.neighbourhood_weights([
        _row([("resolve:c", True)], sim=0.95), _contrast_row("resolve:a", "resolve:b", sim=0.89),
    ]))
    assert "near-identical" not in act.render_priors(far, None)


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


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _stub_similar(adapter, rows):
    """Priors from the outcomes nearest the query — the source that is about
    this kind of task, and the one a shift may send exploring. Needs an
    embedder for the query vector; the semantic channel itself stays off
    because these tests run without the async adapter."""
    from amfs_http import server

    adapter.similar_outcomes = lambda entity_path, embedding, **kw: list(rows)  # type: ignore[attr-defined]
    adapter.action_stats = lambda entity_path, **kw: []  # type: ignore[attr-defined]
    server._get_server_embedder = lambda: _StubEmbedder()  # restored by the server_mem monkeypatch


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
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    body, meta = _priors_meta(client)
    assert all(e.get("key") != "fix-a" for e in body if not e.get("_meta")), "discredited rule left the head"
    assert meta["regime_shift"] is True
    assert meta["regime_shift_scope"] == "query"
    assert meta["recommendation"]["mode"] == "explore"
    assert meta["recommendation"]["suggested_action"] == "resolve:b"
    assert "regime shift" in meta["recommendation"]["why"]


def test_the_shift_still_fires_when_the_confidence_gate_hid_the_discredited_rule(client, server_mem) -> None:
    """The candidate fetch honours min_confidence; a discredited rule sits below
    the discredit threshold, so a retrieve gated there (the benchmark's setting)
    never saw it — and the two-failure signal went with it. The flag reads a
    separate, ungated fetch of the query-matched discredited rows."""
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    entry = server_mem._adapter.read("acme/support", "fix-a")
    assert entry.discredited_at is not None and entry.confidence < 0.5

    body, meta = _priors_meta(client, min_confidence=0.5)
    assert all(e.get("key") != "fix-a" for e in body if not e.get("_meta"))
    assert meta["regime_shift"] is True
    assert meta["regime_shift_scope"] == "query", "the hidden rule matches this query"
    assert meta["recommendation"]["mode"] == "explore"


def test_the_below_gate_read_is_entity_wide_not_a_rerun_of_the_query(client, server_mem) -> None:
    """The rule that stopped working need not share a word with this query, and
    a rule validated over months is old by write time. The entity-wide flag
    reads the entity's discredited rows, whatever they say and whenever they
    were written — the same scope the briefing uses.

    The flag is reported; it does not steer the recommendation. The rule that
    shifted is about a stuck queue and this query is about a declined card, so
    the winner for declined cards is still the recommendation. Sending every
    class of task on the entity to explore past its own winning action was
    measured in grid v3 at 14% success on the explores it produced."""
    server_mem.write("acme/support", "fix-old", "rotate the ingest worker on a stuck queue", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-old", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    assert server_mem._adapter.read("acme/support", "fix-old").discredited_at is not None

    _, meta = _priors_meta(client, min_confidence=0.5)  # query: "card declined" — no overlap with fix-old
    assert meta["regime_shift"] is True
    assert meta["regime_shift_scope"] == "entity"
    assert meta["recommendation"]["mode"] == "act"
    assert meta["recommendation"]["suggested_action"] == "resolve:a"


def _contrast_outcome_row(failed: str, resolved: str, *, agent="a1", ref="o-contrast"):
    row = _row([], agent=agent)
    row["actions_taken"] = [
        {"action_key": failed, "success": False, "attempt": 1},
        {"action_key": resolved, "success": True, "attempt": None},
    ]
    row["outcome_ref"] = ref
    return row


def test_a_contrast_in_the_priors_acts_on_the_resolver_and_names_it_on_the_avoid_row(
    client, server_mem
) -> None:
    """One nearby fail-then-succeed outcome: the recommendation is the action
    that resolved it, the priors carry the pair, and the avoided rule — the one
    the failed attempt acted on — says what resolved the task instead. The
    action reaches the avoid row through the lesson for the same outcome, so
    it lands on the rule that outcome named and on no other."""
    from amfs_core import evidence as ev

    _stub_similar(server_mem._adapter, [_contrast_outcome_row("resolve:a", "resolve:b")])
    _outcomes(server_mem, "fix-a", OutcomeType.FAILURE, OutcomeType.FAILURE)
    _outcomes(server_mem, "fix-b", OutcomeType.FAILURE, OutcomeType.FAILURE)
    # The lesson for o-contrast: fix-a misled the attempt. No resolved_action
    # of its own (a repair-loop pointer), so the action joins from the priors.
    server_mem.write(
        "acme/support", ev.contrast_lesson_key("o-contrast"),
        {"kind": "contrast", "outcome_ref": "o-contrast", "avoid": ["acme/support/fix-a"],
         "resolved_with": [], "task_excerpt": "card declined"},
        confidence=0.8,
    )
    server_mem._read_tracker.clear()

    body, meta = _priors_meta(client, include_avoid=True, compact=True)
    assert meta["recommendation"]["mode"] == "act"
    assert meta["recommendation"]["suggested_action"] == "resolve:b"
    assert meta["recommendation"]["contrast"] == {
        "failed": ["resolve:a"], "resolved_with": "resolve:b", "outcome_ref": "o-contrast",
    }
    assert meta["priors"]["contrasts"][0]["resolved_with"] == "resolve:b"
    avoid = {e["key"]: e for e in body if e.get("_avoid")}
    assert sorted(avoid) == ["fix-a", "fix-b"]
    assert avoid["fix-a"]["_breakdown"]["resolved_with_action"] == "resolve:b"
    assert avoid["fix-a"]["_breakdown"]["locally_discredited"] is False
    assert avoid["fix-a"]["value"].startswith("discredited; last failure ")
    assert avoid["fix-a"]["value"].endswith("resolved instead with resolve:b")
    assert "value_truncated" not in avoid["fix-a"]
    # fix-b was not named by that outcome: it does not claim the fix.
    assert avoid["fix-b"]["_breakdown"]["resolved_with_action"] is None
    assert "resolved instead" not in avoid["fix-b"]["value"]
    # The elements keep their order: hits, avoid rows, then the meta element.
    kinds = ["meta" if e.get("_meta") else "avoid" if e.get("_avoid") else "hit" for e in body]
    assert kinds == sorted(kinds, key=["hit", "avoid", "meta"].index)


def test_under_a_query_shift_contested_alternatives_are_pruned_behind_the_leader(
    client, server_mem
) -> None:
    """A rule the query is about has stopped working (query-scoped shift). The
    alternatives the record has already marked against on tasks like this are
    the rows grid v4 found in failing contexts; behind the leader they go."""
    server_mem.write("acme/support", "fix-c", "card declined: retry the charge tomorrow", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    _outcomes(server_mem, "fix-b", *([OutcomeType.SUCCESS] * 3))
    _outcomes(server_mem, "fix-c", OutcomeType.SUCCESS, OutcomeType.SUCCESS, OutcomeType.SUCCESS, OutcomeType.FAILURE)
    assert server_mem._adapter.read("acme/support", "fix-c").evidence_status == "contested"

    body, meta = _priors_meta(client, adaptive_k=True)
    assert meta["regime_shift_scope"] == "query"
    hits = [e["key"] for e in body if not e.get("_meta")]
    assert hits == ["fix-b"], hits

    # Without adaptive k the contested alternative stays.
    body, _ = _priors_meta(client, adaptive_k=False)
    assert "fix-c" in [e["key"] for e in body if not e.get("_meta")]


def test_a_rescued_hit_is_not_pruned_under_the_shift(client, server_mem, monkeypatch) -> None:
    """A rescued entry carries the ``contested`` label too, but it is in the
    list because its local record says it works on tasks like this — the
    opposite of what the label means on a pooled record. The shift pruning
    must leave it where the rescue put it."""
    from amfs_http import server

    from tests.unit.test_retrieve_evidence import _AsyncShim

    server_mem.write("acme/support", "fix-c", "card declined: retry the charge tomorrow", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    monkeypatch.setattr(server, "_async_adapter", _AsyncShim(server_mem._adapter))
    monkeypatch.setattr(
        server_mem._adapter, "evidence_near",
        lambda keys, vec, **kw: {
            "acme/support/fix-c": {"success": 3.0, "failure": 0.0, "n": 3, "best_similarity": 0.9,
                                   "recent": [{"success": True}] * 3},
        },
        raising=False,
    )
    # fix-a: the shifted rule. fix-b: untested, so it leads without the
    # validated-leader pass (step 11) firing — what is under test is the shift
    # pass (11b). fix-c: discredited by the pooled record, working on tasks
    # like this one -> rescued, and labelled ``contested``.
    _outcomes(server_mem, "fix-a", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)
    _outcomes(server_mem, "fix-c", OutcomeType.FAILURE, OutcomeType.FAILURE)
    assert server_mem._adapter.read("acme/support", "fix-c").discredited_at is not None

    body, meta = _priors_meta(client, adaptive_k=True, include_avoid=True)
    assert meta["regime_shift_scope"] == "query"
    hits = {e["key"]: e for e in body if not e.get("_meta") and not e.get("_avoid")}
    assert "fix-c" in hits, sorted(hits)
    assert hits["fix-c"]["_rescued"] is True and hits["fix-c"]["evidence_status"] == "contested"
    assert list(hits)[0] == "fix-b"


def test_the_off_query_rule_fires_the_shift_at_the_default_gate_too(client, server_mem) -> None:
    """``min_confidence`` defaults to 0, so the documented include_priors call
    has no gate to lift — the entity-wide read must still run, or a rule that
    stopped working but shares no words with the query never flags."""
    server_mem.write("acme/support", "fix-old", "rotate the ingest worker on a stuck queue", confidence=0.8)
    server_mem._read_tracker.clear()
    _stub_similar(server_mem._adapter, [_row([("resolve:a", True)], agent=f"a{i}") for i in range(3)])
    _outcomes(server_mem, "fix-old", *([OutcomeType.SUCCESS] * 8), OutcomeType.FAILURE, OutcomeType.FAILURE)

    _, meta = _priors_meta(client)  # default min_confidence=0.0
    assert meta["regime_shift"] is True
    assert meta["regime_shift_scope"] == "entity"
    assert meta["recommendation"]["mode"] == "act"


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


def test_lessons_for_the_exact_situation_reorder_the_plan() -> None:
    """Ops-queue CI, 2026-09-22, PR #45: the audit winner had turned and the
    plan's second action was bump_dependency — it wins on the unpinned audit
    class the neighbourhood also held — while the lesson for the pinned class
    said it had failed there. The situation-exact record outranks the pooled
    one: a 'did not work' claim goes to the back, a 'worked' claim to the
    front (after an act's own action), discredited lessons say nothing, and
    a 'worked' claim about an action whose record here has stopped working
    is not moved up."""
    pr = {"tried": [_prior("fix:add_audit_exception", 9, 10, ["lost", "won", "won"]),
                    _prior("fix:bump_dependency", 1, 2, ["won", "lost"]),
                    _prior("fix:edit_generated_file", 0, 1, ["lost"])],
          "untried": ["fix:fix_code", "fix:run_formatter", "fix:regen_migrations"]}
    cands = ["fix:add_audit_exception", "fix:bump_dependency", "fix:edit_generated_file",
             "fix:fix_code", "fix:run_formatter", "fix:regen_migrations"]
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    assert rec["mode"] == "act" and rec["suggested_action"] == "fix:add_audit_exception"
    assert rec["plan"][1] == "fix:bump_dependency"
    failed = [{"action": "fix:bump_dependency", "worked": False, "evidence_status": "validated"}]
    ranked = act.rank_by_lessons(rec["plan"], failed, priors=pr, recommendation=rec)
    assert ranked[0] == "fix:add_audit_exception" and ranked[-1] == "fix:bump_dependency"
    assert set(ranked) == set(rec["plan"])
    # Through recommend and render_priors alike, the same order.
    rec2 = act.recommend(pr, agent_id="a", candidate_actions=cands, lessons=failed)
    assert rec2["plan"] == ranked
    text = act.render_priors(pr, dict(rec, plan=None), candidate_actions=cands, lessons=failed)
    # The barred action is not named in the order at all: the lesson for this
    # situation outranks the pooled record that has it winning.
    assert "Try fix:add_audit_exception first." in text and "then fix:bump_dependency" not in text
    # A 'worked' claim goes to the front — after the act's own action.
    worked = [{"action": "fix:fix_code", "worked": True, "evidence_status": "untested"}]
    assert act.rank_by_lessons(rec["plan"], worked, priors=pr, recommendation=rec)[:2] == [
        "fix:add_audit_exception", "fix:fix_code"]
    # In an explore there is no action of its own to keep first: the claim leads.
    explore = dict(rec, mode="explore", suggested_action="fix:run_formatter")
    assert act.rank_by_lessons(rec["plan"], worked, priors=pr, recommendation=explore)[0] == "fix:fix_code"
    # Discredited lessons and lessons about a turned record say nothing.
    assert act.rank_by_lessons(rec["plan"], [dict(failed[0], evidence_status="discredited")],
                               priors=pr, recommendation=rec) == rec["plan"]
    turned = dict(pr, tried=[_prior("fix:add_audit_exception", 9, 11, ["lost", "lost", "won"])] + pr["tried"][1:])
    claim = [{"action": "fix:add_audit_exception", "worked": True, "evidence_status": "contested"}]
    plan = ["fix:fix_code", "fix:add_audit_exception"]
    assert act.rank_by_lessons(plan, claim, priors=turned, recommendation={"mode": "explore"}) == plan
    # Both claims about one action: the positive wins (the lesson flipped and back).
    both = failed + [{"action": "fix:bump_dependency", "worked": True}]
    assert act.rank_by_lessons(rec["plan"], both, priors=pr, recommendation=rec)[1] == "fix:bump_dependency"
    # Nothing to say: the plan is returned as given, never grown or shrunk.
    assert act.rank_by_lessons(rec["plan"], [], priors=pr, recommendation=rec) == rec["plan"]
    assert act.rank_by_lessons([], failed) == []


def test_the_server_reorders_the_plan_by_the_lessons_about_this_task(client, server_mem) -> None:
    """The record over the neighbourhood has resolve:a winning and resolve:b
    with one win, so the plan is a then b then the untried c. A lesson filed
    for this exact situation says b did not work: the plan the server sends
    puts b last. A lesson for another situation changes nothing."""
    from amfs_core.lessons import make_lesson

    _stub_stats(server_mem._adapter, [
        _row([("resolve:a", True)], agent="a1"), _row([("resolve:a", True)], agent="a2"),
        _row([("resolve:b", True)], agent="a3"),
    ])
    server_mem.write("acme/support", "learned-card-declined",
                     make_lesson("card declined at checkout", "resolve:b", False, "the card is fine"))
    server_mem.write("acme/support", "learned-export",
                     make_lesson("export timed out for a large workspace", "resolve:c", True))
    server_mem._read_tracker.clear()

    cands = ["resolve:a", "resolve:b", "resolve:c"]
    body, meta = _priors_meta(client, query="card declined at checkout", candidate_actions=cands)
    assert "learned-card-declined" in {e.get("key") for e in body if not e.get("_meta")}
    plan = meta["recommendation"]["plan"]
    assert plan[0] == "resolve:a" and plan[-1] == "resolve:b" and "resolve:c" in plan
    # The lesson is a hit but its situation is not in this query: the record's order stands.
    _, meta = _priors_meta(client, query="declined", candidate_actions=cands)
    assert meta["recommendation"]["plan"][:2] == ["resolve:a", "resolve:b"]
    # Declared situation: compared exactly (case and spacing folded), the text is not consulted.
    _, meta = _priors_meta(client, query="declined", situation="  Card declined AT checkout ",
                           candidate_actions=cands)
    assert meta["recommendation"]["plan"][-1] == "resolve:b"
    _, meta = _priors_meta(client, query="declined", situation="webhook late", candidate_actions=cands)
    assert meta["recommendation"]["plan"][:2] == ["resolve:a", "resolve:b"]
    # A lesson retrieve avoids — it failed on tasks like this one, twice — says
    # nothing: its "worked" claim must not promote the action it was falsified on.
    server_mem.write("acme/support", "learned-card-declined-c",
                     make_lesson("card declined at checkout", "resolve:c", True, "try the other card"))
    _outcomes(server_mem, "learned-card-declined-c", OutcomeType.FAILURE, OutcomeType.FAILURE)
    body, meta = _priors_meta(client, query="card declined at checkout", candidate_actions=cands)
    assert all(e.get("key") != "learned-card-declined-c" for e in body if not e.get("_meta"))
    assert meta["recommendation"]["plan"][0] == "resolve:a" and meta["recommendation"]["plan"][-1] == "resolve:b"


# ── the situation's own record ─────────────────────────────────────────────


def _sit_row(situation, actions, *, outcome="success", agent="a1", days_ago=0, sim=1.0):
    row = _row(actions, agent=agent, days_ago=days_ago, sim=sim)
    row["situation"] = situation
    row["outcome_type"] = outcome
    return row


def test_a_situation_with_no_record_abstains_instead_of_planning_from_its_neighbours() -> None:
    """Ops-queue CI #5 in two runs: the UI snapshot class's first PR was told
    ``act: fix_code`` from the backend snapshot class's single win — the
    opposite of its own fix — and cost three CI runs where the same model
    with no memory cost none. With the situation's own record empty, the
    recommendation abstains, the strength is at most ``thin`` whatever a
    neighbour's lesson is rated, and the text shows the neighbourhood as
    another kind of task."""
    cands = ["fix:fix_code", "fix:update_snapshots", "fix:rerun_job"]
    pr = {"tried": [], "untried": cands, "source": "similar_outcomes",
          "situation_record": {"situation": "jest snapshot · UI PR", "outcomes": 0},
          "nearby": {"outcomes": 1, "by_situation": [
              {"situation": "jest snapshot · backend-only PR", "outcomes": 1,
               "tried": [{"action_key": "fix:fix_code", "won": 1, "lost": 0, "n": 1},
                         {"action_key": "fix:update_snapshots", "won": 0, "lost": 1, "n": 1}]}]}}
    assert act.recommend(pr, agent_id="a", candidate_actions=cands) is None
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands, abstain=True)
    assert rec["mode"] == "abstain" and rec["suggested_action"] is None
    assert rec["situation_record"] == "none" and "plan" not in rec
    assert act.guidance_strength(pr, ["validated"]) == "thin"
    assert act.guidance_strength(dict(pr, nearby=None), []) == "none"
    text = act.render_priors(pr, rec, candidate_actions=cands)
    assert "No outcome recorded for this situation yet." in text
    assert "On nearby kinds of task (not this situation) — “jest snapshot · backend-only PR”: fix:fix_code 1/1; fix:update_snapshots 0/1" in text
    assert "Not yet tried on this situation: fix:fix_code, fix:update_snapshots, fix:rerun_job" in text
    assert "Try " not in text and "Do not spend" not in text
    # The same neighbourhood, pooled (no situation declared, or none recorded):
    # the old reading stands, so callers without situations lose nothing.
    pooled = {"tried": [_prior("fix:fix_code", 1, 1, ["won"]), _prior("fix:update_snapshots", 0, 1, ["lost"])],
              "untried": ["fix:rerun_job"], "source": "similar_outcomes"}
    assert act.recommend(pooled, agent_id="a", candidate_actions=cands)["mode"] == "act"


def test_one_unsolved_loss_on_the_situations_own_record_turns_the_action() -> None:
    """Ops-queue CI #45/#48 and support #49/#53: after a change, the class's
    validated fix failed on a task nothing then solved, and the next task of
    the class was still told ``act`` on it — three runs wasted twice. On the
    situation's own record one such loss turns the action: the mode is
    ``explore``, the turned action is kept second in the plan as the hedge,
    and the text says to make an own pick first. A loss inside a task another
    action solved is a contrast, not a turn; and the same loss on a pooled
    record is forgiven as before."""
    cands = ["fix:add_audit_exception", "fix:bump_dependency", "fix:fix_code", "fix:rerun_job"]
    rows = [
        _sit_row("pip-audit · pinned", [("fix:add_audit_exception", False), ("fix:bump_dependency", False)],
                 outcome="failure"),
        _sit_row("pip-audit · pinned", [("fix:add_audit_exception", True)], days_ago=1),
        _sit_row("pip-audit · pinned", [("fix:add_audit_exception", True)], days_ago=2),
        _sit_row("pip-audit · pinned", [("fix:bump_dependency", False), ("fix:add_audit_exception", True)], days_ago=3),
    ]
    pr = act.aggregate_priors(rows, candidate_actions=cands, situation_exact=True)
    pr["situation_record"] = {"situation": "pip-audit · pinned", "outcomes": 4}
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["fix:add_audit_exception"]["situation_exact"] is True
    assert by["fix:add_audit_exception"]["last_unsolved"] is True
    assert by["fix:add_audit_exception"]["won"] == 3 and by["fix:add_audit_exception"]["last_3"][0] == "lost"
    assert act.turned_unsolved(by["fix:add_audit_exception"]) and not act.turned(by["fix:add_audit_exception"])
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    assert rec["mode"] == "explore"
    assert rec["suggested_action"] in ("fix:fix_code", "fix:rerun_job")
    assert rec["plan"][1] == "fix:add_audit_exception"
    assert "newest take lost on a task nothing solved" in rec["why"]
    text = act.render_priors(pr, rec, candidate_actions=cands)
    assert "Recommendation: explore." in text and "-> fix:" not in text.split("Recommendation")[1].split("\n")[0]
    assert "Make your own pick first; if it fails, try fix:add_audit_exception next" in text
    assert "Do not spend an attempt on: fix:bump_dependency (0/2)" in text
    assert "Untried on this situation: fix:fix_code, fix:rerun_job — the record cannot order these" in text
    assert "stopped working" not in text
    # A loss that another action then resolved is a contrast, not a turn.
    solved = act.aggregate_priors([
        _sit_row("s", [("fix:a", False), ("fix:b", True)]),
        _sit_row("s", [("fix:a", True)], days_ago=1),
    ], situation_exact=True)
    a = next(t for t in solved["tried"] if t["action_key"] == "fix:a")
    assert a["last_unsolved"] is False and not act.turned_unsolved(a)
    # Pooled: the same rows without the mark keep the one-loss forgiveness.
    pooled = act.aggregate_priors(rows, candidate_actions=cands)
    assert not act.turned_unsolved(next(t for t in pooled["tried"] if t["action_key"] == "fix:add_audit_exception"))
    assert act.recommend(pooled, agent_id="a", candidate_actions=cands)["mode"] == "act"


def test_an_explore_names_its_pick_as_an_assignment_not_an_instruction() -> None:
    """Grid v3: an explore that was followed won 14% of the time, one that
    was ignored 67%; the ops-queue demo lost three tickets to a hash-picked
    action on a class the model got right on its own. The pick stays in the
    payload for a fleet to spread out over; the prose says what it is and
    hands the untried to the agent's judgement."""
    cands = ["resolve:a", "resolve:b", "resolve:c"]
    pr = {"tried": [_prior("resolve:a", 0, 2, ["lost", "lost"])], "untried": ["resolve:b", "resolve:c"]}
    rec = act.recommend(pr, agent_id="a", candidate_actions=cands)
    assert rec["mode"] == "explore" and rec["suggested_action"] in ("resolve:b", "resolve:c")
    assert "exploration assignment if you have no better guess" in rec["why"]
    assert "use your own judgement" in rec["why"]
    text = act.render_priors(pr, rec, candidate_actions=cands)
    assert "Recommendation: explore." in text
    assert "Try " not in text
    assert "Untried on tasks like this: resolve:b, resolve:c — the record cannot order these" in text


def test_the_server_serves_the_situations_own_record_and_its_neighbours_apart(client, server_mem) -> None:
    """Rows carry the situation their run declared. With one declared on the
    request, the priors are that situation's outcomes; the rest of the
    neighbourhood is reported per situation, and a situation with no
    outcomes abstains. Without a declared situation — or when no row carries
    one — the block is pooled as before."""
    rows = [
        _sit_row("jest snapshot · backend-only PR", [("fix:update_snapshots", False), ("fix:fix_code", True)]),
        _sit_row("jest snapshot · UI PR", [("fix:fix_code", False), ("fix:rerun_job", False)], outcome="failure",
                 agent="a2", sim=0.98),
    ]
    _stub_similar(server_mem._adapter, rows)
    cands = ["fix:fix_code", "fix:update_snapshots", "fix:rerun_job"]
    body = {"query": "card declined", "entity_path": "acme/support", "include_priors": True,
            "agent_id": "a9", "candidate_actions": cands, "abstain": True}
    # The UI class: its own record is two losses; the backend class is nearby.
    meta = client.post("/api/v1/retrieve", json=dict(body, situation="Jest Snapshot · UI PR")).json()[-1]
    pr = meta["priors"]
    assert pr["situation_record"] == {"situation": "Jest Snapshot · UI PR", "outcomes": 1}
    assert {t["action_key"] for t in pr["tried"]} == {"fix:fix_code", "fix:rerun_job"}
    assert all(t["situation_exact"] for t in pr["tried"])
    assert pr["untried"] == ["fix:update_snapshots"]
    assert pr["nearby"]["outcomes"] == 1
    assert pr["nearby"]["by_situation"][0]["situation"] == "jest snapshot · backend-only PR"
    assert {t["action_key"]: t["won"] for t in pr["nearby"]["by_situation"][0]["tried"]} == {
        "fix:fix_code": 1, "fix:update_snapshots": 0}
    assert meta["recommendation"]["mode"] == "explore"
    assert meta["recommendation"]["suggested_action"] == "fix:update_snapshots"
    # A situation nothing has ended on: abstain, and not ``strong``.
    meta = client.post("/api/v1/retrieve", json=dict(body, situation="pip-audit · pinned")).json()[-1]
    assert meta["priors"]["situation_record"]["outcomes"] == 0 and meta["priors"]["tried"] == []
    assert meta["recommendation"]["mode"] == "abstain" and meta["recommendation"]["situation_record"] == "none"
    assert meta["guidance_strength"] == "thin"
    assert len(meta["priors"]["nearby"]["by_situation"]) == 2
    # No situation on the request: pooled, as before.
    meta = client.post("/api/v1/retrieve", json=body).json()[-1]
    assert "situation_record" not in meta["priors"]
    assert {t["action_key"] for t in meta["priors"]["tried"]} == {"fix:fix_code", "fix:update_snapshots", "fix:rerun_job"}
    # Rows without situations (older clients): pooled too, even with one declared.
    _stub_similar(server_mem._adapter, [dict(r, situation=None) for r in rows])
    meta = client.post("/api/v1/retrieve", json=dict(body, situation="jest snapshot · UI PR")).json()[-1]
    assert "situation_record" not in meta["priors"]
