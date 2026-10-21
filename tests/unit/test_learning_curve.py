"""The number: ``amfs_core.learning`` and ``GET /api/v1/learning``.

Success over sessions, cut into episodes, with recovery after each regime
shift and the share of outcomes decided outside the agent.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime, timedelta

import pytest
from amfs_core import learning

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _rows(flags: str, *, start: datetime = T0, verified: str = "") -> list[dict]:
    """One outcome per character: ``s`` success, ``f`` failure, an hour apart.
    ``verified`` marks the same positions with ``v`` for a CI-decided outcome."""
    out = []
    for i, ch in enumerate(flags):
        out.append(
            {
                "at": start + timedelta(hours=i),
                "outcome_type": "success" if ch == "s" else "failure",
                "agent_id": "a",
                "verified_by": "ci" if i < len(verified) and verified[i] == "v" else None,
            }
        )
    return out


class TestCurve:
    def test_an_empty_window_is_an_empty_curve(self) -> None:
        out = learning.learning_curve([])
        assert out["sessions"] == 0
        assert out["success_rate"] is None
        assert out["curve"] == [] and out["first"] is None and out["change"] is None
        assert out["recovery"]["shifts"] == 0

    def test_the_headline_is_first_window_against_latest_window(self) -> None:
        # 10 sessions at 40%, then 10 at 90%.
        out = learning.learning_curve(_rows("ssffffsfff" + "ssssssssfs"), window=10, episodes=2)
        assert out["sessions"] == 20
        assert out["first"] == {"sessions": 10, "success_rate": 0.3}
        assert out["latest"] == {"sessions": 10, "success_rate": 0.9}
        assert out["change"] == pytest.approx(0.6)
        assert [b["success_rate"] for b in out["curve"]] == [0.3, 0.9]
        assert [b["episode"] for b in out["curve"]] == [1, 2]

    def test_episodes_are_equal_sized_bins_in_time_order(self) -> None:
        out = learning.learning_curve(_rows("s" * 7 + "f" * 7), window=3, episodes=4)
        sizes = [b["sessions"] for b in out["curve"]]
        assert sizes == [4, 4, 3, 3]
        assert out["curve"][0]["success_rate"] == 1.0
        assert out["curve"][-1]["success_rate"] == 0.0
        assert out["curve"][0]["from"] < out["curve"][-1]["to"]

    def test_too_few_sessions_have_no_change_yet(self) -> None:
        out = learning.learning_curve(_rows("sfs"), window=10)
        assert out["sessions"] == 3
        assert out["first"] is None and out["latest"] is None
        assert out["change"] is None
        # Five sessions would give halves of two: under min_sessions, no headline.
        assert learning.learning_curve(_rows("sfsfs"), window=10)["change"] is None

    def test_the_windows_never_overlap(self) -> None:
        # 13 sessions under a window of 10 used to compare slices sharing
        # seven points, which dilutes the delta; the halves shrink instead.
        out = learning.learning_curve(_rows("ffffff" + "s" + "ssssss"), window=10)
        assert out["first"] == {"sessions": 6, "success_rate": 0.0}
        assert out["latest"] == {"sessions": 6, "success_rate": 1.0}
        assert out["change"] == pytest.approx(1.0)
        # Six sessions is the first point with two halves of min_sessions.
        out = learning.learning_curve(_rows("fffsss"), window=10)
        assert out["first"]["sessions"] == 3 and out["change"] == pytest.approx(1.0)
        # With two full windows the halves stop growing.
        out = learning.learning_curve(_rows("f" * 15 + "s" * 15), window=10)
        assert out["first"]["sessions"] == 10 and out["latest"]["sessions"] == 10

    def test_rows_arrive_in_any_order_and_unknown_types_are_dropped(self) -> None:
        rows = _rows("sf")
        rows.append({"at": T0 + timedelta(days=1), "outcome_type": "shrug", "agent_id": "a"})
        out = learning.learning_curve(reversed(rows), window=3)
        assert out["sessions"] == 2
        assert out["span"]["from"] == T0.isoformat()

    def test_verified_share_counts_outcomes_decided_outside_the_agent(self) -> None:
        out = learning.learning_curve(_rows("ssff", verified="v.v."))
        assert out["verified_share"] == 0.5

    def test_objects_work_as_well_as_mappings(self) -> None:
        rows = [
            types.SimpleNamespace(committed_at=T0, outcome_type="success", verified_by="ci"),
            types.SimpleNamespace(committed_at=T0 + timedelta(hours=1), outcome_type="failure"),
        ]
        out = learning.learning_curve(rows)
        assert (out["sessions"], out["successes"], out["verified_share"]) == (2, 1, 0.5)


class TestRecovery:
    def test_three_sessions_to_recover_after_the_rule_changed(self) -> None:
        # 10 sessions at 80%, the rule is discredited, then f f s s s ...: the
        # rolling rate over the post-shift sessions reaches 0.8 at the 10th
        # (8 of 10). Before that: at k=5 it is 3/5, k=8 is 6/8 = 0.75.
        before = "ssssssssff"
        after = "ff" + "s" * 8
        rows = _rows(before + after)
        shift_at = T0 + timedelta(hours=len(before))
        out = learning.learning_curve(
            rows, shifts=[{"entity_path": "svc/mod", "key": "rule", "at": shift_at}], window=10
        )
        rec = out["recovery"]
        assert rec["shifts"] == 1 and rec["measured"] == 1 and rec["recovered"] == 1
        assert rec["median_sessions_to_recover"] == 10
        event = rec["events"][0]
        assert event["bar"] == 0.8
        assert event["sessions_to_recover"] == 10
        assert event["entity_path"] == "svc/mod" and event["key"] == "rule"

        # A faster recovery: the agent read the lesson and succeeded straight away.
        rows = _rows(before + "sss")
        out = learning.learning_curve(
            rows, shifts=[{"entity_path": "svc/mod", "key": "rule", "at": shift_at}], window=10
        )
        assert out["recovery"]["median_sessions_to_recover"] == 3

    def test_a_shift_the_agent_has_not_recovered_from_is_open(self) -> None:
        before = "ssssssssss"
        rows = _rows(before + "fffff")
        shift_at = T0 + timedelta(hours=len(before))
        out = learning.learning_curve(rows, shifts=[{"key": "rule", "at": shift_at}], window=10)
        rec = out["recovery"]
        assert rec["measured"] == 1 and rec["recovered"] == 0
        assert rec["median_sessions_to_recover"] is None
        assert rec["open"] == [
            {"entity_path": None, "key": "rule", "at": shift_at.isoformat(), "after": 5}
        ]

    def test_a_shift_with_too_little_on_either_side_is_not_measured(self) -> None:
        rows = _rows("ss" + "sss")
        out = learning.learning_curve(
            rows, shifts=[{"key": "rule", "at": T0 + timedelta(hours=2)}], window=10
        )
        rec = out["recovery"]
        assert rec["shifts"] == 1 and rec["measured"] == 0
        assert rec["events"][0]["measured"] is False
        assert rec["events"][0]["before"] == 2
        assert rec["events"][0]["reason"] == "short_before"
        out = learning.learning_curve(
            _rows("sssss" + "ss"), shifts=[{"key": "rule", "at": T0 + timedelta(hours=5)}], window=10
        )
        assert out["recovery"]["events"][0]["reason"] == "short_after"

    def test_a_zero_bar_is_not_something_to_recover_to(self) -> None:
        # Read on dev: "refund-policy changed · back to 0% in 3 sessions". The
        # agent failed every session before the shift, so any rolling rate
        # clears the bar at once. That is improvement on the curve, not
        # recovery from the change; the shift is reported, not measured.
        rows = _rows("ffffff" + "fffsss")
        out = learning.learning_curve(
            rows, shifts=[{"key": "rule", "at": T0 + timedelta(hours=6)}], window=10
        )
        rec = out["recovery"]
        assert rec["shifts"] == 1 and rec["measured"] == 0 and rec["recovered"] == 0
        assert rec["median_sessions_to_recover"] is None and rec["open"] == []
        event = rec["events"][0]
        assert event["bar"] == 0.0 and event["measured"] is False and event["reason"] == "no_bar"
        # One success before the shift is a bar, and the measurement runs.
        out = learning.learning_curve(
            _rows("fffffs" + "fffsss"), shifts=[{"key": "rule", "at": T0 + timedelta(hours=6)}], window=10
        )
        assert out["recovery"]["measured"] == 1 and out["recovery"]["events"][0]["reason"] is None

    def test_the_median_is_over_recovered_shifts_only(self) -> None:
        before = "ssssssssss"
        rows = _rows(before + "sss" + "fs" + "ssss")
        s1 = T0 + timedelta(hours=10)
        s2 = T0 + timedelta(hours=13)
        out = learning.learning_curve(
            rows,
            shifts=[
                {"key": "r1", "at": s1},
                {"key": "r2", "at": s2},
                {"key": "r3", "discredited_at": s2},
            ],
            window=10,
        )
        rec = out["recovery"]
        assert rec["shifts"] == 3
        costs = [e["sessions_to_recover"] for e in rec["events"]]
        assert costs[0] == 3
        assert rec["median_sessions_to_recover"] is not None

    def test_recovery_after_is_the_building_block(self) -> None:
        flags = [True] * 5 + [False, True, True, True]
        sessions = [(T0 + timedelta(hours=i), ok, None) for i, ok in enumerate(flags)]
        out = learning.recovery_after(sessions, T0 + timedelta(hours=5), window=5)
        assert out["bar"] == 1.0
        assert out["recovered"] is False  # 3/4 then; the failure stays in the window
        out = learning.recovery_after(sessions, T0 + timedelta(hours=5), window=3)
        assert out["recovered"] is True and out["sessions_to_recover"] == 4


# ── The endpoint ────────────────────────────────────────────────────────

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


class _Adapter:
    def __init__(self, sessions, shifts=()) -> None:
        self.sessions = sessions
        self.shifts = list(shifts)
        self.calls: list[dict] = []

    def learning_series(
        self, *, since, agent=None, entity_path=None, visible_agents=None, limit=5000
    ):
        self.calls.append(
            {
                "since": since,
                "agent": agent,
                "entity_path": entity_path,
                "visible_agents": visible_agents,
            }
        )
        rows = self.sessions
        if visible_agents is not None:
            rows = [r for r in rows if r["agent_id"] in visible_agents]
        if agent:
            rows = [r for r in rows if r["agent_id"] == agent]
        return {"sessions": rows, "shifts": self.shifts}


class _Vis:
    def __init__(self, agents=None) -> None:
        self._agents = agents

    def should_filter(self):
        return self._agents is not None

    def get_visible_agent_ids(self):
        return set(self._agents or ())


def _client(monkeypatch, adapter, vis=None):
    mem = types.SimpleNamespace(namespace="test-ns", _adapter=adapter)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
    monkeypatch.setattr(server, "verify_api_key", lambda: None)
    server.app.dependency_overrides[server.verify_api_key] = lambda: None
    return TestClient(server.app)


def test_the_endpoint_returns_the_curve(monkeypatch) -> None:
    adapter = _Adapter(_rows("ssffffsfff" + "ssssssssfs"))
    res = _client(monkeypatch, adapter).get("/api/v1/learning?days=30&window=10&episodes=2")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is True
    assert body["days"] == 30
    assert body["sessions"] == 20
    assert body["first"]["success_rate"] == 0.3 and body["latest"]["success_rate"] == 0.9
    assert len(body["curve"]) == 2
    assert adapter.calls[0]["visible_agents"] is None


def test_a_backend_without_outcomes_says_so(monkeypatch) -> None:
    res = _client(monkeypatch, types.SimpleNamespace()).get("/api/v1/learning")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert "Postgres" in body["reason"]


def test_the_visibility_scope_reaches_the_adapter_and_agent_narrows_on_top(monkeypatch) -> None:
    rows = _rows("ssss") + [dict(r, agent_id="b") for r in _rows("ffff")]
    adapter = _Adapter(rows)
    client = _client(monkeypatch, adapter, _Vis(agents=["a"]))
    body = client.get("/api/v1/learning").json()
    assert body["sessions"] == 4 and body["success_rate"] == 1.0
    assert adapter.calls[-1]["visible_agents"] == {"a"}
    body = client.get("/api/v1/learning?agent=b").json()
    # The adapter is handed both; here the fake applies them the way the real one does.
    assert adapter.calls[-1] == {
        "since": adapter.calls[-1]["since"],
        "agent": "b",
        "entity_path": None,
        "visible_agents": {"a"},
    }
    assert body["sessions"] == 0


def test_parameters_are_clamped(monkeypatch) -> None:
    adapter = _Adapter(_rows("sf"))
    client = _client(monkeypatch, adapter)
    body = client.get("/api/v1/learning?days=9999&window=1&episodes=999").json()
    assert body["days"] == 365
    assert body["window"] == 3
    assert len(body["curve"]) <= learning.MAX_EPISODES
