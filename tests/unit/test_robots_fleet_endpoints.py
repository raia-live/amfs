"""The robot fleet endpoints read the ``fleet/robots`` registry and count learning per day.

A robot runtime writes one ``kind: robot`` row under ``fleet/robots/<id>``; the endpoints
list those rows (with an ``online`` flag from ``last_seen``), and turn the account's
outcomes and entries into per-day counts: episodes per loop with their success rate,
rules per loop (entries under ``<prefix>/<loop>``), facts per topic, corrections. Entries
outside a robot's prefix never count for it, and ``user/*`` counts only when the robot's
own owner agent wrote it — a fleet agent is shared across robots.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import amfs_http.server as server
import pytest
from amfs import AgentMemory
from amfs_core.models import OutcomeRecord, OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_http.robots import _learning_series, _topic_of
from fastapi.testclient import TestClient

NOW = datetime.now(timezone.utc)


class _OutcomeKeepingAdapter(FilesystemAdapter):
    """The filesystem adapter does not list outcomes; this one remembers what it committed."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.kept: list[OutcomeRecord] = []

    def commit_outcome(self, record: OutcomeRecord):  # type: ignore[override]
        self.kept.append(record)
        return super().commit_outcome(record)

    def list_outcomes(self, *, entity_path=None, since=None, limit=1000, outcome_ref=None):  # type: ignore[override]
        rows = [r for r in self.kept if since is None or r.committed_at >= since]
        return rows[:limit]


def _robot_row(rid: str, prefix: str, owner: str, *, last_seen: datetime) -> dict:
    return {
        "kind": "robot", "id": rid, "name": rid.title(), "platform": "reachy-mini-wireless",
        "entity_prefix": prefix, "agents": {"fleet": "reachy-fleet", "owner": owner},
        "loops": {"greet": {}, "idle": {}, "observe": {}}, "last_seen": last_seen.isoformat(),
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, _OutcomeKeepingAdapter]:
    adapter = _OutcomeKeepingAdapter(tmp_path)
    mem = AgentMemory(agent_id="amfs-server", adapter=adapter)
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_visible_agent_ids", lambda request: None)
    return TestClient(server.app), adapter


def _seed(adapter: _OutcomeKeepingAdapter) -> None:
    fleet = AgentMemory("reachy-fleet", adapter=adapter)
    chucky = AgentMemory("reachy-owner-chucky", adapter=adapter)
    fleet.write("fleet/robots", "chucky", _robot_row("chucky", "reachy-pilot", "reachy-owner-chucky", last_seen=NOW))
    fleet.write("fleet/robots", "nemo", _robot_row("nemo", "nemo-pilot", "reachy-owner-nemo",
                                                   last_seen=NOW - timedelta(days=3)))
    fleet.write("fleet/robots", "gone", {"kind": "retracted"})           # not a robot row
    fleet.write("reachy-pilot/greet", "learned-owner-first-abc123", {"kind": "rule"})
    chucky.write("reachy-pilot/environment/homedesk", "scene-left", {"objects": ["laptop"]})
    chucky.write("reachy-pilot/owner-model", "activity-afternoon", {"counts": {"talking": 1}})
    chucky.write("user/people", "fact-a", {"text": "x"})
    chucky.write("user/corrections", "fact-b", {"text": "y"})
    for i, outcome in enumerate([OutcomeType.SUCCESS, OutcomeType.SUCCESS, OutcomeType.FAILURE]):
        fleet.read("reachy-pilot/greet", "learned-owner-first-abc123")
        fleet.commit_outcome(f"ep-{i}", outcome, entity_path="reachy-pilot/greet")


class TestRegistry:
    def test_lists_robot_rows_only_with_an_online_flag(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        body = c.get("/api/v1/robots").json()
        assert [r["id"] for r in body["robots"]] == ["chucky", "nemo"]
        assert body["count"] == 2 and body["online"] == 1
        by_id = {r["id"]: r for r in body["robots"]}
        assert by_id["chucky"]["online"] is True and by_id["nemo"]["online"] is False

    def test_unknown_robot_is_404(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        assert c.get("/api/v1/robots/ghost").status_code == 404
        assert c.get("/api/v1/robots/gone").status_code == 404


class TestLearning:
    def test_counts_episodes_rules_facts_and_corrections_per_day(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        body = c.get("/api/v1/robots/chucky/learning?days=7").json()
        assert body["robot"] == "chucky" and body["bucket"] == "day"
        assert len(body["series"]) == 1
        day = body["series"][0]
        assert day["episodes"] == {"greet": {"n": 3, "success": 2}}
        assert day["success_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert day["rules"] == {"greet": 1}
        assert day["facts"] == {"environment": 1, "owner-model": 1, "people": 1}
        assert day["corrections"] == 1
        assert body["totals"] == {"episodes": 3, "successes": 2, "rules": 1, "facts": 3, "corrections": 1}

    def test_another_robot_does_not_inherit_the_fleet_agents_writes(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        body = c.get("/api/v1/robots/nemo/learning?days=7").json()
        assert body["totals"] == {"episodes": 0, "successes": 0, "rules": 0, "facts": 0, "corrections": 0}

    def test_fleet_view_sums_robots_and_names_contributors(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        body = c.get("/api/v1/robots/learning?days=7").json()
        assert body["robots"] == ["chucky", "nemo"]
        assert body["per_robot"]["chucky"]["episodes"] == 3
        assert body["series"][0]["robots"] == ["chucky"]
        assert body["series"][0]["episodes_total"] == 3

    def test_bad_timezone_is_400_and_a_real_one_shifts_the_day(self, client) -> None:
        c, adapter = client
        _seed(adapter)
        assert c.get("/api/v1/robots/chucky/learning?tz=Mars/Olympus").status_code == 400
        assert c.get("/api/v1/robots/chucky/learning?tz=Asia/Tokyo").status_code == 200


class TestSeriesBuilder:
    def test_topic_classification(self) -> None:
        loops = {"greet", "observe"}
        assert _topic_of("reachy-pilot/greet", "reachy-pilot", loops) == ("rules", "greet")
        assert _topic_of("reachy-pilot/environment/homedesk", "reachy-pilot", loops) == ("facts", "environment")
        assert _topic_of("reachy-pilot", "reachy-pilot", loops) == ("facts", "overview")
        assert _topic_of("user/people", "reachy-pilot", loops) == ("facts", "people")
        assert _topic_of("user/corrections", "reachy-pilot", loops) == ("corrections", "corrections")
        assert _topic_of("other-robot/greet", "reachy-pilot", loops) is None

    def test_episode_loop_comes_from_entity_paths_then_causal_keys(self) -> None:
        row = _robot_row("chucky", "reachy-pilot", "reachy-owner-chucky", last_seen=NOW)
        outcomes = [
            OutcomeRecord(outcome_ref="ep-1", outcome_type=OutcomeType.SUCCESS, committed_at=NOW,
                          agent_id="reachy-fleet", entity_paths=["reachy-pilot/observe"]),
            OutcomeRecord(outcome_ref="ep-2", outcome_type=OutcomeType.FAILURE, committed_at=NOW,
                          agent_id="reachy-fleet", causal_entry_keys=["reachy-pilot/greet/learned-x"]),
            OutcomeRecord(outcome_ref="ep-3", outcome_type=OutcomeType.SUCCESS, committed_at=NOW,
                          agent_id="reachy-fleet"),                      # no path under the prefix: not attributable
            OutcomeRecord(outcome_ref="ep-4", outcome_type=OutcomeType.SUCCESS, committed_at=NOW,
                          agent_id="somebody-else", entity_paths=["reachy-pilot/greet"]),   # not this robot
        ]
        data = _learning_series(row=row, outcomes=outcomes, entries=[], tz=timezone.utc)
        assert data["series"][0]["episodes"] == {"greet": {"n": 1, "success": 0}, "observe": {"n": 1, "success": 1}}
        assert data["totals"]["episodes"] == 2
