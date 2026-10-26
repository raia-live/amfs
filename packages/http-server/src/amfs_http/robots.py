"""The robot fleet: ``/api/v1/robots`` for the fleet dashboard.

A robot runtime registers itself by writing one row under ``fleet/robots/<robot-id>``
(name, platform, host, capabilities, the agents it writes as, the entity prefix its
episodes and facts live under, loops, last self-check, last seen). These endpoints turn
those rows and the account's outcomes and entries into what a fleet dashboard shows:

- ``GET  /api/v1/robots``                       every robot, with an ``online`` flag
- ``GET  /api/v1/robots/learning``              fleet-wide learning per day and topic
- ``GET  /api/v1/robots/{robot_id}``            one robot's row plus its agents' summary
- ``GET  /api/v1/robots/{robot_id}/learning``   that robot's learning per day and topic

"Learning" is counted, never interpreted: episodes per loop with their success rate
(outcomes committed by the robot's agents), rules learned per loop (entries under
``<prefix>/<loop>``), facts per topic (entries under the prefix's other children and the
owner agent's ``user/<topic>``), and corrections (``user/corrections``). Buckets are local
days of the server clock in UTC unless ``tz`` names an IANA zone.

Nothing here reads a value: the series carry counts, keys and paths only.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request

from amfs_core.models import SearchQuery

logger = logging.getLogger(__name__)

REGISTRY_PATH = "fleet/robots"
ONLINE_WINDOW = timedelta(minutes=15)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _day(dt: datetime | None, tz: Any) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(tz).date().isoformat()


def _row(entry: Any) -> dict[str, Any] | None:
    value = getattr(entry, "value", None)
    if not isinstance(value, dict) or value.get("kind") != "robot":
        return None
    row = dict(value)
    row.setdefault("id", getattr(entry, "key", None))
    row["registered_by"] = getattr(getattr(entry, "provenance", None), "agent_id", None)
    seen = _parse_dt(row.get("last_seen"))
    row["online"] = bool(seen and datetime.now(timezone.utc) - seen <= ONLINE_WINDOW)
    return row


def _robot_agents(row: dict[str, Any]) -> set[str]:
    agents = row.get("agents") or {}
    if isinstance(agents, dict):
        return {str(a) for a in agents.values() if a}
    if isinstance(agents, list):
        return {str(a) for a in agents if a}
    return set()


def _topic_of(entity_path: str, prefix: str, loops: set[str]) -> tuple[str, str] | None:
    """(kind, topic) for an entry path: rules per loop, facts per topic, corrections.

    ``None`` for a path that is not this robot's: a fleet agent is shared across robots,
    so only entries under the robot's own prefix (any of its agents) or the owner's
    ``user/*`` count — the caller checks the author for the latter.
    """
    if entity_path == "user/corrections":
        return "corrections", "corrections"
    if entity_path.startswith("user/"):
        return "facts", entity_path.split("/", 2)[1]
    if prefix and (entity_path == prefix or entity_path.startswith(prefix + "/")):
        rest = entity_path[len(prefix):].strip("/")
        head = rest.split("/", 1)[0] if rest else "overview"
        if head in loops:
            return "rules", head
        return "facts", head or "overview"
    return None


def _learning_series(
    *,
    row: dict[str, Any],
    outcomes: list[Any],
    entries: list[Any],
    tz: Any,
) -> dict[str, Any]:
    prefix = str(row.get("entity_prefix") or "")
    loops = set((row.get("loops") or {}).keys()) if isinstance(row.get("loops"), dict) else set()
    agents = _robot_agents(row)
    agent_roles = row.get("agents") if isinstance(row.get("agents"), dict) else {}
    owner_agent = str(agent_roles.get("owner") or "") if agent_roles else ""
    days: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "episodes": defaultdict(lambda: {"n": 0, "success": 0}),
        "rules": defaultdict(int),
        "facts": defaultdict(int),
        "corrections": 0,
    })

    for o in outcomes:
        if agents and getattr(o, "agent_id", None) not in agents:
            continue
        day = _day(getattr(o, "committed_at", None), tz)
        if day is None:
            continue
        loop = None
        for p in list(getattr(o, "entity_paths", None) or []) + list(getattr(o, "causal_entry_keys", None) or []):
            p = str(p)
            if prefix and p.startswith(prefix + "/"):
                head = p[len(prefix) + 1:].split("/", 1)[0]
                if head in loops or not loops:
                    loop = head
                    break
        if loop is None:
            # No path under this robot's prefix: not its episode. A fleet agent is shared
            # across robots, so the author alone cannot attribute an outcome.
            continue
        bucket = days[day]["episodes"][loop]
        bucket["n"] += 1
        otype = getattr(o, "outcome_type", None)
        otype = getattr(otype, "value", otype)
        if str(otype) == "success":
            bucket["success"] += 1

    for e in entries:
        prov = getattr(e, "provenance", None)
        if agents and getattr(prov, "agent_id", None) not in agents:
            continue
        day = _day(_parse_dt(getattr(prov, "written_at", None)), tz)
        if day is None:
            continue
        path = str(getattr(e, "entity_path", "") or "")
        if path == REGISTRY_PATH:
            continue
        classified = _topic_of(path, prefix, loops)
        if classified is None:
            continue
        # user/* is the owner's memory: only the owner agent's writes belong to this robot
        if path.startswith("user/") and owner_agent and getattr(prov, "agent_id", None) != owner_agent:
            continue
        kind, topic = classified
        if kind == "corrections":
            days[day]["corrections"] += 1
        elif kind == "rules":
            days[day]["rules"][topic] += 1
        else:
            days[day]["facts"][topic] += 1

    series = []
    totals: dict[str, Any] = {"episodes": 0, "successes": 0, "rules": 0, "facts": 0, "corrections": 0}
    for day in sorted(days):
        d = days[day]
        eps = {k: dict(v) for k, v in sorted(d["episodes"].items())}
        n = sum(v["n"] for v in eps.values())
        s = sum(v["success"] for v in eps.values())
        series.append({
            "day": day,
            "episodes": eps,
            "episodes_total": n,
            "success_rate": round(s / n, 3) if n else None,
            "rules": dict(sorted(d["rules"].items())),
            "facts": dict(sorted(d["facts"].items())),
            "corrections": d["corrections"],
        })
        totals["episodes"] += n
        totals["successes"] += s
        totals["rules"] += sum(d["rules"].values())
        totals["facts"] += sum(d["facts"].values())
        totals["corrections"] += d["corrections"]
    return {"series": series, "totals": totals}


def mount_robots(
    app: FastAPI,
    *,
    get_memory: Callable[[], Any],
    visible_agent_ids: Callable[[Request], set[str] | None],
    verify_api_key: Any,
) -> None:
    from fastapi import Depends

    def _tz(name: str | None) -> Any:
        if not name:
            return timezone.utc
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"unknown tz {name!r}") from None

    def _rows(request: Request) -> list[dict[str, Any]]:
        mem = get_memory()
        allowed = visible_agent_ids(request)
        rows = []
        for entry in mem.list(REGISTRY_PATH):
            row = _row(entry)
            if row is None:
                continue
            if allowed is not None and row.get("registered_by") not in allowed:
                continue
            rows.append(row)
        rows.sort(key=lambda r: str(r.get("name") or r.get("id")))
        return rows

    def _since(since: str | None, days: int) -> datetime:
        if since:
            dt = _parse_dt(since)
            if dt is None:
                raise HTTPException(status_code=400, detail="since must be ISO-8601")
            return dt
        return datetime.now(timezone.utc) - timedelta(days=days)

    def _collect(request: Request, rows: list[dict[str, Any]], since_dt: datetime, limit: int) -> tuple[list, list]:
        mem = get_memory()
        allowed = visible_agent_ids(request)
        outcomes = mem._adapter.list_outcomes(since=since_dt, limit=limit)
        if allowed is not None:
            outcomes = [o for o in outcomes if o.agent_id in allowed]
        entries: list[Any] = []
        seen: set[tuple[str, str, int]] = set()
        paths = {str(r.get("entity_prefix")) for r in rows if r.get("entity_prefix")} | {"user"}
        for path in sorted(paths):
            query = SearchQuery(entity_path=path, include_descendants=True, since=since_dt,
                                sort_by="recency", limit=limit)
            try:
                found = mem._adapter.search(query)
            except Exception:  # noqa: BLE001 - an adapter without include_descendants
                logger.warning("robots: search under %s failed", path, exc_info=True)
                continue
            for e in found:
                key = (str(e.entity_path), str(e.key), int(getattr(e, "version", 0) or 0))
                if key in seen:
                    continue
                seen.add(key)
                if allowed is not None and getattr(e.provenance, "agent_id", None) not in allowed:
                    continue
                entries.append(e)
        return outcomes, entries

    @app.get("/api/v1/robots")
    async def list_robots(request: Request, _auth: str | None = Depends(verify_api_key)) -> dict[str, Any]:
        rows = _rows(request)
        return {"robots": rows, "count": len(rows), "online": sum(1 for r in rows if r["online"])}

    @app.get("/api/v1/robots/learning")
    async def fleet_learning(
        request: Request,
        since: str | None = Query(None),
        days: int = Query(30, ge=1, le=365),
        tz: str | None = Query(None),
        limit: int = Query(5000, ge=1, le=20000),
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        zone = _tz(tz)
        rows = _rows(request)
        since_dt = _since(since, days)
        outcomes, entries = _collect(request, rows, since_dt, limit)
        per_robot = {r["id"]: _learning_series(row=r, outcomes=outcomes, entries=entries, tz=zone) for r in rows}
        # fleet view: the same shape, summed over robots, plus who contributed per day
        merged: dict[str, dict[str, Any]] = {}
        for rid, data in per_robot.items():
            for s in data["series"]:
                m = merged.setdefault(s["day"], {"day": s["day"], "episodes": {}, "rules": {}, "facts": {},
                                                 "corrections": 0, "robots": []})
                for loop, v in s["episodes"].items():
                    cur = m["episodes"].setdefault(loop, {"n": 0, "success": 0})
                    cur["n"] += v["n"]
                    cur["success"] += v["success"]
                for k in ("rules", "facts"):
                    for topic, n in s[k].items():
                        m[k][topic] = m[k].get(topic, 0) + n
                m["corrections"] += s["corrections"]
                m["robots"].append(rid)
        series = []
        for day in sorted(merged):
            m = merged[day]
            n = sum(v["n"] for v in m["episodes"].values())
            sacc = sum(v["success"] for v in m["episodes"].values())
            m["episodes_total"] = n
            m["success_rate"] = round(sacc / n, 3) if n else None
            series.append(m)
        return {"since": since_dt.isoformat(), "bucket": "day", "tz": tz or "UTC",
                "robots": [r["id"] for r in rows], "series": series,
                "per_robot": {rid: d["totals"] for rid, d in per_robot.items()}}

    @app.get("/api/v1/robots/{robot_id}")
    async def get_robot(request: Request, robot_id: str, _auth: str | None = Depends(verify_api_key)) -> dict[str, Any]:
        for row in _rows(request):
            if row["id"] == robot_id:
                return {"robot": row}
        raise HTTPException(status_code=404, detail=f"robot {robot_id!r} not registered")

    @app.get("/api/v1/robots/{robot_id}/learning")
    async def robot_learning(
        request: Request,
        robot_id: str,
        since: str | None = Query(None),
        days: int = Query(30, ge=1, le=365),
        tz: str | None = Query(None),
        limit: int = Query(5000, ge=1, le=20000),
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        zone = _tz(tz)
        rows = [r for r in _rows(request) if r["id"] == robot_id]
        if not rows:
            raise HTTPException(status_code=404, detail=f"robot {robot_id!r} not registered")
        since_dt = _since(since, days)
        outcomes, entries = _collect(request, rows, since_dt, limit)
        data = _learning_series(row=rows[0], outcomes=outcomes, entries=entries, tz=zone)
        return {"robot": robot_id, "since": since_dt.isoformat(), "bucket": "day", "tz": tz or "UTC", **data}


__all__ = ["REGISTRY_PATH", "mount_robots"]
