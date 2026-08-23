"""Persistence for probe history and incidents.

History is stored as a JSON file: for every service we keep per-day aggregates
covering the last 90 days, plus a short ring buffer of the most recent raw
probes (for the live latency sparkline). This is intentionally lightweight — no
database — so the status page itself has zero external dependencies and can never
be the reason it reports an outage.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path
from typing import Any

from checker import ProbeResult
from services import STATUS_ORDER

HISTORY_DAYS = 90
RECENT_SAMPLES = 60  # ~ last 60 probes for the live sparkline

DATA_DIR = Path(os.environ.get("STATUS_DATA_DIR", Path(__file__).parent / "data"))
HISTORY_PATH = DATA_DIR / "history.json"
INCIDENTS_PATH = Path(
    os.environ.get("STATUS_INCIDENTS_PATH", Path(__file__).parent / "incidents.json")
)

_lock = threading.Lock()


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _rank(status: str) -> int:
    try:
        return STATUS_ORDER.index(status)
    except ValueError:
        return 0


def _load() -> dict[str, Any]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"services": {}}


def _save(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(HISTORY_PATH)


def record(results: list[ProbeResult]) -> None:
    """Fold a batch of probe results into the persisted history."""
    day = _today()
    with _lock:
        data = _load()
        services = data.setdefault("services", {})
        for r in results:
            svc = services.setdefault(r.service_id, {"days": {}, "recent": []})

            # Per-day aggregate: count checks, count "up", track worst status.
            days = svc["days"]
            entry = days.get(day) or {"total": 0, "up": 0, "worst": "operational"}
            if r.status != "no_data":
                entry["total"] += 1
                if r.status in ("operational", "maintenance"):
                    entry["up"] += 1
                if _rank(r.status) > _rank(entry["worst"]):
                    entry["worst"] = r.status
                days[day] = entry

            # Prune to HISTORY_DAYS most recent days.
            cutoff = (
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=HISTORY_DAYS)
            ).strftime("%Y-%m-%d")
            for d in list(days.keys()):
                if d < cutoff:
                    del days[d]

            # Recent ring buffer for live latency.
            recent = svc["recent"]
            recent.append(
                {
                    "t": round(r.checked_at),
                    "status": r.status,
                    "latency_ms": round(r.latency_ms, 1) if r.latency_ms is not None else None,
                    "code": r.http_code,
                }
            )
            svc["recent"] = recent[-RECENT_SAMPLES:]

        _save(data)


def uptime_series(service_id: str) -> list[dict[str, Any]]:
    """Return a 90-element list (oldest -> newest) of daily uptime for a service."""
    data = _load()
    svc = data.get("services", {}).get(service_id, {})
    days = svc.get("days", {})

    today = dt.datetime.now(dt.timezone.utc).date()
    out: list[dict[str, Any]] = []
    for i in range(HISTORY_DAYS - 1, -1, -1):
        d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        entry = days.get(d)
        if not entry or entry["total"] == 0:
            out.append({"date": d, "uptime": None, "status": "no_data"})
        else:
            uptime = entry["up"] / entry["total"]
            out.append({"date": d, "uptime": round(uptime, 4), "status": entry["worst"]})
    return out


def uptime_percent(service_id: str) -> float | None:
    series = uptime_series(service_id)
    vals = [d["uptime"] for d in series if d["uptime"] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100, 2)


def recent_samples(service_id: str) -> list[dict[str, Any]]:
    data = _load()
    return data.get("services", {}).get(service_id, {}).get("recent", [])


def load_incidents() -> list[dict[str, Any]]:
    if INCIDENTS_PATH.exists():
        try:
            payload = json.loads(INCIDENTS_PATH.read_text())
            return payload.get("incidents", []) if isinstance(payload, dict) else payload
        except (json.JSONDecodeError, OSError):
            return []
    return []
