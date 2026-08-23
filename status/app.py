"""SenseLab status page — FastAPI backend.

Serves an interactive status page (static/) and a JSON API that reports the live
health of SenseLab production services. Health probes run server-side (avoiding
browser CORS issues) on a background loop; results are folded into a lightweight
90-day history stored on disk.

Run locally:
    pip install -r requirements.txt
    python app.py            # http://localhost:8080
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import store
from checker import probe_all
from services import (
    STATUS_LABELS,
    load_services,
    worst_status,
)

POLL_INTERVAL = int(os.environ.get("STATUS_POLL_INTERVAL", "60"))  # seconds
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

SERVICES = load_services()

# In-memory snapshot of the most recent probe of each service, keyed by id.
_latest: dict[str, dict[str, Any]] = {}
_last_checked_at: float | None = None


async def _run_probes() -> None:
    global _last_checked_at
    results = await probe_all(SERVICES)
    store.record(results)
    for r in results:
        _latest[r.service_id] = {
            "status": r.status,
            "latency_ms": round(r.latency_ms, 1) if r.latency_ms is not None else None,
            "http_code": r.http_code,
            "error": r.error,
        }
    _last_checked_at = max((r.checked_at for r in results), default=None)


async def _poller() -> None:
    while True:
        with contextlib.suppress(Exception):
            await _run_probes()
        await asyncio.sleep(POLL_INTERVAL)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Prime once so the first request has data, then poll in the background.
    with contextlib.suppress(Exception):
        await _run_probes()
    task = asyncio.create_task(_poller())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="SenseLab Status", lifespan=lifespan)


def _build_status_payload() -> dict[str, Any]:
    components = []
    for svc in SERVICES:
        latest = _latest.get(svc.id, {})
        status = latest.get("status", "no_data")
        components.append(
            {
                "id": svc.id,
                "name": svc.name,
                "description": svc.description,
                "group": svc.group,
                "internal": svc.internal,
                "tags": list(svc.tags),
                "status": status,
                "status_label": STATUS_LABELS.get(status, status),
                "latency_ms": latest.get("latency_ms"),
                "http_code": latest.get("http_code"),
                "error": latest.get("error"),
                "uptime_90d": store.uptime_percent(svc.id),
                "history": store.uptime_series(svc.id),
                "recent": store.recent_samples(svc.id),
            }
        )

    overall = worst_status([c["status"] for c in components])
    incidents = store.load_incidents()
    active = [i for i in incidents if i.get("status") != "resolved"]

    return {
        "overall": {
            "status": overall,
            "label": STATUS_LABELS.get(overall, overall),
            "all_operational": overall == "operational",
        },
        "components": components,
        "active_incidents": active,
        "incidents": incidents,
        "last_checked_at": (
            dt.datetime.fromtimestamp(_last_checked_at, dt.timezone.utc).isoformat()
            if _last_checked_at
            else None
        ),
        "poll_interval": POLL_INTERVAL,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(
        _build_status_payload(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        reload=bool(os.environ.get("STATUS_RELOAD")),
    )
