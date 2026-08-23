"""Async health probing for SenseLab production services."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from services import AGGREGATE_URL, Service

VALID_STATUSES = {
    "operational",
    "maintenance",
    "degraded",
    "partial_outage",
    "major_outage",
    "no_data",
}


@dataclass
class ProbeResult:
    service_id: str
    status: str  # operational | degraded | major_outage | no_data
    latency_ms: float | None
    http_code: int | None
    error: str | None
    checked_at: float  # unix seconds


async def fetch_aggregate(client: httpx.AsyncClient) -> dict:
    """Fetch the API's deep-health aggregator (/api/v1/health/components).

    Returns a {component_key: {status, latency_ms, ...}} map, or {} on failure
    (in which case aggregate-sourced components fall back to no_data).
    """
    if not AGGREGATE_URL:
        return {}
    try:
        resp = await client.get(
            AGGREGATE_URL,
            timeout=10.0,
            headers={"User-Agent": "SenseLab-StatusPage/1.0"},
        )
        if resp.status_code >= 400:
            return {}
        data = resp.json()
        return data.get("components", {}) if isinstance(data, dict) else {}
    except (httpx.HTTPError, ValueError):
        return {}


def result_from_aggregate(svc: Service, components: dict, now: float) -> ProbeResult:
    comp = components.get(svc.component) or {}
    status = comp.get("status", "no_data")
    if status not in VALID_STATUSES:
        status = "no_data"
    return ProbeResult(
        svc.id,
        status,
        comp.get("latency_ms"),
        comp.get("http_code"),
        comp.get("error"),
        now,
    )


async def probe(client: httpx.AsyncClient, svc: Service) -> ProbeResult:
    now = time.time()

    if not svc.url:
        # Internal component with no configured probe endpoint.
        return ProbeResult(svc.id, "no_data", None, None, None, now)

    start = time.perf_counter()
    try:
        resp = await client.request(
            svc.method,
            svc.url,
            follow_redirects=False,
            timeout=10.0,
            headers={"User-Agent": "SenseLab-StatusPage/1.0"},
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        code = resp.status_code

        if code in svc.healthy_codes:
            level = "degraded" if latency_ms > svc.degraded_above_ms else "operational"
            return ProbeResult(svc.id, level, latency_ms, code, None, now)

        # 5xx => outage, other unexpected codes => partial/degraded.
        if code >= 500:
            return ProbeResult(svc.id, "major_outage", latency_ms, code, f"HTTP {code}", now)
        return ProbeResult(svc.id, "degraded", latency_ms, code, f"HTTP {code}", now)

    except (httpx.TimeoutException, asyncio.TimeoutError):
        return ProbeResult(svc.id, "major_outage", None, None, "timeout", now)
    except httpx.HTTPError as exc:
        return ProbeResult(svc.id, "major_outage", None, None, type(exc).__name__, now)


async def probe_all(services: list[Service]) -> list[ProbeResult]:
    async with httpx.AsyncClient() as client:
        # Services with an explicit probe URL (or no aggregate mapping) are
        # probed directly. Aggregate-sourced components are resolved from a
        # single deep-health call to the API.
        needs_aggregate = any(svc.component and not svc.url for svc in services)
        aggregate = await fetch_aggregate(client) if needs_aggregate else {}

        now = time.time()
        tasks: list = []
        for svc in services:
            if svc.component and not svc.url:
                tasks.append(_wrap(result_from_aggregate(svc, aggregate, now)))
            else:
                tasks.append(probe(client, svc))
        return await asyncio.gather(*tasks)


async def _wrap(result: ProbeResult) -> ProbeResult:
    return result
