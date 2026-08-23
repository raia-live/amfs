"""Production service definitions for the SenseLab status page.

Each service is probed over HTTP. Public, internet-facing services are probed
directly at their real production URLs. Internal services (retrieval intelligence,
async workers, database) are only reachable from inside the GCP project/VPC, so
their health URLs are configurable via environment variables — when the status
page runs on Cloud Run in the same project it gets real status; otherwise those
components report "no data" rather than a fake green.

Topology reference (amfs-492505, us-central1, all Cloud Run):
  - dashboard        -> amfs.sense-lab.ai, hub.sense-lab.ai
  - amfs-api         -> amfs-login.sense-lab.ai, mcp.sense-lab.ai
  - pro-api          -> internal svc-to-svc (retrieval / intelligence)
  - amfs-doc-worker  -> async (room document extraction)
  - amfs-import-worker -> async (Mem0 / Zep import)
  - Cloud SQL amfs-postgres (POSTGRES_16)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Service:
    id: str
    name: str
    description: str
    group: str
    # URL to probe. Empty string => component is monitored internally and has no
    # public probe (renders as "no data" until history is populated).
    url: str = ""
    method: str = "GET"
    # HTTP status codes that count as healthy. Redirects (3xx) are healthy for
    # front doors that bounce to a canonical host / login.
    healthy_codes: tuple[int, ...] = (200, 201, 204, 301, 302, 307, 308)
    # Optional latency budget (ms). Above this the service is "degraded".
    degraded_above_ms: int = 2500
    internal: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    # When set, this component's status is sourced from the API's aggregate
    # deep-health endpoint (/api/v1/health/components) under this key, instead
    # of being probed directly. Used for internal services the status page
    # cannot reach itself.
    component: str = ""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# The API's deep-health aggregator. amfs-api runs inside the project/VPC and is
# the authoritative source for internal component health (DB, pro-api, Cortex,
# doc worker). The status page reads it and maps each key onto a component.
AGGREGATE_URL = _env(
    "STATUS_AGGREGATE_URL",
    "https://amfs-login.sense-lab.ai/api/v1/health/components",
)


def load_services() -> list[Service]:
    """Build the service list, honoring env overrides for internal endpoints."""

    return [
        Service(
            id="dashboard",
            name="Dashboard",
            description="Web app for reading memory, agents, rooms and traces.",
            group="Web",
            url=_env("STATUS_DASHBOARD_URL", "https://amfs.sense-lab.ai/"),
            tags=("amfs.sense-lab.ai", "hub.sense-lab.ai"),
        ),
        Service(
            id="memory-api",
            name="Memory API",
            description="REST + control-plane gateway. Reads, writes, auth, billing.",
            group="API",
            url=_env("STATUS_API_URL", "https://amfs-login.sense-lab.ai/health"),
            tags=("amfs-login.sense-lab.ai",),
        ),
        Service(
            id="mcp",
            name="Hosted MCP",
            description="Streamable-HTTP MCP endpoint for Cursor, Claude, ChatGPT & more.",
            group="API",
            url=_env("STATUS_MCP_URL", "https://mcp.sense-lab.ai/health"),
            tags=("mcp.sense-lab.ai",),
        ),
        Service(
            id="retrieval",
            name="Retrieval Intelligence",
            description="Multi-strategy retrieval, reranking and briefing compilation.",
            group="API",
            # Internal svc-to-svc — status comes from the API's deep-health
            # aggregator, which can reach pro-api from inside the project.
            url=_env("STATUS_PRO_API_URL", ""),
            internal=True,
            component="retrieval",
        ),
        Service(
            id="database",
            name="Memory Engine",
            description="Durable memory store.",
            group="Data",
            # DB connectivity is proven server-side by the API's deep-health
            # aggregator (a SELECT 1), reported under the "database" key.
            url=_env("STATUS_DB_HEALTH_URL", ""),
            internal=True,
            component="database",
        ),
        Service(
            id="cortex",
            name="ML Engine Pipelines",
            description="Memory Cortex — consolidation, briefing compilation and knowledge digests.",
            group="Workers",
            url=_env("STATUS_CORTEX_URL", ""),
            internal=True,
            component="cortex",
        ),
        Service(
            id="doc-worker",
            name="Document Ingestion",
            description="Async extraction of room documents (PDF, DOCX, Markdown).",
            group="Workers",
            url=_env("STATUS_DOC_WORKER_URL", ""),
            internal=True,
            component="doc_worker",
        ),
        Service(
            id="managed-models",
            name="Managed Models",
            description="OpenAI/Anthropic-compatible serving of account-tuned models.",
            group="Models",
            url=_env("STATUS_MANAGED_MODELS_URL", ""),
            internal=True,
            component="managed_models",
        ),
        Service(
            id="website",
            name="Website",
            description="Marketing site at sense-lab.ai.",
            group="Web",
            url=_env("STATUS_WEBSITE_URL", "https://sense-lab.ai/"),
            tags=("sense-lab.ai",),
        ),
    ]


# Status levels, ordered from best to worst. The overall page status is the
# worst level across all components (matching Statuspage semantics).
STATUS_ORDER = [
    "operational",
    "maintenance",
    "degraded",
    "partial_outage",
    "major_outage",
    "no_data",
]

STATUS_LABELS = {
    "operational": "Operational",
    "maintenance": "Under Maintenance",
    "degraded": "Degraded Performance",
    "partial_outage": "Partial Outage",
    "major_outage": "Major Outage",
    "no_data": "No Data",
}


def worst_status(levels: list[str]) -> str:
    """Return the worst (highest-severity) status, ignoring no_data unless all are."""
    real = [lvl for lvl in levels if lvl != "no_data"]
    if not real:
        return "no_data"
    ranked = {name: i for i, name in enumerate(STATUS_ORDER)}
    return max(real, key=lambda lvl: ranked.get(lvl, 0))
