# SenseLab Status Page

An interactive status page for SenseLab production services, modeled on
Statuspage / githubstatus.com. Intended to be served at **status.sense-lab.ai**.

- **Live health** — server-side health probes of every production service
  (no browser CORS issues), refreshed on a background loop.
- **90-day uptime bars** — per-day uptime history with hover tooltips, built up
  from the probe loop and stored in a lightweight JSON file (no database).
- **Incidents** — an editable `incidents.json` drives the active-incident banner
  and the "Past Incidents" timeline.
- **Zero heavy deps** — FastAPI + httpx. The status page can never itself be the
  reason it reports an outage.

## Monitored services

Derived from the live GCP topology (project `amfs-492505`, us-central1, Cloud Run):

| Component              | Group   | Probe                                            |
| ---------------------- | ------- | ------------------------------------------------ |
| Dashboard              | Web     | `https://amfs.sense-lab.ai/`                     |
| Website                | Web     | `https://sense-lab.ai/`                          |
| Memory API             | API     | `https://amfs-login.sense-lab.ai/health`         |
| Hosted MCP             | API     | `https://mcp.sense-lab.ai/health`                |
| Retrieval Intelligence | API     | internal (aggregator `retrieval`)                |
| Memory Engine          | Data    | internal (aggregator `database`)                 |
| ML Engine Pipelines    | Workers | internal (aggregator `cortex`)                   |
| Document Ingestion     | Workers | internal (aggregator `doc_worker`)               |
| Managed Models         | Models  | internal (aggregator `managed_models`)           |

Public services are probed at their real production URLs. Internal services
(Retrieval Intelligence, Memory Engine, ML Engine Pipelines, Document Ingestion)
are only reachable from inside the project/VPC, so their status is sourced from
the **API deep-health aggregator** — see below. Without a reachable aggregator
they render as **No Data** (honest) rather than a fake green.

## How internal services report real status

The status page cannot reach `pro-api`, Cloud SQL, the Cortex pipeline, or the
doc worker directly. `amfs-api` can — it sits inside the project/VPC. So it
exposes an authoritative, unauthenticated aggregator:

```
GET https://amfs-login.sense-lab.ai/api/v1/health/components
->
{
  "status": "ok",
  "components": {
    "database":       { "status": "operational", "latency_ms": 4.2 },   // SELECT 1
    "retrieval":      { "status": "operational", "latency_ms": 38.1 },  // GET {AMFS_PRO_URL}/health
    "cortex":         { "status": "operational", "digest_count": 1284 },// Cortex digest table
    "doc_worker":     { "status": "operational", "latency_ms": 21.0 },  // GET {AMFS_DOC_WORKER_URL}/health
    "managed_models": { "status": "operational" }                       // /api/v1/models serving router mounted
  }
}
```

The status page fetches this once per poll and maps each key onto the matching
component (`STATUS_AGGREGATE_URL`, default the URL above). Every check in the
aggregator is fail-open: a failure reports `major_outage`/`no_data` for that one
component and never breaks the endpoint.

For this to return real (not `no_data`) values, the `amfs-api` service must have
`AMFS_PRO_URL` set (already used by the Pro proxy) and, to include the doc
worker, `AMFS_DOC_WORKER_URL` pointing at the worker's health URL. The endpoint
lives in `packages/http-server/src/amfs_http/server.py` and ships with the
normal `amfs-api` deploy.

## Run locally

```bash
cd status
pip install -r requirements.txt
python app.py
# open http://localhost:8080
```

The first probe runs on startup, then every `STATUS_POLL_INTERVAL` seconds
(default 60). Uptime history accumulates over time in `data/history.json`.

## Configuration (env vars)

| Variable                  | Default                                    | Purpose                                  |
| ------------------------- | ------------------------------------------ | ---------------------------------------- |
| `PORT`                    | `8080`                                     | HTTP port                                |
| `STATUS_POLL_INTERVAL`    | `60`                                       | Seconds between health probes            |
| `STATUS_DATA_DIR`         | `./data`                                   | Where `history.json` is written          |
| `STATUS_INCIDENTS_PATH`   | `./incidents.json`                         | Incident feed                            |
| `STATUS_AGGREGATE_URL`    | `…/api/v1/health/components`               | API deep-health aggregator (internal svc status) |
| `STATUS_DASHBOARD_URL`    | `https://amfs.sense-lab.ai/`               | Override dashboard probe                 |
| `STATUS_API_URL`          | `https://amfs-login.sense-lab.ai/health`   | Override Memory API probe                |
| `STATUS_MCP_URL`          | `https://mcp.sense-lab.ai/health`          | Override Hosted MCP probe                |
| `STATUS_WEBSITE_URL`      | `https://sense-lab.ai/`                    | Override website probe                   |
| `STATUS_PRO_API_URL`      | _(unset)_                                  | Internal pro-api health URL              |
| `STATUS_DB_HEALTH_URL`    | _(unset)_                                  | Memory Engine deep health URL            |
| `STATUS_CORTEX_URL`       | _(unset)_                                  | Memory Cortex / ML pipelines health URL  |
| `STATUS_DOC_WORKER_URL`   | _(unset)_                                  | Doc worker health URL                    |
| `STATUS_MANAGED_MODELS_URL`| _(unset)_                                 | Managed Models serving health URL        |

## Publishing an incident

Edit `incidents.json`. Non-`resolved` incidents show in the top banner; all
incidents show in the "Past Incidents" timeline grouped by day.

```json
{
  "id": "2026-08-24-mcp",
  "title": "Elevated errors on Hosted MCP",
  "impact": "partial_outage",        // degraded | partial_outage | major_outage | maintenance
  "status": "investigating",          // investigating | identified | monitoring | resolved
  "date": "2026-08-24",
  "summary": "Short one-line summary shown in the banner.",
  "updates": [
    { "label": "Investigating", "at": "2026-08-24T10:00:00Z", "body": "We are looking into it." }
  ]
}
```

Redeploy (or mount `incidents.json` from a volume / GCS-synced file) to publish.

## Deploy to Cloud Run at status.sense-lab.ai

Deploys are centralized: pushing this repo's `main`/`dev` fires a
`repository-dispatch` to the private `raia-live/amfs-internal` pipeline
(see `.github/workflows/trigger-deploy.yml`), which runs the deploy. The
canonical, runnable recipe lives in [`cloudbuild.yaml`](./cloudbuild.yaml) — it
builds the image, deploys the `status-page` Cloud Run service, and creates the
`status.sense-lab.ai` domain mapping (idempotent).

Run it manually:

```bash
gcloud builds submit --config status/cloudbuild.yaml status/
```

Key deploy choices (baked into `cloudbuild.yaml`):

- `--min-instances=1` keeps one instance warm so the background probe loop runs
  continuously and uptime history stays current.
- `--allow-unauthenticated` — it's a public status page.
- `STATUS_AGGREGATE_URL` defaults to the prod amfs-api deep-health aggregator.

The domain-mapping step prints a one-time DNS record to add at the registrar.
For durable uptime history across revisions, mount a volume at `/data` (e.g. a
GCS bucket via Cloud Run volume mounts) or point `STATUS_DATA_DIR` at
persistent storage.

> Note: this app lives in the public OSS repo. No internal hostnames are
> committed — internal service health is sourced at runtime from the amfs-api
> aggregator, and any per-service URL overrides are supplied as deploy-time env
> vars in the private pipeline.
