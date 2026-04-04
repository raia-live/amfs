---
title: Connectors
layout: default
parent: Guides
nav_order: 7
description: "Connect external systems to AMFS — install built-in connectors or build your own."
---

# Connectors
{: .no_toc }

Connectors let AMFS ingest events from external systems — PagerDuty incidents, GitHub PRs, Slack messages, Jira tickets — and turn them into structured memory entries that agents can query and learn from.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What Are Connectors?

A connector is a plugin that bridges an external system and AMFS. It receives raw events (typically via webhooks), transforms them into `write()` or `record_context()` operations, and feeds them into the memory store.

```
External System          AMFS
┌──────────┐      ┌──────────────────────────────────┐
│ PagerDuty│─────▶│  Webhook Endpoint                │
│ GitHub   │─────▶│  /api/v1/webhooks/{connector}    │
│ Slack    │─────▶│                                  │
│ Jira     │─────▶│  ┌────────────┐  ┌────────────┐ │
│ Custom   │─────▶│  │ Connector  │──▶│ AgentMemory│ │
└──────────┘      │  │ Transform  │  │  write()   │ │
                  │  └────────────┘  │  record()  │ │
                  │                  └────────────┘ │
                  └──────────────────────────────────┘
```

Each connector:
- Validates incoming payloads (HMAC signature verification)
- Deduplicates events (idempotency keys)
- Transforms raw JSON into AMFS memory operations
- Writes structured entries with proper entity paths, confidence scores, and memory types
- Persists all data to the shared memory pool (tagged with `agent_id="webhook/{connector_name}"`)

Events that don't match a registered transform are still persisted as raw entries, ensuring no data is lost.

---

## Installing a Connector

Install connectors from PyPI using the `amfs` CLI:

```bash
amfs connector install pagerduty
```

This installs the `amfs-connector-pagerduty` package and registers it with AMFS. You can also install directly via pip:

```bash
pip install amfs-connector-pagerduty
```

List installed connectors:

```bash
amfs connector list
```

Remove a connector:

```bash
amfs connector remove pagerduty
```

---

## Connecting External Systems

Once a connector is installed, external systems send events to AMFS via a webhook URL.

### Webhook URL

Each connector exposes a webhook endpoint:

```
POST /api/v1/webhooks/{name}
```

For example, after installing the PagerDuty connector:

```
POST https://your-amfs-host/api/v1/webhooks/pagerduty
```

### Authentication

Connectors use HMAC signature verification to validate incoming payloads. Configure the shared secret as an environment variable:

```bash
export AMFS_CONNECTOR_PAGERDUTY_SECRET=whsec_your_secret_here
export AMFS_CONNECTOR_GITHUB_SECRET=whsec_your_secret_here
export AMFS_CONNECTOR_SLACK_SECRET=whsec_your_secret_here
export AMFS_CONNECTOR_JIRA_SECRET=whsec_your_secret_here
```

The naming convention is `AMFS_CONNECTOR_{NAME}_SECRET` where `{NAME}` is the uppercased connector name.

### Headers

Connectors expect standard webhook headers from the source system. For example, PagerDuty sends `X-PagerDuty-Signature`, GitHub sends `X-Hub-Signature-256`. Each connector knows how to validate its own system's signature format.

---

## Built-in Connectors

### PagerDuty

Ingests incident lifecycle events: triggered, acknowledged, resolved.

```bash
amfs connector install pagerduty
```

**What it captures:**
- Incident title, severity, and status
- Assigned responders
- Service and escalation policy
- Resolution notes

**Entity path:** `incidents/pagerduty/{service_name}`

**Example entry:**

```python
{
    "incident_id": "P1234ABC",
    "title": "High error rate on checkout-service",
    "severity": "critical",
    "status": "triggered",
    "service": "checkout-service",
    "assignees": ["oncall-team"],
}
```

---

### GitHub

Ingests pull request events, deployment statuses, and issue updates.

```bash
amfs connector install github
```

**What it captures:**
- PR opened/merged/closed events with diff stats
- Deployment success/failure with environment info
- Issue state changes and label updates

**Entity path:** `repos/{owner}/{repo}`

---

### Slack

Ingests channel messages and thread context relevant to incidents or decisions.

```bash
amfs connector install slack
```

**What it captures:**
- Messages from monitored channels
- Thread context around incident discussions
- Bot interaction logs

**Entity path:** `comms/slack/{channel}`

---

### Jira

Ingests issue transitions, sprint events, and release notes.

```bash
amfs connector install jira
```

**What it captures:**
- Issue status transitions (e.g., In Progress → Done)
- Sprint start/complete events
- Release creation and deployment tracking

**Entity path:** `projects/jira/{project_key}`

---

## Direct Event Ingestion

For lightweight integrations that don't need the full connector framework, AMFS exposes a direct event endpoint:

```
POST /api/v1/events
```

Send a JSON body:

```json
{
  "source": "monitoring",
  "entity_path": "myapp/checkout",
  "key": "high-cpu-alert",
  "value": {"host": "prod-3", "cpu_percent": 95},
  "event_type": "alert.triggered"
}
```

Events ingested through this endpoint are stored as shared memory entries with `agent_id="external/{source}"`, making them visible to all agents via `search()` and automatically included in Cortex digest compilation.

This is useful for:
- Simple scripts or cron jobs that push metrics
- CI/CD pipelines reporting deployment status
- Any system where building a full connector is overkill

---

## Building Your Own Connector

You can build a connector for any system that sends webhooks or has an API.

### Step 1: Create the Package

```bash
mkdir amfs-connector-myservice
cd amfs-connector-myservice
```

### Step 2: Write the Connector Manifest

Create a `connector.yaml` at the package root:

```yaml
name: myservice
version: "1.0.0"
description: "Ingest events from MyService into AMFS"
author: "Your Name"
license: "Apache-2.0"

entry_point: amfs_connector_myservice.connector:MyServiceConnector

webhook:
  signature_header: "X-MyService-Signature"
  signature_algorithm: "hmac-sha256"

default_entity_path: "integrations/myservice"

event_types:
  - "event.created"
  - "event.updated"
  - "event.deleted"

env_vars:
  - name: AMFS_CONNECTOR_MYSERVICE_SECRET
    description: "HMAC secret for webhook signature verification"
    required: true
  - name: AMFS_CONNECTOR_MYSERVICE_API_KEY
    description: "API key for pulling historical data"
    required: false
```

### Step 3: Implement the Connector

Subclass `ConnectorABC` and implement the `transform` method:

```python
from amfs_connectors import ConnectorABC, ConnectorEvent, ConnectorResult
from amfs import MemoryType


class MyServiceConnector(ConnectorABC):
    """Transforms MyService webhook events into AMFS memory operations."""

    name = "myservice"

    def transform(self, event: ConnectorEvent) -> list[ConnectorResult]:
        payload = event.payload
        event_type = event.event_type

        results = []

        if event_type == "event.created":
            results.append(ConnectorResult(
                operation="write",
                entity_path=f"integrations/myservice/{payload['project']}",
                key=f"event-{payload['id']}",
                value={
                    "type": payload["type"],
                    "summary": payload["summary"],
                    "severity": payload.get("severity", "info"),
                },
                confidence=0.7,
                memory_type=MemoryType.FACT,
            ))

        elif event_type == "event.deleted":
            results.append(ConnectorResult(
                operation="record_context",
                label=f"myservice-deletion-{payload['id']}",
                summary=f"Event {payload['id']} deleted from MyService",
                source="myservice",
            ))

        return results

    def validate_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        import hashlib
        import hmac
        expected = hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
```

### Step 4: Add Package Metadata

Create `pyproject.toml`:

```toml
[project]
name = "amfs-connector-myservice"
version = "1.0.0"
description = "MyService connector for AMFS"
requires-python = ">=3.10"
dependencies = ["amfs>=0.1.0"]

[project.entry-points."amfs.connectors"]
myservice = "amfs_connector_myservice.connector:MyServiceConnector"
```

### Step 5: Test with MockAMFS

Use `MockAMFS` to test your connector without a running AMFS instance:

```python
from amfs.testing import MockAMFS
from amfs_connector_myservice.connector import MyServiceConnector


def test_event_created():
    mock = MockAMFS()
    connector = MyServiceConnector(memory=mock)

    event = ConnectorEvent(
        event_type="event.created",
        payload={
            "id": "evt-123",
            "project": "my-project",
            "type": "alert",
            "summary": "CPU spike detected",
            "severity": "warning",
        },
        headers={"X-MyService-Signature": "abc123"},
    )

    results = connector.transform(event)

    assert len(results) == 1
    assert results[0].operation == "write"
    assert results[0].entity_path == "integrations/myservice/my-project"
    assert results[0].key == "event-evt-123"


def test_signature_validation():
    connector = MyServiceConnector(memory=MockAMFS())
    payload = b'{"test": true}'
    import hashlib, hmac
    secret = "test-secret"
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert connector.validate_signature(payload, sig, secret)
```

Run tests:

```bash
pytest tests/ -v
```

---

## Publishing Your Connector

### PyPI Naming Convention

Connector packages should follow the naming convention:

```
amfs-connector-{name}
```

Examples:
- `amfs-connector-pagerduty`
- `amfs-connector-github`
- `amfs-connector-datadog`
- `amfs-connector-opsgenie`

### GitHub Topic

Tag your repository with the `amfs-connector` topic on GitHub so it appears in connector searches:

```
https://github.com/topics/amfs-connector
```

### Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

Once published, users can install your connector with:

```bash
amfs connector install {name}
# or
pip install amfs-connector-{name}
```

---

## connector.yaml Reference

The `connector.yaml` manifest describes your connector to AMFS. Here is the full specification:

```yaml
# Required: unique connector name (lowercase, alphanumeric, hyphens)
name: myservice

# Required: semantic version
version: "1.0.0"

# Required: human-readable description
description: "Ingest events from MyService into AMFS"

# Optional: author name or organization
author: "Your Name <you@example.com>"

# Optional: license identifier (SPDX)
license: "Apache-2.0"

# Required: Python import path to the ConnectorABC subclass
entry_point: amfs_connector_myservice.connector:MyServiceConnector

# Webhook configuration
webhook:
  # Header name containing the HMAC signature
  signature_header: "X-MyService-Signature"

  # HMAC algorithm: hmac-sha256 | hmac-sha1 | hmac-sha512
  signature_algorithm: "hmac-sha256"

  # Optional: prefix stripped from signature header value before validation
  # e.g., PagerDuty uses "v1=" prefix
  signature_prefix: ""

  # Optional: maximum payload size in bytes (default: 1048576 = 1MB)
  max_payload_bytes: 1048576

# Default entity path for entries created by this connector
# Can be overridden per-event in the transform method
default_entity_path: "integrations/myservice"

# List of event types this connector handles
# Used for documentation and routing; the transform method receives the event_type
event_types:
  - "event.created"
  - "event.updated"
  - "event.deleted"

# Environment variables the connector expects
env_vars:
  - name: AMFS_CONNECTOR_MYSERVICE_SECRET
    description: "HMAC secret for webhook signature verification"
    required: true

  - name: AMFS_CONNECTOR_MYSERVICE_API_KEY
    description: "API key for pulling historical data (optional)"
    required: false

  - name: AMFS_CONNECTOR_MYSERVICE_ENTITY_PREFIX
    description: "Custom entity path prefix (overrides default_entity_path)"
    required: false

# Optional: minimum AMFS version required
min_amfs_version: "0.1.0"

# Optional: additional Python dependencies beyond amfs
dependencies:
  - "requests>=2.28.0"

# Optional: connector-specific configuration defaults
config:
  batch_size: 100
  retry_attempts: 3
  retry_delay_seconds: 5
```

| Field | Required | Description |
|:------|:--------:|:------------|
| `name` | Yes | Unique connector name. Lowercase, alphanumeric, hyphens only. |
| `version` | Yes | Semantic version string. |
| `description` | Yes | Human-readable description. |
| `author` | No | Author name or email. |
| `license` | No | SPDX license identifier. |
| `entry_point` | Yes | Python import path to the `ConnectorABC` subclass (`module.path:ClassName`). |
| `webhook.signature_header` | Yes | HTTP header containing the HMAC signature. |
| `webhook.signature_algorithm` | Yes | Algorithm for HMAC verification. |
| `webhook.signature_prefix` | No | Prefix to strip from the signature header value. |
| `webhook.max_payload_bytes` | No | Maximum payload size (default 1 MB). |
| `default_entity_path` | Yes | Default entity path for created entries. |
| `event_types` | Yes | List of event types the connector handles. |
| `env_vars` | No | Environment variables with name, description, and required flag. |
| `min_amfs_version` | No | Minimum compatible AMFS version. |
| `dependencies` | No | Additional Python dependencies. |
| `config` | No | Connector-specific configuration defaults. |
