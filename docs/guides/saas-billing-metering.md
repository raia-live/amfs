---
title: Hosted billing & metering
layout: default
parent: Guides
nav_order: 10
description: "How SenseLab-hosted AMFS bills ops, maps HTTP routes and MCP tools to costs, and integrates with the control plane."
permalink: /guides/saas-billing-metering/
---

# Hosted billing & metering
{: .no_toc }

This page describes the **contract** between the open-source **AMFS HTTP API / MCP server** and the proprietary **control plane** (`amfs-internal`) that records usage and enforces quotas on SenseLab Cloud.
{: .fs-6 .fw-300 }

## Ops model

| Operation class | Ops charged | Examples |
|:----------------|:-----------:|:---------|
| Read-like | **1** | `GET` entry, `search`, `list`, `explain` (read path), MCP `amfs_read`, `amfs_search`, … |
| Write-like | **2** | `POST` write entry, `record_context`, MCP `amfs_write`, … |
| Outcome commit | **0** | `commit_outcome` / MCP `amfs_commit_outcome` (product choice; still logged) |

## HTTP route → ops

The authoritative matrix for the hosted gateway is maintained in **`amfs-internal`** (`amfs_control_plane.metering` — `op_cost_for_request(method, path)`). It mirrors [amfs-http-server](https://github.com/raia-live/amfs/tree/main/packages/http-server) routes, for example:

- `GET /api/v1/entries/...` → 1  
- `POST /api/v1/entries/...` → 2  
- `POST /api/v1/outcomes` → 0  
- `GET /health` → 0  

Default for unmatched API paths: **1 op** (conservative).

## MCP tool → ops

Use `classify_mcp_tool(tool_name)` in the same module. Tools such as `amfs_read`, `amfs_search`, `amfs_list` → **1**; `amfs_write`, `amfs_record_context` → **2**; `amfs_commit_outcome` → **0**.

## Control plane API

The hosted edge should call (server-side only, with `X-AMFS-Internal-Key`):

`POST /api/v1/internal/usage/record` with JSON `{ "account_id": "<uuid>", "ops": <n> }`

Responses:

- **200** — updated counts (`ops_consumed`, `ops_remaining`, …)  
- **402** — `plan_required`, `quota_exceeded`, or over limit; body includes `upgrade_url` / `plan_comparison_url` where applicable  

Dashboard users receive usage hints via `GET /api/v1/me` plus response headers `X-AMFS-Ops-Remaining`, `X-AMFS-Ops-Limit`, and `X-AMFS-Usage-Warning` (80 / 95 / 100).

## Related

- [SaaS connection]({{ site.baseurl }}/guides/saas/) — API keys and `AMFS_HTTP_URL`  
- [OSS vs Pro]({{ site.baseurl }}/editions/) — feature comparison and hosted tiers summary  
