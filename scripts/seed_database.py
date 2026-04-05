"""
Comprehensive AMFS database seed script.

Seeds all tables with realistic, interconnected data so every dashboard
page has meaningful content to display and test.

Usage:
    AMFS_POSTGRES_DSN=postgresql://amfs:amfs@localhost:5432/amfs python scripts/seed_database.py
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import psycopg

DSN = os.environ.get(
    "AMFS_POSTGRES_DSN", "postgresql://amfs:amfs@localhost:5432/amfs"
)
NS = "default"

now = datetime.now(timezone.utc)


def ts(delta_hours: float = 0) -> datetime:
    return now - timedelta(hours=delta_hours)


def ts_iso(delta_hours: float = 0) -> str:
    return ts(delta_hours).isoformat()


def uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Identifiers (reused across tables for referential coherence)
# ---------------------------------------------------------------------------
AGENTS = [
    "deploy-agent",
    "review-agent",
    "monitoring-agent",
    "security-scanner",
    "perf-optimizer",
]

SESSIONS = {a: f"sess-{uuid.uuid4().hex[:8]}" for a in AGENTS}

ENTITIES = [
    "checkout-service",
    "auth",
    "payments",
    "user-service",
    "api-gateway",
    "notification-service",
    "inventory",
]

TEAM_IDS = [uid() for _ in range(3)]
TRACE_IDS = [uid() for _ in range(8)]


def seed_memory_entries(cur):
    """Seed amfs_memory_entries with diverse, multi-agent, multi-entity data."""
    print("  Seeding memory entries...")
    entries = [
        # checkout-service entries
        (NS, "checkout-service", "retry-pattern", 1, json.dumps({"max_retries": 3, "backoff": "exponential", "initial_delay_ms": 100}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(72), ["retry", "resilience"], 0.92, 3, None, "fact", "[]"),
        (NS, "checkout-service", "retry-pattern", 2, json.dumps({"max_retries": 5, "backoff": "exponential", "initial_delay_ms": 200, "max_delay_ms": 5000}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(24), ["retry", "resilience"], 0.88, 4, None, "fact", "[]"),
        (NS, "checkout-service", "timeout-config", 1, json.dumps({"timeout_ms": 5000, "read_timeout_ms": 3000, "write_timeout_ms": 8000}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(48), ["timeout"], 0.95, 2, None, "fact", "[]"),
        (NS, "checkout-service", "risk-race-condition", 1, json.dumps("Potential race condition on cart updates when multiple tabs are open"),
         "review-agent", SESSIONS["review-agent"], ts_iso(36), [], 0.65, 0, None, "belief", "[]"),
        (NS, "checkout-service", "circuit-breaker", 1, json.dumps({"failure_threshold": 5, "reset_timeout_ms": 30000, "half_open_requests": 3}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(60), ["circuit-breaker", "resilience"], 0.90, 2, None, "fact", "[]"),
        (NS, "checkout-service", "deploy-v2.14", 1, json.dumps({"version": "2.14.0", "deployed_at": ts_iso(12), "replicas": 3, "canary_pct": 10}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(12), ["deploy"], 0.97, 1, None, "experience", "[]"),

        # auth entries
        (NS, "auth", "session-timeout", 1, json.dumps({"timeout_ms": 30000, "refresh_window_ms": 5000, "max_sessions": 5}),
         "review-agent", SESSIONS["review-agent"], ts_iso(96), ["session"], 0.91, 2, None, "fact", "[]"),
        (NS, "auth", "jwt-rotation", 1, json.dumps({"rotation_days": 90, "algorithm": "RS256", "key_size": 2048}),
         "security-scanner", SESSIONS["security-scanner"], ts_iso(120), ["security", "jwt"], 0.98, 1, None, "fact", "[]"),
        (NS, "auth", "risk-token-leak", 1, json.dumps("Tokens stored in localStorage are vulnerable to XSS attacks"),
         "security-scanner", SESSIONS["security-scanner"], ts_iso(80), ["security"], 0.72, 0, None, "belief", "[]"),
        (NS, "auth", "rate-limiter", 1, json.dumps({"max_attempts": 5, "window_ms": 60000, "lockout_ms": 300000}),
         "security-scanner", SESSIONS["security-scanner"], ts_iso(100), ["rate-limit", "security"], 0.94, 3, None, "fact", "[]"),

        # payments entries
        (NS, "payments", "stripe-config", 1, json.dumps({"api_version": "2024-12-18", "webhook_tolerance_sec": 300, "retry_count": 3}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(200), ["payments", "stripe"], 0.96, 5, None, "fact", "[]"),
        (NS, "payments", "idempotency-pattern", 1, json.dumps({"key_ttl_hours": 24, "storage": "redis", "collision_strategy": "return_cached"}),
         "review-agent", SESSIONS["review-agent"], ts_iso(168), ["idempotency", "payments"], 0.93, 4, None, "fact", "[]"),
        (NS, "payments", "risk-double-charge", 1, json.dumps("Double charge possible during webhook retry + timeout overlap window"),
         "monitoring-agent", SESSIONS["monitoring-agent"], ts_iso(48), ["payments"], 0.58, 0, None, "belief", "[]"),
        (NS, "payments", "pci-compliance", 1, json.dumps({"last_audit": ts_iso(720), "next_audit": ts_iso(-2160), "scope": "SAQ-A"}),
         "security-scanner", SESSIONS["security-scanner"], ts_iso(720), ["compliance", "security"], 0.99, 2, None, "fact", "[]"),

        # user-service
        (NS, "user-service", "cache-strategy", 1, json.dumps({"type": "write-through", "ttl_sec": 3600, "invalidation": "event-driven"}),
         "perf-optimizer", SESSIONS["perf-optimizer"], ts_iso(150), ["caching", "performance"], 0.87, 2, None, "fact", "[]"),
        (NS, "user-service", "db-pool-config", 1, json.dumps({"min_connections": 5, "max_connections": 20, "idle_timeout_ms": 30000}),
         "perf-optimizer", SESSIONS["perf-optimizer"], ts_iso(140), ["database", "performance"], 0.91, 3, None, "fact", "[]"),

        # api-gateway
        (NS, "api-gateway", "rate-limits", 1, json.dumps({"global_rps": 10000, "per_user_rps": 100, "burst_size": 50}),
         "perf-optimizer", SESSIONS["perf-optimizer"], ts_iso(200), ["rate-limit", "gateway"], 0.95, 4, None, "fact", "[]"),
        (NS, "api-gateway", "cors-config", 1, json.dumps({"allowed_origins": ["*.example.com"], "max_age_sec": 86400}),
         "security-scanner", SESSIONS["security-scanner"], ts_iso(300), ["security", "gateway"], 0.97, 1, None, "fact", "[]"),

        # notification-service
        (NS, "notification-service", "email-provider", 1, json.dumps({"provider": "sendgrid", "daily_limit": 100000, "batch_size": 500}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(250), ["notifications"], 0.89, 2, None, "fact", "[]"),
        (NS, "notification-service", "risk-spam-loop", 1, json.dumps("Retry logic without dedup can cause notification spam when queue backs up"),
         "monitoring-agent", SESSIONS["monitoring-agent"], ts_iso(60), ["notifications"], 0.62, 0, None, "belief", "[]"),

        # inventory
        (NS, "inventory", "stock-sync", 1, json.dumps({"sync_interval_sec": 30, "source": "warehouse-api", "conflict_resolution": "latest_wins"}),
         "deploy-agent", SESSIONS["deploy-agent"], ts_iso(180), ["inventory", "sync"], 0.88, 2, None, "fact", "[]"),
        (NS, "inventory", "low-stock-threshold", 1, json.dumps({"warning_pct": 20, "critical_pct": 5, "auto_reorder": True}),
         "monitoring-agent", SESSIONS["monitoring-agent"], ts_iso(160), ["inventory", "monitoring"], 0.93, 3, None, "fact", "[]"),
    ]

    # Mark old versions as superseded
    cur.execute("DELETE FROM amfs_memory_entries WHERE namespace = %s", (NS,))

    for e in entries:
        cur.execute("""
            INSERT INTO amfs_memory_entries
                (namespace, entity_path, key, version, value, agent_id, session_id,
                 written_at, pattern_refs, confidence, outcome_count, ttl_at,
                 memory_type, artifact_refs, superseded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                    CASE WHEN %s = 1 AND %s = 'checkout-service' AND %s = 'retry-pattern' THEN %s::timestamptz ELSE NULL END)
        """, (*e, e[3], e[1], e[2], ts_iso(24)))  # supersede v1 of retry-pattern

    print(f"    Inserted {len(entries)} memory entries")


def seed_outcomes(cur):
    """Seed amfs_outcomes with various outcome types."""
    print("  Seeding outcomes...")
    cur.execute("DELETE FROM amfs_outcomes WHERE namespace = %s", (NS,))

    outcomes = [
        (NS, "DEP-100", "clean_deploy", 1.0, ts_iso(12), ["checkout-service/retry-pattern", "checkout-service/timeout-config"], "deploy-agent"),
        (NS, "DEP-101", "clean_deploy", 0.95, ts_iso(6), ["checkout-service/circuit-breaker", "checkout-service/deploy-v2.14"], "deploy-agent"),
        (NS, "REV-200", "regression", 0.85, ts_iso(36), ["auth/session-timeout"], "review-agent"),
        (NS, "SEC-300", "p2_incident", 0.90, ts_iso(80), ["auth/risk-token-leak", "auth/jwt-rotation"], "security-scanner"),
        (NS, "MON-400", "p1_incident", 0.95, ts_iso(48), ["payments/risk-double-charge", "payments/stripe-config"], "monitoring-agent"),
        (NS, "DEP-102", "clean_deploy", 1.0, ts_iso(2), ["api-gateway/rate-limits", "api-gateway/cors-config"], "deploy-agent"),
        (NS, "OPT-500", "clean_deploy", 0.92, ts_iso(18), ["user-service/cache-strategy", "user-service/db-pool-config"], "perf-optimizer"),
        (NS, "MON-401", "p2_incident", 0.80, ts_iso(60), ["notification-service/risk-spam-loop"], "monitoring-agent"),
    ]

    for o in outcomes:
        cur.execute("""
            INSERT INTO amfs_outcomes
                (namespace, outcome_ref, outcome_type, causal_confidence, committed_at, causal_entry_keys, agent_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, o)

    print(f"    Inserted {len(outcomes)} outcomes")


def seed_traces(cur):
    """Seed amfs_decision_traces with rich, realistic trace data."""
    print("  Seeding decision traces...")
    cur.execute("DELETE FROM amfs_decision_traces WHERE namespace = %s", (NS,))

    traces = [
        {
            "id": TRACE_IDS[0], "agent_id": "deploy-agent", "session_id": SESSIONS["deploy-agent"],
            "outcome_ref": "DEP-100", "outcome_type": "clean_deploy",
            "decision_summary": "Deployed checkout-service v2.14 after verifying retry pattern, timeout config, and checking PagerDuty for recent incidents.",
            "causal_entries": [
                {"entity_path": "checkout-service", "key": "retry-pattern", "version": 2, "confidence": 0.88,
                 "value": {"max_retries": 5, "backoff": "exponential"}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(12.5), "duration_ms": 2.3},
                {"entity_path": "checkout-service", "key": "timeout-config", "version": 1, "confidence": 0.95,
                 "value": {"timeout_ms": 5000}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(12.4), "duration_ms": 1.8},
                {"entity_path": "checkout-service", "key": "circuit-breaker", "version": 1, "confidence": 0.90,
                 "value": {"failure_threshold": 5}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(12.3), "duration_ms": 1.5},
            ],
            "external_contexts": [
                {"label": "pagerduty-incidents", "summary": "No active incidents. Last P1 was 14 days ago (payments timeout).", "source": "PagerDuty API", "recorded_at": ts_iso(12.5)},
                {"label": "deployment-pipeline", "summary": "All CI checks passed. 98.7% test coverage. No flaky tests.", "source": "GitHub Actions", "recorded_at": ts_iso(12.4)},
                {"label": "canary-metrics", "summary": "Canary at 10% traffic for 30min. Error rate 0.01%, p99 latency 45ms.", "source": "Datadog", "recorded_at": ts_iso(12.2)},
            ],
            "query_events": [
                {"operation": "search", "parameters": {"entity_path": "checkout-service", "min_confidence": 0.5}, "result_count": 5, "duration_ms": 12.4, "occurred_at": ts_iso(12.6)},
                {"operation": "list", "parameters": {"entity_path": "checkout-service"}, "result_count": 6, "duration_ms": 8.1, "occurred_at": ts_iso(12.5)},
            ],
            "error_events": [],
            "state_diff": {"entries_created": 1, "entries_updated": 1, "confidence_changes": [
                {"entity_path": "checkout-service", "key": "retry-pattern", "before": 0.90, "after": 0.88, "outcome_ref": "DEP-100"},
            ]},
            "session_started_at": ts_iso(13), "session_ended_at": ts_iso(12), "session_duration_ms": 3600000,
            "created_at": ts_iso(12),
        },
        {
            "id": TRACE_IDS[1], "agent_id": "review-agent", "session_id": SESSIONS["review-agent"],
            "outcome_ref": "REV-200", "outcome_type": "regression",
            "decision_summary": "Auth session timeout change caused regression. Users were logged out after 30s instead of 30min due to unit mismatch (ms vs sec).",
            "causal_entries": [
                {"entity_path": "auth", "key": "session-timeout", "version": 1, "confidence": 0.91,
                 "value": {"timeout_ms": 30000}, "memory_type": "fact",
                 "written_by": "review-agent", "read_at": ts_iso(36.5), "duration_ms": 3.1},
            ],
            "external_contexts": [
                {"label": "sentry-errors", "summary": "1,247 SessionExpiredError in last hour. All from checkout flow.", "source": "Sentry", "recorded_at": ts_iso(36.3)},
                {"label": "user-complaints", "summary": "42 Zendesk tickets about being logged out during checkout.", "source": "Zendesk API", "recorded_at": ts_iso(36.2)},
            ],
            "query_events": [
                {"operation": "search", "parameters": {"entity_path": "auth", "sort_by": "recency"}, "result_count": 4, "duration_ms": 15.2, "occurred_at": ts_iso(37)},
            ],
            "error_events": [
                {"operation": "read", "error_type": "TimeoutError", "message": "Read timed out after 5000ms on first attempt", "occurred_at": ts_iso(37.1)},
            ],
            "state_diff": {"entries_created": 0, "entries_updated": 1, "confidence_changes": [
                {"entity_path": "auth", "key": "session-timeout", "before": 0.95, "after": 0.91, "outcome_ref": "REV-200"},
            ]},
            "session_started_at": ts_iso(38), "session_ended_at": ts_iso(36), "session_duration_ms": 7200000,
            "created_at": ts_iso(36),
        },
        {
            "id": TRACE_IDS[2], "agent_id": "security-scanner", "session_id": SESSIONS["security-scanner"],
            "outcome_ref": "SEC-300", "outcome_type": "p2_incident",
            "decision_summary": "Token leak vulnerability detected in localStorage. JWT rotation scheduled. Coordinated with auth team for emergency key rotation.",
            "causal_entries": [
                {"entity_path": "auth", "key": "risk-token-leak", "version": 1, "confidence": 0.72,
                 "value": "Tokens stored in localStorage are vulnerable to XSS attacks", "memory_type": "belief",
                 "written_by": "security-scanner", "read_at": ts_iso(80.5), "duration_ms": 2.0},
                {"entity_path": "auth", "key": "jwt-rotation", "version": 1, "confidence": 0.98,
                 "value": {"rotation_days": 90, "algorithm": "RS256"}, "memory_type": "fact",
                 "written_by": "security-scanner", "read_at": ts_iso(80.4), "duration_ms": 1.9},
                {"entity_path": "auth", "key": "rate-limiter", "version": 1, "confidence": 0.94,
                 "value": {"max_attempts": 5}, "memory_type": "fact",
                 "written_by": "security-scanner", "read_at": ts_iso(80.3), "duration_ms": 1.7},
            ],
            "external_contexts": [
                {"label": "cve-database", "summary": "CVE-2025-31337: XSS via localStorage token theft. CVSS 7.5 High.", "source": "NVD API", "recorded_at": ts_iso(80.6)},
                {"label": "penetration-test", "summary": "Confirmed XSS vector in profile page. Attacker can exfiltrate JWT.", "source": "HackerOne Report #4521", "recorded_at": ts_iso(80.5)},
            ],
            "query_events": [
                {"operation": "search", "parameters": {"pattern_ref": "security", "min_confidence": 0.3}, "result_count": 6, "duration_ms": 18.5, "occurred_at": ts_iso(81)},
            ],
            "error_events": [],
            "state_diff": {"entries_created": 1, "entries_updated": 0, "confidence_changes": []},
            "session_started_at": ts_iso(82), "session_ended_at": ts_iso(80), "session_duration_ms": 7200000,
            "created_at": ts_iso(80),
        },
        {
            "id": TRACE_IDS[3], "agent_id": "monitoring-agent", "session_id": SESSIONS["monitoring-agent"],
            "outcome_ref": "MON-400", "outcome_type": "p1_incident",
            "decision_summary": "Double charge incident in payments. Root cause: webhook retry overlapping with timeout fallback. 23 affected customers, $4,521 in duplicate charges.",
            "causal_entries": [
                {"entity_path": "payments", "key": "risk-double-charge", "version": 1, "confidence": 0.58,
                 "value": "Double charge possible during webhook retry + timeout overlap window", "memory_type": "belief",
                 "written_by": "monitoring-agent", "read_at": ts_iso(48.5), "duration_ms": 2.5},
                {"entity_path": "payments", "key": "stripe-config", "version": 1, "confidence": 0.96,
                 "value": {"api_version": "2024-12-18", "retry_count": 3}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(48.4), "duration_ms": 2.1},
                {"entity_path": "payments", "key": "idempotency-pattern", "version": 1, "confidence": 0.93,
                 "value": {"key_ttl_hours": 24, "storage": "redis"}, "memory_type": "fact",
                 "written_by": "review-agent", "read_at": ts_iso(48.3), "duration_ms": 1.8},
            ],
            "external_contexts": [
                {"label": "stripe-dashboard", "summary": "23 duplicate charges detected in last 2h. Total: $4,521.00.", "source": "Stripe Dashboard", "recorded_at": ts_iso(48.3)},
                {"label": "pagerduty-incident", "summary": "P1 triggered: INC-2847 'Duplicate payment charges in checkout'. On-call: @sarah.", "source": "PagerDuty API", "recorded_at": ts_iso(48.2)},
                {"label": "redis-metrics", "summary": "Redis idempotency key TTL expired for 23 transactions due to clock skew.", "source": "Redis Insight", "recorded_at": ts_iso(48.1)},
            ],
            "query_events": [
                {"operation": "search", "parameters": {"entity_path": "payments"}, "result_count": 4, "duration_ms": 9.8, "occurred_at": ts_iso(49)},
                {"operation": "search", "parameters": {"query": "double charge", "min_confidence": 0.3}, "result_count": 1, "duration_ms": 22.3, "occurred_at": ts_iso(48.8)},
            ],
            "error_events": [
                {"operation": "tool", "error_type": "WebhookValidationError", "message": "Stripe webhook signature mismatch on 3 events", "occurred_at": ts_iso(48.4)},
                {"operation": "adapter", "error_type": "ConnectionPoolExhausted", "message": "All 20 Postgres connections in use during incident spike", "occurred_at": ts_iso(48.2)},
            ],
            "state_diff": {"entries_created": 0, "entries_updated": 2, "confidence_changes": [
                {"entity_path": "payments", "key": "risk-double-charge", "before": 0.45, "after": 0.58, "outcome_ref": "MON-400"},
                {"entity_path": "payments", "key": "idempotency-pattern", "before": 0.95, "after": 0.93, "outcome_ref": "MON-400"},
            ]},
            "session_started_at": ts_iso(50), "session_ended_at": ts_iso(48), "session_duration_ms": 7200000,
            "created_at": ts_iso(48),
        },
        {
            "id": TRACE_IDS[4], "agent_id": "deploy-agent", "session_id": SESSIONS["deploy-agent"],
            "outcome_ref": "DEP-101", "outcome_type": "clean_deploy",
            "decision_summary": "Deployed circuit breaker update with canary. All health checks green.",
            "causal_entries": [
                {"entity_path": "checkout-service", "key": "circuit-breaker", "version": 1, "confidence": 0.90,
                 "value": {"failure_threshold": 5, "reset_timeout_ms": 30000}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(6.5), "duration_ms": 1.4},
            ],
            "external_contexts": [
                {"label": "health-checks", "summary": "All 3 replicas healthy. CPU 45%, Memory 62%.", "source": "Kubernetes", "recorded_at": ts_iso(6.3)},
            ],
            "query_events": [],
            "error_events": [],
            "state_diff": {"entries_created": 0, "entries_updated": 0, "confidence_changes": []},
            "session_started_at": ts_iso(7), "session_ended_at": ts_iso(6), "session_duration_ms": 3600000,
            "created_at": ts_iso(6),
        },
        {
            "id": TRACE_IDS[5], "agent_id": "perf-optimizer", "session_id": SESSIONS["perf-optimizer"],
            "outcome_ref": "OPT-500", "outcome_type": "clean_deploy",
            "decision_summary": "Optimized user-service cache strategy and DB pool. p99 latency improved from 120ms to 45ms.",
            "causal_entries": [
                {"entity_path": "user-service", "key": "cache-strategy", "version": 1, "confidence": 0.87,
                 "value": {"type": "write-through", "ttl_sec": 3600}, "memory_type": "fact",
                 "written_by": "perf-optimizer", "read_at": ts_iso(18.5), "duration_ms": 2.8},
                {"entity_path": "user-service", "key": "db-pool-config", "version": 1, "confidence": 0.91,
                 "value": {"min_connections": 5, "max_connections": 20}, "memory_type": "fact",
                 "written_by": "perf-optimizer", "read_at": ts_iso(18.4), "duration_ms": 2.2},
            ],
            "external_contexts": [
                {"label": "apm-metrics", "summary": "p99 latency: 120ms → 45ms. Cache hit ratio: 34% → 89%.", "source": "Datadog APM", "recorded_at": ts_iso(18.3)},
                {"label": "load-test", "summary": "Sustained 5000 RPS for 10min. No errors. Max CPU 72%.", "source": "k6 Load Test", "recorded_at": ts_iso(18.2)},
            ],
            "query_events": [
                {"operation": "search", "parameters": {"pattern_ref": "performance"}, "result_count": 3, "duration_ms": 11.2, "occurred_at": ts_iso(19)},
            ],
            "error_events": [],
            "state_diff": {"entries_created": 0, "entries_updated": 2, "confidence_changes": [
                {"entity_path": "user-service", "key": "cache-strategy", "before": 0.82, "after": 0.87, "outcome_ref": "OPT-500"},
                {"entity_path": "user-service", "key": "db-pool-config", "before": 0.85, "after": 0.91, "outcome_ref": "OPT-500"},
            ]},
            "session_started_at": ts_iso(20), "session_ended_at": ts_iso(18), "session_duration_ms": 7200000,
            "created_at": ts_iso(18),
        },
        {
            "id": TRACE_IDS[6], "agent_id": "monitoring-agent", "session_id": SESSIONS["monitoring-agent"],
            "outcome_ref": "MON-401", "outcome_type": "p2_incident",
            "decision_summary": "Notification spam loop detected. 12,000 duplicate emails sent in 15 minutes due to missing dedup in retry queue.",
            "causal_entries": [
                {"entity_path": "notification-service", "key": "risk-spam-loop", "version": 1, "confidence": 0.62,
                 "value": "Retry logic without dedup can cause notification spam", "memory_type": "belief",
                 "written_by": "monitoring-agent", "read_at": ts_iso(60.5), "duration_ms": 2.0},
                {"entity_path": "notification-service", "key": "email-provider", "version": 1, "confidence": 0.89,
                 "value": {"provider": "sendgrid", "daily_limit": 100000}, "memory_type": "fact",
                 "written_by": "deploy-agent", "read_at": ts_iso(60.4), "duration_ms": 1.6},
            ],
            "external_contexts": [
                {"label": "sendgrid-stats", "summary": "12,847 emails sent in 15min window. Normal: ~200/15min.", "source": "SendGrid API", "recorded_at": ts_iso(60.3)},
            ],
            "query_events": [
                {"operation": "list", "parameters": {"entity_path": "notification-service"}, "result_count": 2, "duration_ms": 7.3, "occurred_at": ts_iso(61)},
            ],
            "error_events": [
                {"operation": "tool", "error_type": "RateLimitExceeded", "message": "SendGrid rate limit hit: 429 Too Many Requests", "occurred_at": ts_iso(60.2)},
            ],
            "state_diff": {"entries_created": 0, "entries_updated": 1, "confidence_changes": [
                {"entity_path": "notification-service", "key": "risk-spam-loop", "before": 0.50, "after": 0.62, "outcome_ref": "MON-401"},
            ]},
            "session_started_at": ts_iso(62), "session_ended_at": ts_iso(60), "session_duration_ms": 7200000,
            "created_at": ts_iso(60),
        },
        {
            "id": TRACE_IDS[7], "agent_id": "deploy-agent", "session_id": SESSIONS["deploy-agent"],
            "outcome_ref": "DEP-102", "outcome_type": "clean_deploy",
            "decision_summary": "API gateway rate limit and CORS config updated. Load tested at 15k RPS with no issues.",
            "causal_entries": [
                {"entity_path": "api-gateway", "key": "rate-limits", "version": 1, "confidence": 0.95,
                 "value": {"global_rps": 10000, "per_user_rps": 100}, "memory_type": "fact",
                 "written_by": "perf-optimizer", "read_at": ts_iso(2.5), "duration_ms": 1.3},
                {"entity_path": "api-gateway", "key": "cors-config", "version": 1, "confidence": 0.97,
                 "value": {"allowed_origins": ["*.example.com"]}, "memory_type": "fact",
                 "written_by": "security-scanner", "read_at": ts_iso(2.4), "duration_ms": 1.1},
            ],
            "external_contexts": [
                {"label": "load-test-results", "summary": "15,000 RPS sustained. p99: 12ms. Zero 5xx.", "source": "k6", "recorded_at": ts_iso(2.3)},
            ],
            "query_events": [],
            "error_events": [],
            "state_diff": {"entries_created": 0, "entries_updated": 0, "confidence_changes": []},
            "session_started_at": ts_iso(3), "session_ended_at": ts_iso(2), "session_duration_ms": 3600000,
            "created_at": ts_iso(2),
        },
    ]

    for t in traces:
        cur.execute("""
            INSERT INTO amfs_decision_traces
                (id, namespace, agent_id, session_id, outcome_ref, outcome_type,
                 decision_summary, causal_entries, external_contexts,
                 query_events, error_events, state_diff,
                 session_started_at, session_ended_at, session_duration_ms, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            t["id"], NS, t["agent_id"], t["session_id"],
            t["outcome_ref"], t["outcome_type"], t["decision_summary"],
            json.dumps(t["causal_entries"]), json.dumps(t["external_contexts"]),
            json.dumps(t["query_events"], default=str),
            json.dumps(t["error_events"], default=str),
            json.dumps(t["state_diff"]) if t["state_diff"] else None,
            t["session_started_at"], t["session_ended_at"], t["session_duration_ms"],
            t["created_at"],
        ))

    print(f"    Inserted {len(traces)} decision traces")


def seed_detected_patterns(cur):
    """Seed amfs_detected_patterns with various pattern types and severities."""
    print("  Seeding detected patterns...")
    cur.execute("DELETE FROM amfs_detected_patterns WHERE namespace = %s", (NS,))

    patterns = [
        (NS, "recurring_failure", "critical", "payments",
         "Recurring payment failures during high-traffic windows",
         json.dumps({"occurrences": 23, "last_seen": ts_iso(48), "affected_endpoints": ["/api/checkout", "/api/payment/confirm"],
                      "error_rate_pct": 2.3, "recommendation": "Implement circuit breaker with fallback payment queue"}),
         False, ts_iso(48), None),
        (NS, "hot_entity", "warning", "checkout-service",
         "checkout-service has 340 reads and 45 writes in the last 24 hours",
         json.dumps({"reads_24h": 340, "writes_24h": 45, "unique_agents": 4,
                      "top_keys": ["retry-pattern", "timeout-config", "circuit-breaker"],
                      "recommendation": "Consider read-through cache for frequently accessed keys"}),
         False, ts_iso(24), None),
        (NS, "stale_cluster", "info", "notification-service",
         "2 entries in notification-service haven't been read or updated in 30+ days",
         json.dumps({"stale_keys": ["legacy-sms-config", "old-template-v1"], "last_access_days_ago": 45,
                      "recommendation": "Archive or remove stale entries to reduce noise"}),
         True, ts_iso(720), ts_iso(168)),
        (NS, "confidence_drift", "warning", "auth",
         "auth/session-timeout confidence dropped from 0.95 to 0.65 over 3 outcome events",
         json.dumps({"entity_path": "auth", "key": "session-timeout", "initial_confidence": 0.95,
                      "current_confidence": 0.65, "drift_pct": -31.6, "outcome_count": 3,
                      "outcomes": ["REV-200 (regression)", "SEC-300 (p2_incident)"],
                      "recommendation": "Review and re-validate session timeout configuration"}),
         False, ts_iso(36), None),
        (NS, "recurring_failure", "warning", "api-gateway",
         "Intermittent 503 errors from api-gateway during deployments",
         json.dumps({"occurrences": 8, "last_seen": ts_iso(6), "pattern": "Occurs during rolling deployments",
                      "avg_duration_sec": 12, "recommendation": "Increase deployment surge capacity or use blue-green deployments"}),
         True, ts_iso(168), ts_iso(6)),
        (NS, "hot_entity", "info", "auth",
         "auth entity accessed by 4 different agents in the last 48 hours",
         json.dumps({"reads_48h": 156, "writes_48h": 12, "unique_agents": 4,
                      "agents": ["review-agent", "security-scanner", "deploy-agent", "monitoring-agent"],
                      "recommendation": "Normal for security-sensitive entities — no action needed"}),
         False, ts_iso(48), None),
        (NS, "confidence_drift", "critical", "payments",
         "payments/risk-double-charge confidence increased from 0.30 to 0.72 — risk is materializing",
         json.dumps({"entity_path": "payments", "key": "risk-double-charge", "initial_confidence": 0.30,
                      "current_confidence": 0.72, "drift_pct": 140.0, "outcome_count": 2,
                      "outcomes": ["MON-400 (p1_incident)"],
                      "recommendation": "URGENT: Implement idempotency key with longer TTL and clock-skew tolerance"}),
         False, ts_iso(48), None),
    ]

    for p in patterns:
        cur.execute("""
            INSERT INTO amfs_detected_patterns
                (namespace, pattern_type, severity, entity_path, description, details, resolved, detected_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, p)

    print(f"    Inserted {len(patterns)} detected patterns")


def seed_teams(cur):
    """Seed amfs_teams and amfs_team_members."""
    print("  Seeding teams...")
    cur.execute("DELETE FROM amfs_team_members WHERE namespace = %s", (NS,))
    cur.execute("DELETE FROM amfs_teams WHERE namespace = %s", (NS,))

    teams = [
        (TEAM_IDS[0], NS, "Platform Engineering", "platform-engineering",
         "Core platform team responsible for infrastructure, CI/CD, and deployment automation.",
         ts_iso(2000), ts_iso(24)),
        (TEAM_IDS[1], NS, "Security & Compliance", "security-compliance",
         "Application security, penetration testing, PCI compliance, and incident response.",
         ts_iso(1800), ts_iso(80)),
        (TEAM_IDS[2], NS, "Product Backend", "product-backend",
         "Backend services for checkout, payments, user management, and notifications.",
         ts_iso(1500), ts_iso(12)),
    ]

    for t in teams:
        cur.execute("""
            INSERT INTO amfs_teams (id, namespace, name, slug, description, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, t)

    members = [
        # Platform Engineering
        (NS, TEAM_IDS[0], "sarah@example.com", "Sarah Chen", "admin", ts_iso(2000), ts_iso(1990)),
        (NS, TEAM_IDS[0], "marcus@example.com", "Marcus Johnson", "developer", ts_iso(1500), ts_iso(1490)),
        (NS, TEAM_IDS[0], "deploy-agent@agents.amfs", "Deploy Agent", "developer", ts_iso(1000), ts_iso(999)),
        (NS, TEAM_IDS[0], "perf-optimizer@agents.amfs", "Performance Optimizer", "developer", ts_iso(800), ts_iso(799)),
        # Security & Compliance
        (NS, TEAM_IDS[1], "alex@example.com", "Alex Rivera", "admin", ts_iso(1800), ts_iso(1790)),
        (NS, TEAM_IDS[1], "security-scanner@agents.amfs", "Security Scanner", "developer", ts_iso(1200), ts_iso(1199)),
        (NS, TEAM_IDS[1], "priya@example.com", "Priya Patel", "viewer", ts_iso(600), ts_iso(595)),
        # Product Backend
        (NS, TEAM_IDS[2], "james@example.com", "James Wilson", "admin", ts_iso(1500), ts_iso(1490)),
        (NS, TEAM_IDS[2], "review-agent@agents.amfs", "Review Agent", "developer", ts_iso(1000), ts_iso(999)),
        (NS, TEAM_IDS[2], "monitoring-agent@agents.amfs", "Monitoring Agent", "developer", ts_iso(900), ts_iso(899)),
        (NS, TEAM_IDS[2], "lisa@example.com", "Lisa Park", "developer", ts_iso(700), ts_iso(695)),
        (NS, TEAM_IDS[2], "david@example.com", "David Kim", "viewer", ts_iso(400), None),
    ]

    for m in members:
        cur.execute("""
            INSERT INTO amfs_team_members
                (namespace, team_id, email, display_name, role, invited_at, accepted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, m)

    print(f"    Inserted {len(teams)} teams, {len(members)} members")


def seed_api_keys(cur):
    """Seed amfs_api_keys with test keys."""
    print("  Seeding API keys...")
    cur.execute("DELETE FROM amfs_api_keys WHERE namespace = %s", (NS,))

    keys = [
        (NS, "Production Agent Key", hashlib.sha256(b"amfs_prod_key_001").hexdigest(),
         "amfs_pk_", "agent", True, json.dumps(["read", "write", "search"]), 120, ts_iso(2), ts_iso(500), None),
        (NS, "CI/CD Pipeline Key", hashlib.sha256(b"amfs_cicd_key_002").hexdigest(),
         "amfs_ci_", "agent", True, json.dumps(["read", "write", "search", "commit_outcome"]), 60, ts_iso(6), ts_iso(400), None),
        (NS, "Dashboard Read-Only", hashlib.sha256(b"amfs_dash_key_003").hexdigest(),
         "amfs_ro_", "viewer", True, json.dumps(["read", "search", "list"]), 300, ts_iso(1), ts_iso(300), None),
        (NS, "Deprecated Test Key", hashlib.sha256(b"amfs_test_key_004").hexdigest(),
         "amfs_ts_", "agent", False, json.dumps(["read"]), 10, ts_iso(720), ts_iso(1000), ts_iso(168)),
        (NS, "Admin Master Key", hashlib.sha256(b"amfs_admin_key_005").hexdigest(),
         "amfs_ad_", "admin", True, json.dumps(["read", "write", "search", "admin", "commit_outcome"]), 60, ts_iso(0.5), ts_iso(200), None),
    ]

    for k in keys:
        cur.execute("""
            INSERT INTO amfs_api_keys
                (namespace, name, key_hash, prefix, key_type, active, scopes, rate_limit_rpm,
                 last_used, created_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
        """, k)

    print(f"    Inserted {len(keys)} API keys")


def seed_audit_log(cur):
    """Seed amfs_audit_log with realistic audit events."""
    print("  Seeding audit log...")
    cur.execute("DELETE FROM amfs_audit_log WHERE namespace = %s", (NS,))

    events = [
        (NS, "agent", "deploy-agent", "write", "checkout-service/retry-pattern",
         "10.0.1.15", json.dumps({"version": 2, "confidence": 0.88}), ts_iso(24)),
        (NS, "agent", "deploy-agent", "commit_outcome", "DEP-100",
         "10.0.1.15", json.dumps({"outcome_type": "clean_deploy", "affected_entries": 2}), ts_iso(12)),
        (NS, "agent", "security-scanner", "write", "auth/risk-token-leak",
         "10.0.2.30", json.dumps({"memory_type": "belief", "confidence": 0.72}), ts_iso(80)),
        (NS, "agent", "security-scanner", "commit_outcome", "SEC-300",
         "10.0.2.30", json.dumps({"outcome_type": "p2_incident", "affected_entries": 2}), ts_iso(80)),
        (NS, "user", "sarah@example.com", "create_team", "platform-engineering",
         "192.168.1.100", json.dumps({"team_name": "Platform Engineering"}), ts_iso(2000)),
        (NS, "user", "alex@example.com", "create_api_key", "Production Agent Key",
         "192.168.1.101", json.dumps({"key_type": "agent", "scopes": ["read", "write", "search"]}), ts_iso(500)),
        (NS, "agent", "monitoring-agent", "commit_outcome", "MON-400",
         "10.0.3.45", json.dumps({"outcome_type": "p1_incident", "severity": "critical"}), ts_iso(48)),
        (NS, "system", "pattern-detector", "detect_pattern", "payments/recurring_failure",
         None, json.dumps({"pattern_type": "recurring_failure", "severity": "critical"}), ts_iso(48)),
        (NS, "user", "james@example.com", "resolve_pattern", "notification-service/stale_cluster",
         "192.168.1.102", json.dumps({"resolution": "Archived stale entries"}), ts_iso(168)),
        (NS, "agent", "perf-optimizer", "write", "user-service/cache-strategy",
         "10.0.4.60", json.dumps({"version": 1, "confidence": 0.87}), ts_iso(150)),
        (NS, "agent", "perf-optimizer", "commit_outcome", "OPT-500",
         "10.0.4.60", json.dumps({"outcome_type": "clean_deploy", "latency_improvement": "63%"}), ts_iso(18)),
        (NS, "user", "sarah@example.com", "revoke_api_key", "Deprecated Test Key",
         "192.168.1.100", json.dumps({"reason": "Key compromised in test environment"}), ts_iso(168)),
        (NS, "agent", "review-agent", "commit_outcome", "REV-200",
         "10.0.5.75", json.dumps({"outcome_type": "regression", "impact": "1247 users affected"}), ts_iso(36)),
        (NS, "system", "pattern-detector", "detect_pattern", "auth/confidence_drift",
         None, json.dumps({"pattern_type": "confidence_drift", "drift_pct": -31.6}), ts_iso(36)),
        (NS, "agent", "deploy-agent", "commit_outcome", "DEP-102",
         "10.0.1.15", json.dumps({"outcome_type": "clean_deploy", "load_test_rps": 15000}), ts_iso(2)),
    ]

    for e in events:
        cur.execute("""
            INSERT INTO amfs_audit_log
                (namespace, actor_type, actor_name, action, resource, ip_address, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        """, e)

    print(f"    Inserted {len(events)} audit log entries")


def main():
    print(f"Connecting to: {DSN}")
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            print("Seeding AMFS database with comprehensive test data...\n")
            seed_memory_entries(cur)
            seed_outcomes(cur)
            seed_traces(cur)
            seed_detected_patterns(cur)
            seed_teams(cur)
            seed_api_keys(cur)
            seed_audit_log(cur)

        conn.commit()

    print("\nDone! All tables seeded successfully.")
    print("\nSummary:")
    print("  - 22 memory entries across 7 entities, 5 agents")
    print("  -  8 outcomes (clean_deploy, regression, p1/p2_incident)")
    print("  -  8 decision traces with rich causal chains, external contexts, query/error events")
    print("  -  7 detected patterns (recurring_failure, hot_entity, stale_cluster, confidence_drift)")
    print("  -  3 teams with 12 members (admins, developers, viewers, agents)")
    print("  -  5 API keys (active, inactive, various scopes)")
    print("  - 15 audit log entries")


if __name__ == "__main__":
    main()
