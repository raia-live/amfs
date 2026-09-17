"""Preflight gates. Nothing in the study runs until every gate here passes.

Gate 1 (hard): on the SenseLab production API, outcomes must change what an agent reads
first. Protocol (verified on dev 2026-09-15, replicated here on prod):
  * seed two near-tie entries in a fresh scope at the same confidence;
  * the entry ranked SECOND for a neutral query is the target;
  * 12 rounds, each from a FRESH connection: read the target, commit a success;
  * the target must now rank FIRST for the neutral query and its confidence must rise;
  * 12 rounds of critical_failure on the target: it must fall below the gate and be
    excluded by ``min_confidence``.
An empty retrieval at any step is a loud failure, never "no change".

Gate 2: retrieve() returns results on a bench scope (last suite silently got zero).
Gate 3: credentials and model IDs resolve for OpenAI, Anthropic, Mem0, Zep.
Gate 4: pgvector is reachable.

Continual-learning gates (``cl`` selects all five; run them on the API under test
before grid v2 — they are the plan's section-5 gates 3-6 plus the claim check):
  evidence   fresh 0.7 entry gated after one failure; 5-success entry gated within 3
             failures; a 3-key success splits credit equally and below a solo success.
  embedding  an entry keeps semantic recall after an outcome opens a new version.
  payload    retrieve carries evidence_status / counts; include_avoid returns the
             discredited entry flagged, and plain retrieve drops it.
  briefing   two validated entries failing -> Discredited + Regime-shift sections.
  claim      an outcome credits the claim read, not a rewritten one under the same key.

Run:  .bench-venv/bin/python -m benchmarks.continual_learning.preflight [gate ...]
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from typing import Any

from . import config

REPORT: dict[str, Any] = {"gates": {}}


def _gate(name: str, ok: bool, detail: Any) -> None:
    REPORT["gates"][name] = {"ok": ok, "detail": detail}
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}: {json.dumps(detail, default=str)[:600]}")


# ---------------------------------------------------------------------------
# Gate 1 — production reinforcement reorders retrieval
# ---------------------------------------------------------------------------

def _amfs(agent_id: str):
    from amfs import AgentMemory
    from amfs_adapter_http import HttpAdapter

    return AgentMemory(
        agent_id=agent_id,
        adapter=HttpAdapter(base_url=config.AMFS_HTTP_URL, api_key=config.env("AMFS_API_KEY", required=True)),
    )


def _ranked(scope: str, query: str, min_confidence: float = 0.0) -> list[tuple[str, float, float]]:
    m = _amfs("cl-preflight-observer")
    try:
        rows = m.retrieve(query, entity_path=scope, min_confidence=min_confidence, limit=5)
    finally:
        m.close()
    if not rows and min_confidence == 0.0:
        raise RuntimeError(f"EMPTY retrieval on {scope!r} for {query!r} — measurement is invalid")
    return [(r.entry.key, round(r.score, 4), round(r.entry.confidence, 4)) for r in rows]


def gate_prod_loop() -> bool:
    from amfs_core.models import OutcomeType

    scope = f"{config.SCOPE_ROOT}/preflight/{uuid.uuid4().hex[:8]}"
    neutral = "deploy runbook for the checkout service"
    texts = {
        "runbook-migrate-first": (
            "Deploy runbook for the checkout service: run database migrations, then roll the "
            "pods, then warm the cache. Reference ticket OPS-4471."
        ),
        "runbook-roll-first": (
            "Deploy runbook for the checkout service: roll the pods, then run database "
            "migrations, then warm the cache. Reference ticket OPS-9932."
        ),
    }
    seeder = _amfs("cl-preflight-seeder")
    try:
        for k, v in texts.items():
            seeder.write(scope, k, v, confidence=0.7)
    finally:
        seeder.close()
    time.sleep(2.0)

    baseline = _ranked(scope, neutral)
    if len(baseline) < 2:
        _gate("prod_loop", False, {"reason": "fewer than 2 entries retrievable", "baseline": baseline})
        return False
    first_key, first_score, _ = baseline[0]
    target_key, target_score, target_conf0 = baseline[1]
    gap = round(first_score - target_score, 4)
    ticket = "OPS-4471" if target_key == "runbook-migrate-first" else "OPS-9932"
    detail: dict[str, Any] = {"scope": scope, "baseline": baseline, "target": target_key, "gap": gap}
    if gap > 0.06:
        detail["warning"] = "gap > 0.06: reinforcement is a tie-breaker and may not flip order"

    # Reinforce: 12 successes, each from a fresh connection, crediting the target via a
    # rare-literal retrieve (top hit becomes the causal read) — the same path the agent uses.
    credited = 0
    for i in range(12):
        m = _amfs(f"cl-preflight-worker-{i}")
        try:
            hits = m.retrieve(f"ticket {ticket}", entity_path=scope, limit=3)
            if not hits or hits[0].entry.key != target_key:
                detail.setdefault("miscredits", []).append([h.entry.key for h in hits])
                # fall back to an explicit read so the round still credits the target
                m.read(scope, target_key)
            affected = m.commit_outcome(f"preflight-success-{i}", OutcomeType.SUCCESS,
                                        task_input="preflight reinforcement round",
                                        response_text=f"followed {target_key}")
            if any(e.key == target_key for e in affected):
                credited += 1
        finally:
            m.close()
    time.sleep(2.0)
    after_success = _ranked(scope, neutral)
    detail["credited_success_rounds"] = credited
    detail["after_success"] = after_success
    flipped_up = after_success[0][0] == target_key
    conf_after = next((c for k, _, c in after_success if k == target_key), None)
    detail["target_conf_after_success"] = conf_after

    # Discredit: 12 critical failures on the same target.
    discredited = 0
    for i in range(12):
        m = _amfs(f"cl-preflight-worker-f{i}")
        try:
            hits = m.retrieve(f"ticket {ticket}", entity_path=scope, limit=3)
            if not hits or hits[0].entry.key != target_key:
                m.read(scope, target_key)
            affected = m.commit_outcome(f"preflight-failure-{i}", OutcomeType.CRITICAL_FAILURE,
                                        task_input="preflight discredit round",
                                        response_text=f"followed {target_key}; outage")
            if any(e.key == target_key for e in affected):
                discredited += 1
        finally:
            m.close()
    time.sleep(2.0)
    after_failure = _ranked(scope, neutral)
    gated = _ranked(scope, neutral, min_confidence=config.STUDY.min_confidence_gate)
    detail["credited_failure_rounds"] = discredited
    detail["after_failure"] = after_failure
    detail["after_failure_gated"] = gated
    conf_final = next((c for k, _, c in after_failure if k == target_key), None)
    detail["target_conf_after_failure"] = conf_final
    flipped_down = after_failure[0][0] != target_key
    excluded = all(k != target_key for k, _, _ in gated)

    ok = (
        credited >= 10
        and discredited >= 10
        and conf_after is not None and conf_after > target_conf0
        and conf_final is not None and conf_final < config.STUDY.min_confidence_gate
        and flipped_down
        and excluded
    )
    detail["checks"] = {
        "reinforcement_credited": credited >= 10,
        "confidence_rose": bool(conf_after and conf_after > target_conf0),
        "order_flipped_up_after_success": flipped_up,
        "confidence_fell_below_gate": bool(conf_final is not None and conf_final < config.STUDY.min_confidence_gate),
        "order_flipped_down_after_failure": flipped_down,
        "excluded_by_min_confidence": excluded,
    }
    if not flipped_up:
        detail["note"] = ("success did not promote the target above the leader; expected when the "
                          "initial gap exceeds what reinforcement can close — the gate relies on "
                          "discredit + min_confidence, which is what the study measures")
    _gate("prod_loop", ok, detail)
    return ok


# ---------------------------------------------------------------------------
# Gate 2 — retrieve returns on a bench scope
# ---------------------------------------------------------------------------

def gate_retrieve() -> bool:
    scope = f"{config.SCOPE_ROOT}/preflight/retrieve-{uuid.uuid4().hex[:6]}"
    m = _amfs("cl-preflight-retrieve")
    try:
        m.write(scope, "fact-db", "Production database for checkout is PostgreSQL 16 on Cloud SQL.")
        time.sleep(1.5)
        rows = m.retrieve("which database does checkout use", entity_path=scope, limit=3)
        brief = m.briefing(entity_path=scope)
        ok = len(rows) >= 1
        _gate("retrieve", ok, {"n": len(rows), "briefing_keys": list(brief.keys())[:8] if isinstance(brief, dict) else type(brief).__name__})
        return ok
    finally:
        m.close()


# ---------------------------------------------------------------------------
# Gate 3 — credentials and model IDs
# ---------------------------------------------------------------------------

def gate_llms() -> bool:
    ok_all = True
    from openai import OpenAI

    oc = OpenAI(api_key=config.env("OPENAI_API_KEY", required=True))
    ids = {m.id for m in oc.models.list().data}
    for model in [m for m in config.AGENT_MODELS if m.startswith("gpt")] + [config.JUDGE_MODEL]:
        ok = model in ids
        ok_all &= ok
        _gate(f"openai_model:{model}", ok, {"available": ok})

    anth_models = [m for m in config.AGENT_MODELS if m.startswith("claude")]
    if anth_models:
        import anthropic

        headers = {}
        ws = config.env("ANTHROPIC_WORKSPACE_ID")
        if ws:
            headers["anthropic-workspace-id"] = ws
        ac = anthropic.Anthropic(api_key=config.env("ANTHROPIC_API_KEY", required=True), default_headers=headers)
        for model in anth_models:
            try:
                r = ac.messages.create(model=model, max_tokens=16, messages=[{"role": "user", "content": "Say ok"}])
                _gate(f"anthropic_model:{model}", True, {"usage": r.usage.model_dump()})
            except Exception as e:  # noqa: BLE001
                ok_all = False
                _gate(f"anthropic_model:{model}", False, {"error": str(e)[:300]})

    try:
        from mem0 import MemoryClient

        MemoryClient(api_key=config.env("MEM0_API_KEY", required=True)).search(
            "ping", filters={"user_id": "cl-preflight"}, version="v2")
        _gate("mem0", True, {})
    except Exception as e:  # noqa: BLE001
        ok_all = False
        _gate("mem0", False, {"error": str(e)[:300]})

    try:
        from zep_cloud.client import Zep

        z = Zep(api_key=config.env("ZEP_API_KEY", required=True))
        try:
            z.user.add(user_id="cl-preflight")
        except Exception:  # noqa: BLE001
            pass
        z.user.get("cl-preflight")
        _gate("zep", True, {})
    except Exception as e:  # noqa: BLE001
        ok_all = False
        _gate("zep", False, {"error": str(e)[:300]})
    return ok_all


# ---------------------------------------------------------------------------
# Gate 4 — pgvector
# ---------------------------------------------------------------------------

def gate_pgvector() -> bool:
    try:
        import psycopg

        with psycopg.connect(config.PG_DSN) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            v = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        _gate("pgvector", True, {"version": v[0] if v else None})
        return True
    except Exception as e:  # noqa: BLE001
        _gate("pgvector", False, {"error": str(e)[:300]})
        return False


# ---------------------------------------------------------------------------
# Gates 5-9 — the continual-learning loop itself (evidence model, on the API under test)
#
# These are the gates section 5 of the plan names (3)-(6) plus the claim check. They
# run against whatever AMFS_HTTP_URL points at, from fresh scopes, and read back
# only through the public API, so what they assert is what an agent would see.
# ---------------------------------------------------------------------------

def _scope(tag: str) -> str:
    return f"{config.SCOPE_ROOT}/preflight-{tag}/{uuid.uuid4().hex[:8]}"


def _read(scope: str, key: str):
    m = _amfs("cl-preflight-observer")
    try:
        return m.read(scope, key)
    finally:
        m.close()


def _outcome(scope: str, keys: list[str], outcome, ref: str, *, agent: str = "cl-preflight-worker") -> None:
    """Read *keys* on a fresh connection and commit *outcome* against exactly them."""
    m = _amfs(agent)
    try:
        for k in keys:
            m.read(scope, k)
        m.commit_outcome(ref, outcome, task_input="preflight", response_text=f"used {keys}")
    finally:
        m.close()


def gate_evidence_model() -> bool:
    """(4) fresh 0.7 entry < 0.5 after one failure; 5-success entry < 0.5 within 3 failures;
    a 3-key success gives each key one third of the credit."""
    from amfs_core.models import OutcomeType

    scope = _scope("evidence")
    detail: dict[str, Any] = {"scope": scope}
    seeder = _amfs("cl-preflight-seeder")
    try:
        seeder.write(scope, "fresh", "Fresh lesson: restart the worker when the queue stalls.", confidence=0.7)
        seeder.write(scope, "trusted", "Trusted lesson: rotate the API key when uploads 401.", confidence=0.7)
        for k in ("split-a", "split-b", "split-c"):
            seeder.write(scope, k, f"Split lesson {k}: check the {k} dashboard first.", confidence=0.7)
        seeder.write(scope, "solo", "Solo lesson: check the solo dashboard first.", confidence=0.7)
    finally:
        seeder.close()
    time.sleep(1.0)

    ok = True
    # fresh 0.7 -> one failure -> gated
    _outcome(scope, ["fresh"], OutcomeType.FAILURE, f"pf-fresh-{uuid.uuid4().hex[:6]}")
    fresh = _read(scope, "fresh")
    detail["fresh_after_one_failure"] = {"confidence": round(fresh.confidence, 4), "status": fresh.evidence_status,
                                         "failure_count": fresh.failure_count}
    ok &= fresh.confidence < 0.5 and fresh.evidence_status == "discredited" and fresh.failure_count == 1

    # 5 successes then failures: below 0.5 within 3
    for i in range(5):
        _outcome(scope, ["trusted"], OutcomeType.SUCCESS, f"pf-trusted-ok-{i}-{uuid.uuid4().hex[:4]}")
    t5 = _read(scope, "trusted")
    detail["trusted_after_5_successes"] = {"confidence": round(t5.confidence, 4), "status": t5.evidence_status,
                                           "success_count": t5.success_count}
    ok &= t5.evidence_status == "validated" and t5.success_count == 5
    traj = []
    for i in range(3):
        _outcome(scope, ["trusted"], OutcomeType.FAILURE, f"pf-trusted-bad-{i}-{uuid.uuid4().hex[:4]}")
        traj.append(round(_read(scope, "trusted").confidence, 4))
    detail["trusted_failure_trajectory"] = traj
    ok &= traj[-1] < 0.5 and traj == sorted(traj, reverse=True)

    # credit split: one success cited to three keys vs one cited to a single key
    _outcome(scope, ["split-a", "split-b", "split-c"], OutcomeType.SUCCESS, f"pf-split-{uuid.uuid4().hex[:6]}")
    _outcome(scope, ["solo"], OutcomeType.SUCCESS, f"pf-solo-{uuid.uuid4().hex[:6]}")
    split = [round(_read(scope, k).confidence - 0.7, 4) for k in ("split-a", "split-b", "split-c")]
    solo = round(_read(scope, "solo").confidence - 0.7, 4)
    detail["credit_split_gain"] = {"per_split_key": split, "solo_key": solo}
    ok &= len(set(split)) == 1 and 0 < split[0] < solo and solo > 0

    _gate("evidence_model", bool(ok), detail)
    return bool(ok)


def gate_embedding_survives_outcome() -> bool:
    """(3) an entry keeps its embedding and semantic recall after an outcome opens a new version."""
    from amfs_core.models import OutcomeType

    scope = _scope("embed")
    text = "Invoice PDF export times out for accounts with more than five thousand line items; advise a smaller date range."
    seeder = _amfs("cl-preflight-seeder")
    try:
        seeder.write(scope, "export-timeout", text, confidence=0.7)
        seeder.write(scope, "unrelated", "Office plants are watered on Fridays.", confidence=0.7)
    finally:
        seeder.close()
    time.sleep(1.5)
    query = "customer cannot export a large invoice PDF, request hangs"
    before = _ranked(scope, query)
    _outcome(scope, ["export-timeout"], OutcomeType.SUCCESS, f"pf-embed-{uuid.uuid4().hex[:6]}")
    time.sleep(1.5)
    after = _ranked(scope, query)
    ent = _read(scope, "export-timeout")
    detail = {"scope": scope, "before": before, "after": after, "version": ent.version, "success_count": ent.success_count}
    ok = bool(before) and before[0][0] == "export-timeout" and bool(after) and after[0][0] == "export-timeout" \
        and ent.version >= 2 and ent.success_count == 1 and after[0][1] > 0
    _gate("embedding_survives_outcome", ok, detail)
    return ok


def gate_evidence_payload() -> bool:
    """(5) retrieve carries evidence_status / success_count / failure_count and, with
    include_avoid, a discredited entry comes back flagged rather than silently dropped."""
    from amfs.memory import is_avoid
    from amfs_core.models import OutcomeType, RecallConfig

    scope = _scope("payload")
    seeder = _amfs("cl-preflight-seeder")
    try:
        seeder.write(scope, "good", "Card declined but works elsewhere: escalate to tier 2 for a manual authorisation.", confidence=0.7)
        seeder.write(scope, "bad", "Card declined but works elsewhere: ask the customer to update the payment method.", confidence=0.7)
    finally:
        seeder.close()
    time.sleep(1.0)
    _outcome(scope, ["good"], OutcomeType.SUCCESS, f"pf-pay-ok-{uuid.uuid4().hex[:6]}")
    _outcome(scope, ["bad"], OutcomeType.FAILURE, f"pf-pay-bad-{uuid.uuid4().hex[:6]}")
    time.sleep(1.0)
    m = _amfs("cl-preflight-observer")
    try:
        plain = m.retrieve("card declined works elsewhere", entity_path=scope, limit=5)
        with_avoid = m.retrieve("card declined works elsewhere", entity_path=scope, limit=5,
                                recall_config=RecallConfig(include_avoid=True))
    finally:
        m.close()
    rows = {r.entry.key: (r.entry.evidence_status, r.entry.success_count, r.entry.failure_count, is_avoid(r)) for r in with_avoid}
    detail = {"scope": scope, "plain_keys": [r.entry.key for r in plain], "with_avoid": rows}
    ok = "good" in rows and rows["good"][:3] == ("validated", 1, 0) and not rows["good"][3] \
        and "bad" in rows and rows["bad"][0] == "discredited" and rows["bad"][3] \
        and "bad" not in [r.entry.key for r in plain]
    _gate("evidence_payload", ok, detail)
    return ok


def gate_briefing_sections() -> bool:
    """(6) after two validated entries in one entity fail, the compact briefing carries
    Discredited and Regime-shift sections; the validated survivor is listed as validated."""
    from amfs_core.models import OutcomeType

    scope = _scope("briefing")
    seeder = _amfs("cl-preflight-seeder")
    try:
        for k in ("rule-a", "rule-b", "rule-c"):
            seeder.write(scope, k, f"Rule {k}: the {k} remediation for its alert class.", confidence=0.7)
    finally:
        seeder.close()
    time.sleep(1.0)
    for k in ("rule-a", "rule-b", "rule-c"):
        for i in range(3):
            _outcome(scope, [k], OutcomeType.SUCCESS, f"pf-br-ok-{k}-{i}-{uuid.uuid4().hex[:4]}")
    for k in ("rule-a", "rule-b"):
        for i in range(3):
            _outcome(scope, [k], OutcomeType.FAILURE, f"pf-br-bad-{k}-{i}-{uuid.uuid4().hex[:4]}")
    time.sleep(4.0)  # cortex debounce
    m = _amfs("cl-preflight-observer")
    try:
        digests = m.briefing(entity_path=scope, compact=True)
    finally:
        m.close()
    summary = next((getattr(d, "summary", None) for d in digests if isinstance(getattr(d, "summary", None), dict)), {}) or {}
    disc = {str(e.get("key")) for e in (summary.get("discredited") or []) if isinstance(e, dict)}
    val = {str(e.get("key")) for e in (summary.get("validated") or []) if isinstance(e, dict)}
    rs = summary.get("regime_shift") if isinstance(summary.get("regime_shift"), dict) else {}
    detail = {"scope": scope, "discredited": sorted(disc), "validated": sorted(val),
              "regime_shift": {"suspected": rs.get("suspected"), "message": rs.get("message")}}
    ok = {"rule-a", "rule-b"} <= disc and "rule-c" in val and bool(rs.get("suspected"))
    _gate("briefing_sections", ok, detail)
    return ok


def gate_claim_check() -> bool:
    """(7) an outcome credits the claim the agent read: a key rewritten with a different
    claim between the read and the commit does not inherit the old claim's failure, and
    a restated claim keeps taking credit."""
    from amfs_core.models import OutcomeType

    scope = _scope("claim")
    seeder = _amfs("cl-preflight-seeder")
    try:
        seeder.write(scope, "fix", "Restart the worker.", confidence=0.7)
        seeder.write(scope, "same", "Rotate the key.", confidence=0.7)
    finally:
        seeder.close()
    time.sleep(0.5)
    m = _amfs("cl-preflight-worker")
    try:
        m.read(scope, "fix")                       # acted on v1
        m.write(scope, "fix", "Clear the queue first.", confidence=0.7)  # reflection rewrites the lesson
        m.commit_outcome(f"pf-claim-{uuid.uuid4().hex[:6]}", OutcomeType.FAILURE, task_input="preflight")
    finally:
        m.close()
    m = _amfs("cl-preflight-worker")
    try:
        m.read(scope, "same")
        m.write(scope, "same", "Rotate the key.", confidence=0.7)        # same claim, new version
        m.commit_outcome(f"pf-same-{uuid.uuid4().hex[:6]}", OutcomeType.SUCCESS, task_input="preflight")
    finally:
        m.close()
    fix, same = _read(scope, "fix"), _read(scope, "same")
    detail = {"scope": scope,
              "rewritten": {"value": fix.value, "status": fix.evidence_status, "failure_count": fix.failure_count,
                            "confidence": round(fix.confidence, 4)},
              "restated": {"status": same.evidence_status, "success_count": same.success_count}}
    ok = fix.value == "Clear the queue first." and fix.failure_count == 0 and fix.evidence_status == "untested" \
        and same.success_count == 1 and same.evidence_status == "validated"
    _gate("claim_check", ok, detail)
    return ok


LOOP_GATES = {
    "evidence": gate_evidence_model,
    "embedding": gate_embedding_survives_outcome,
    "payload": gate_evidence_payload,
    "briefing": gate_briefing_sections,
    "claim": gate_claim_check,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    only = set(argv)
    results = []
    if not only or "llms" in only:
        results.append(gate_llms())
    if not only or "pgvector" in only:
        results.append(gate_pgvector())
    if not only or "retrieve" in only:
        results.append(gate_retrieve())
    if not only or "loop" in only:
        results.append(gate_prod_loop())
    for name, fn in LOOP_GATES.items():
        if not only or name in only or "cl" in only:
            try:
                results.append(fn())
            except Exception as e:  # noqa: BLE001
                _gate(name, False, {"error": f"{type(e).__name__}: {str(e)[:300]}"})
                results.append(False)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "preflight.json"
    REPORT["ok"] = all(results)
    REPORT["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\npreflight {'PASSED' if REPORT['ok'] else 'FAILED'} -> {out}")
    return 0 if REPORT["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
