"""Train the students: SenseLab Managed Models scoped to the harness's own decision traces.

    python -m benchmarks.continual_learning.student.train plan   --matrix student/matrix.json
    python -m benchmarks.continual_learning.student.train run    --matrix student/matrix.json [--only a,b] [--dry-run]
    python -m benchmarks.continual_learning.student.train status
    python -m benchmarks.continual_learning.student.train dataset --alias <alias>

A matrix row names a teacher (scenario, arm, model, seeds, the result runs the cells live in)
and a student (family, serving options, gate). The scope is derived from the episode logs of
those runs: each selected cell's memory scope (``clbench/<cell>-<suffix>``, unique per cell,
carried by every trace the cell sealed) becomes one item of the model's ``entity_prefix``
list, so the student trains on exactly the traces those cells sealed and nothing else on the
account. Agent ids cannot do this — the harness derives their suffix once per process, so
cells in one run share them. ``plan`` shows the cells, the scope and the expected trace count.

Training is serial per account (the product allows one active run), so ``run`` walks the
matrix in order: create or reuse the model, compare product readiness with the local count,
set the gate, start the run, poll until the version is scored, record the report in the
registry that ``agent/clients.py`` resolves aliases from. A model that trains but fails the
default gate is still a finding — the registry keeps the product's verdict beside the
relaxed one the study promotes under.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from .. import config

API = config.AMFS_HTTP_URL.rstrip("/") + "/api/v1/models"
HERE = Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.json"
DATASETS_DIR = config.RESULTS_DIR / "students" / "datasets"
DEFAULT_SERVING = {"context_k": 3, "avoid_rows": True, "priors": True, "retrieval": "server",
                   "adaptive_k": False, "fallback": False}
RELAXED_GATE = {"tool_agreement": 0.50, "exact_agreement": 0.0, "decision_agreement": 0.0, "parse_rate": 0.80,
                "max_fallback_rate": 0.50, "max_failure_reproduction": 0.50, "max_instability": 0.50}
DEFAULT_GATE = {"tool_agreement": 0.75, "exact_agreement": 0.45, "parse_rate": 0.90,
                "max_fallback_rate": 0.20, "max_failure_reproduction": 0.30, "max_instability": 0.25}
POLL_S = 60


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0),
                        headers={"Authorization": f"Bearer {config.env('AMFS_API_KEY', required=True)}"})


def _registry() -> dict[str, Any]:
    try:
        return json.loads(config.STUDENT_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_registry(reg: dict[str, Any], *, only: str | None = None) -> None:
    """Write the registry; with *only*, merge that one alias into a fresh read of the file.

    Two trainers ran at once overnight (the v1 matrix, the v2 matrix, a requeue loop), each
    holding the whole registry it had loaded at start, and the last one to save wrote its
    stale copy over the others' rows — two live students came back ``is_live: false`` with no
    metrics and dropped out of the grid. Every per-alias update now goes through *only*.
    """
    config.STUDENT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if only is not None:
        fresh = _registry()
        fresh[only] = reg[only]
        reg = fresh
    tmp = config.STUDENT_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1, sort_keys=True, default=str))
    tmp.replace(config.STUDENT_REGISTRY)


# ----------------------------------------------------------------------------- scope from the logs
def _cells(runs: list[str]) -> list[dict[str, Any]]:
    """Finished cells of the named runs, with their memory scope (``clbench/<cell>-<suffix>``).
    The scope is what a cell's SenseLab arm read and wrote under, and what every trace it
    sealed carries in its causal entries; it is unique per cell where agent ids are not (the
    harness derives the agent suffix once per process, so cells in one run share it)."""
    out: list[dict[str, Any]] = []
    for run in runs:
        path = config.RESULTS_DIR / run / "cells.json"
        if not path.exists():
            print(f"  ! {path} missing", file=sys.stderr)
            continue
        for c in json.loads(path.read_text()):
            if c.get("scope"):
                out.append({**c, "_run": run})
    return out


def _cell_parts(cell: str) -> dict[str, str]:
    sc, arm, model, seed, params = (cell.split("|") + [""] * 5)[:5]
    return {"scenario": sc, "arm": arm, "model": model, "seed": seed.lstrip("s"), "params": params}


def _sealed_by_scope(runs: list[str]) -> dict[str, dict[str, int]]:
    """Per scope: episodes and episodes whose session committed a verifiable trace."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "sealed": 0, "successes": 0})
    scope_of: dict[tuple[str, str, str, int, str, bool], str] = {}
    for c in _cells(runs):
        parts = _cell_parts(c["cell"])
        m = re.search(r"-a(\d+)", parts["params"])
        fleet = bool(m and int(m.group(1)) > 1)
        scope_of[(parts["scenario"], parts["arm"], parts["model"], int(parts["seed"] or 0), c["_run"], fleet)] = c["scope"]
    for run in runs:
        path = config.RESULTS_DIR / run / "episodes.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # a run may hold both fleet sizes of one (scenario, arm, model, seed); fleet members
                # carry a second numeric suffix (``support-agent-788979-3``), singles do not
                fleet = bool(re.search(r"-\d+-\d+$", str(r.get("agent_id", ""))))
                key = (r.get("scenario"), r.get("arm"), r.get("model"), int(r.get("seed", 0)), run, fleet)
                scope = scope_of.get(key)
                if scope is None:
                    continue
                counts[scope]["episodes"] += 1
                counts[scope]["sealed"] += bool((r.get("memory") or {}).get("trace_verifiable"))
                counts[scope]["successes"] += bool(r.get("success"))
    return counts


def derive_scope(spec: dict[str, Any]) -> dict[str, Any]:
    seeds = {int(x) for x in (spec.get("seeds") or [])}
    params_re = re.compile(spec["cell_params"]) if spec.get("cell_params") else None
    # ``model`` is one model string or a list: a self-retrained student learns from the
    # cells of its parent under several serving variants (``x``, ``x+notes=...``).
    models = {spec["model"]} if isinstance(spec["model"], str) else set(spec["model"])
    sel = []
    for c in _cells(spec["runs"]):
        parts = _cell_parts(c["cell"])
        if parts["scenario"] != spec["scenario"] or parts["arm"] != spec["arm"] or parts["model"] not in models:
            continue
        if seeds and int(parts["seed"] or -1) not in seeds:
            continue
        if params_re and not params_re.search(parts["params"]):
            continue
        sel.append(c)
    scopes = sorted({c["scope"] for c in sel})
    per_scope = _sealed_by_scope(spec["runs"])
    episodes = sum(c.get("episodes", 0) for c in sel)
    sealed = sum(per_scope.get(sc, {}).get("sealed", 0) for sc in scopes)
    successes = sum(c.get("success", 0) for c in sel)
    return {"entity_prefix": "|".join(scopes), "scopes": scopes, "cells": len(sel), "episodes": episodes,
            "sealed_traces": sealed, "successes": successes,
            "cell_names": sorted(c["cell"] for c in sel)}


# ----------------------------------------------------------------------------- API
def _request(c: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """One call, retried on transport faults and 429/503/504 with backoff. Reads are
    idempotent; the writes this driver makes (create model, PATCH options, POST train)
    are too, or answer 409 on a repeat, so a retry cannot double anything."""
    delay = 5.0
    for attempt in range(8):
        try:
            r = c.request(method, API + path, **kwargs)
        except httpx.TransportError as e:
            if attempt == 7:
                raise
            print(f"   {method} {path}: {type(e).__name__}, retrying in {delay:.0f}s")
        else:
            if r.status_code not in (429, 503, 504) or attempt == 7:
                return r
            print(f"   {method} {path}: {r.status_code}, retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"{method} {path}: retries exhausted")


def _get(c: httpx.Client, path: str, **params: Any) -> dict[str, Any]:
    r = _request(c, "GET", path, params={k: v for k, v in params.items() if v is not None})
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _send(c: httpx.Client, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    r = _request(c, method, path, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
    return r.json() if r.content else {}


def _find_model(c: httpx.Client, display_name: str) -> dict[str, Any] | None:
    for m in _get(c, "").get("models") or []:
        if m.get("display_name") == display_name and not m.get("deleted_at"):
            return m
    return None


def ensure_model(c: httpx.Client, spec: dict[str, Any], scope: dict[str, Any], *, dry_run: bool) -> dict[str, Any] | None:
    name = f"cl:{spec['alias']}"[:80]
    serving = {**DEFAULT_SERVING, **(spec.get("serving") or {})}
    existing = _find_model(c, name)
    if existing:
        if existing.get("entity_prefix") != scope["entity_prefix"]:
            print(f"  ! existing model {existing['model_id']} has a different scope; patching")
            if not dry_run:
                _send(c, "PATCH", f"/{existing['id']}", {"entity_prefix": scope["entity_prefix"]})
        if not dry_run:
            _send(c, "PATCH", f"/{existing['id']}", {"serving_options": serving})
        return existing if dry_run else _get(c, f"/{existing['id']}")
    body = {"display_name": name, "entity_prefix": scope["entity_prefix"], "family": spec.get("family", "senselab-s1"),
            "serving_options": serving}
    if spec.get("agent_group"):
        body["agent_group"] = spec["agent_group"]
    if dry_run:
        print("  would create:", json.dumps(body)[:300])
        return None
    model = _send(c, "POST", "", body)
    print(f"  created {model['model_id']} ({model['id']})")
    return model


def set_gate(c: httpx.Client, model: dict[str, Any], gate: str) -> None:
    thresholds = RELAXED_GATE if gate == "relaxed" else None
    _send(c, "PUT", f"/{model['id']}/eval-config", {"thresholds": thresholds})


def wait_for_version(c: httpx.Client, model_id: str, n_before: int, *, timeout_s: float = 6 * 3600) -> dict[str, Any]:
    t0 = time.time()
    last_state = None
    while time.time() - t0 < timeout_s:
        try:
            m = _get(c, f"/{model_id}")
        except (httpx.HTTPError, RuntimeError) as e:
            # A poll that fails (a redeploy, a slow read) says nothing about the run, which is
            # server-side; the driver dying here is what would lose the queue's order.
            print(f"  [{time.strftime('%H:%M:%S')}] poll failed, retrying: {str(e)[:120]}")
            time.sleep(POLL_S)
            continue
        state = m.get("state")
        versions = m.get("versions") or []
        scored = [v for v in versions if v.get("metrics") or v.get("eval_passed") is not None]
        if state != last_state:
            print(f"  [{time.strftime('%H:%M:%S')}] state={state} versions={len(versions)} scored={len(scored)}")
            last_state = state
        if state == "failed":
            return m
        if len(scored) > n_before and state not in ("queued", "training", "evaluating"):
            return m
        time.sleep(POLL_S)
    raise TimeoutError(f"model {model_id} still {last_state} after {timeout_s}s")


def default_gate_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    """The product's default bars applied to a report, so a study model promoted under the
    relaxed gate still carries the verdict a customer would have seen."""
    fails = []
    for k, floor in (("tool_agreement", 0.75), ("exact_agreement", 0.45), ("parse_rate", 0.90)):
        v = metrics.get(k)
        if isinstance(v, (int, float)) and v < floor:
            fails.append(f"{k} {v:.3f} < {floor}")
    for k, ceil in (("fallback_rate", 0.20), ("failure_reproduction", 0.30), ("instability", 0.25)):
        v = metrics.get(k)
        if isinstance(v, (int, float)) and v > ceil:
            fails.append(f"{k} {v:.3f} > {ceil}")
    return {"passed": not fails, "failures": fails}


# ----------------------------------------------------------------------------- commands
def cmd_plan(args: argparse.Namespace) -> None:
    matrix = json.loads(Path(args.matrix).read_text())
    only = set(args.only.split(",")) if args.only else None
    for spec in matrix:
        if only and spec["alias"] not in only:
            continue
        scope = derive_scope(spec)
        print(f"\n== {spec['alias']}  ({spec['scenario']}/{spec['arm']}/{spec['model']} seeds={spec.get('seeds')} "
              f"runs={spec['runs']})")
        print(f"   cells={scope['cells']} episodes={scope['episodes']} sealed_traces={scope['sealed_traces']} "
              f"successes={scope['successes']} ({scope['successes'] / max(1, scope['episodes']):.0%})")
        print(f"   entity_prefix ({len(scope['entity_prefix'])} chars), {len(scope['scopes'])} scopes:")
        for name in scope["cell_names"]:
            print(f"      {name}")
        if scope["episodes"] and scope["sealed_traces"] < 0.9 * scope["episodes"]:
            print(f"   ! only {scope['sealed_traces']}/{scope['episodes']} episodes report a verifiable trace")


def cmd_run(args: argparse.Namespace) -> None:
    matrix = json.loads(Path(args.matrix).read_text())
    only = set(args.only.split(",")) if args.only else None
    reg = _registry()
    with _client() as c:
        for spec in matrix:
            alias = spec["alias"]
            if only and alias not in only:
                continue
            entry = reg.get(alias) or {}
            if entry.get("public_model_id") and entry.get("eval") and not args.retrain:
                print(f"== {alias}: already trained ({entry['public_model_id']}); --retrain to redo")
                continue
            scope = derive_scope(spec)
            print(f"\n== {alias}: {scope['cells']} cells, {scope['sealed_traces']} sealed traces, "
                  f"{len(scope['scopes'])} scopes")
            if not scope["scopes"]:
                print("   refusing: no cells matched")
                continue
            if scope["sealed_traces"] < 100:
                print("   refusing: fewer than 100 sealed traces (product floor)")
                continue
            model = ensure_model(c, spec, scope, dry_run=args.dry_run)
            if args.dry_run or model is None:
                try:
                    ready = _get(c, "/readiness", entity_prefix=scope["entity_prefix"])
                    print(f"   readiness: eligible={ready.get('eligible')} learnable={ready.get('learnable')} "
                          f"positive={ready.get('positive_share')} shapes={ready.get('task_shapes')} ready={ready.get('ready')}")
                except Exception as e:  # noqa: BLE001
                    print(f"   readiness unavailable: {e}")
                continue
            ready = _get(c, "/readiness", entity_prefix=scope["entity_prefix"])
            print(f"   readiness: eligible={ready.get('eligible')} (local sealed {scope['sealed_traces']}) "
                  f"learnable={ready.get('learnable')} positive={ready.get('positive_share')} "
                  f"shapes={ready.get('task_shapes')} ready={ready.get('ready')} gaps={ready.get('gaps')}")
            if not ready.get("ready") and not args.force:
                print("   refusing: product readiness says not ready; --force to proceed")
                continue
            gate = spec.get("gate", "relaxed")
            set_gate(c, model, gate)
            base_price = {"senselab-s1": "gemini-2.5-flash-lite", "senselab-s1-pro": "gemini-2.5-flash"}
            if spec.get("serve_base"):
                # The untuned base model behind the same rendered prompt: a control, not a student.
                # No training run; the row serves the tier's base while it has no live version.
                _send(c, "PATCH", f"/{model['id']}", {"serve_base_enabled": True})
                reg[alias] = {**entry, "public_model_id": model["model_id"], "model_uuid": model["id"],
                              "family": spec.get("family", "senselab-s1"),
                              "price": spec.get("price") or base_price.get(spec.get("family", "senselab-s1"), "gemini-2.5-flash-lite"),
                              "serve_base": True, "gate": gate, "state": "base",
                              "scenario": spec["scenario"],
                              "serving": {**DEFAULT_SERVING, **(spec.get("serving") or {})},
                              "teacher": None, "scope": {"entity_prefix": scope["entity_prefix"], "cells": scope["cells"]},
                              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
                _save_registry(reg, only=alias)
                print(f"   base-served control registered: {model['model_id']}")
                continue
            reg[alias] = {**entry, "public_model_id": model["model_id"], "model_uuid": model["id"],
                          "family": spec.get("family", "senselab-s1"), "price": spec.get("price") or spec.get("family", "senselab-s1"),
                          "serve_base": False, "gate": gate,
                          "serving": {**DEFAULT_SERVING, **(spec.get("serving") or {})},
                          "teacher": {k: spec.get(k) for k in ("scenario", "arm", "model", "seeds", "runs")},
                          # A self-retrained student names the student whose traces it learned from; the
                          # grid keeps it out of the phases that run on its training seeds.
                          "parent": spec.get("parent"), "note": spec.get("note"),
                          "scope": {k: scope[k] for k in ("entity_prefix", "cells", "episodes", "sealed_traces", "successes", "cell_names")},
                          "readiness": {k: ready.get(k) for k in ("eligible", "learnable", "positive_share", "task_shapes", "ready")},
                          "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            _save_registry(reg, only=alias)
            # dataset snapshot before training: what a run started now would use
            try:
                dump_dataset(c, model["id"], alias)
            except Exception as e:  # noqa: BLE001
                print(f"   dataset dump failed: {e}")
            n_before = len([v for v in (model.get("versions") or []) if v.get("metrics")])
            try:
                est = _get(c, f"/{model['id']}/training-estimate")
                print(f"   estimate: {json.dumps(est)[:200]}")
            except Exception:  # noqa: BLE001
                pass
            for attempt in range(120):
                try:
                    job = _send(c, "POST", f"/{model['id']}/train")
                    print(f"   queued job {job.get('job_id')} position={job.get('queue_position')} "
                          f"cost={json.dumps(job.get('training_cost'))[:160]}")
                    break
                except RuntimeError as e:
                    msg = e.args[0] if e.args else str(e)
                    if "409" in msg:
                        # This model already has a run queued/training (a previous invocation
                        # of this script queued it): nothing to submit, go and wait for it.
                        print(f"   run already queued for this model; waiting for it ({msg[:80]})")
                        break
                    if "429" in msg:
                        # Another model's run holds the account's single slot.
                        print(f"   account slot busy; retrying in 5 min ({msg[:80]})")
                        time.sleep(300)
                        continue
                    raise
            m = wait_for_version(c, model["id"], n_before)
            versions = m.get("versions") or []
            latest = max(versions, key=lambda v: v.get("version") or 0) if versions else {}
            metrics = latest.get("metrics") or {}
            reg[alias].update({"state": m.get("state"), "version": latest.get("version"), "version_id": latest.get("id"),
                               "is_live": latest.get("is_live"), "eval_passed": latest.get("eval_passed"),
                               "eval": metrics, "default_gate": default_gate_verdict(metrics),
                               "train_examples": latest.get("train_examples"), "holdout_examples": latest.get("holdout_examples"),
                               "trained": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _save_registry(reg, only=alias)
            print(f"   done: state={m.get('state')} v{latest.get('version')} live={latest.get('is_live')} "
                  f"eval_passed={latest.get('eval_passed')} default_gate={reg[alias]['default_gate']}")
            print(f"   metrics: {json.dumps({k: metrics.get(k) for k in ('tool_agreement', 'decision_agreement', 'exact_agreement', 'parse_rate', 'fallback_rate', 'failure_reproduction', 'instability', 'holdout_examples')})}")
            if m.get("state") == "failed":
                print(f"   FAILED: {json.dumps(m.get('last_error') or m.get('error') or m)[:400]}")
            elif not latest.get("is_live") and latest.get("eval_passed") and latest.get("id"):
                try:
                    _send(c, "POST", f"/{model['id']}/versions/{latest['id']}/promote")
                    reg[alias]["is_live"] = True
                    _save_registry(reg, only=alias)
                    print("   promoted")
                except RuntimeError as e:
                    print(f"   promote failed: {e}")


def dump_dataset(c: httpx.Client, model_uuid: str, alias: str) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    stats = _get(c, f"/{model_uuid}/dataset")
    (DATASETS_DIR / f"{alias}.stats.json").write_text(json.dumps(stats, indent=1))
    for split in ("train", "validation", "holdout"):
        r = c.get(API + f"/{model_uuid}/dataset", params={"split": split, "limit": 20000})
        if r.status_code >= 400:
            raise RuntimeError(f"dataset {split}: {r.status_code} {r.text[:200]}")
        (DATASETS_DIR / f"{alias}.{split}.jsonl").write_text(r.text)
    s = stats.get("stats") or {}
    print(f"   dataset: hash={s.get('dataset_hash', '?')[:12]} train={s.get('train')} val={s.get('validation')} "
          f"holdout={s.get('holdout')} dropped={s.get('dropped')} -> {DATASETS_DIR}/{alias}.*")


def cmd_dataset(args: argparse.Namespace) -> None:
    reg = _registry()
    entry = reg.get(args.alias)
    if not entry:
        sys.exit(f"{args.alias} not in registry")
    with _client() as c:
        dump_dataset(c, entry["model_uuid"], args.alias)


def cmd_refresh(args: argparse.Namespace) -> None:
    """Re-read every trained row's state, live version and metrics from the API.

    Repairs rows a concurrent trainer overwrote (see _save_registry) and picks up a
    version promoted or retrained outside this script."""
    reg = _registry()
    with _client() as c:
        for alias, e in sorted(reg.items()):
            if e.get("serve_base") or (args.only and alias != args.only):
                continue
            m = _get(c, f"/{e['model_uuid']}")
            versions = m.get("versions") or []
            live = [v for v in versions if v.get("is_live")]
            latest = live[0] if live else (max(versions, key=lambda v: v.get("version") or 0) if versions else {})
            metrics = latest.get("metrics") or {}
            reg[alias].update({"state": m.get("state"), "version": latest.get("version"), "version_id": latest.get("id"),
                               "is_live": bool(live), "eval_passed": latest.get("eval_passed"),
                               "eval": metrics, "default_gate": default_gate_verdict(metrics),
                               "train_examples": latest.get("train_examples"),
                               "holdout_examples": latest.get("holdout_examples"),
                               "refreshed": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _save_registry(reg, only=alias)
            print(f"{alias:28s} state={m.get('state'):10s} v{latest.get('version')} live={bool(live)} "
                  f"dec={metrics.get('decision_agreement')} parse={metrics.get('parse_rate')}")


def cmd_status(args: argparse.Namespace) -> None:
    reg = _registry()
    if not reg:
        print("registry empty")
        return
    with _client() as c:
        for alias, e in sorted(reg.items()):
            line = f"{alias:28s} {e.get('public_model_id', '-'):30s} {e.get('family', ''):16s} "
            try:
                m = _get(c, f"/{e['model_uuid']}")
                live = [v for v in (m.get("versions") or []) if v.get("is_live")]
                line += f"state={m.get('state'):10s} live_v={(live[0].get('version') if live else '-')} "
            except Exception as ex:  # noqa: BLE001
                line += f"(lookup failed: {str(ex)[:60]}) "
            ev = e.get("eval") or {}
            line += (f"tool={ev.get('tool_agreement')} dec={ev.get('decision_agreement')} exact={ev.get('exact_agreement')} "
                     f"parse={ev.get('parse_rate')} default_gate={'pass' if (e.get('default_gate') or {}).get('passed') else 'FAIL'}")
            print(line)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--matrix", default=str(DEFAULT_MATRIX)); p.add_argument("--only")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run"); p.add_argument("--matrix", default=str(DEFAULT_MATRIX)); p.add_argument("--only")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--retrain", action="store_true"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("refresh"); p.add_argument("--only"); p.set_defaults(fn=cmd_refresh)
    p = sub.add_parser("dataset"); p.add_argument("--alias", required=True); p.set_defaults(fn=cmd_dataset)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
