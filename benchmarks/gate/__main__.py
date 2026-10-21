"""``python -m benchmarks.gate`` — run the gate benchmark against a loop, or
score a ledger a loop wrote itself.

    # the reference loops, in-process, in milliseconds
    python -m benchmarks.gate run --loop reference:replay --out results/reference-replay
    python -m benchmarks.gate run --loop reference:accept-all --out results/reference-accept-all

    # SenseLab, against a deployment (dry-run probes close after Tier 1, minutes each)
    AMFS_PRO_URL=https://api.dev.sense-lab.ai AMFS_EVAL_API_KEY=... \\
      python -m benchmarks.gate run --loop senselab --agent deploy-agent \\
        --probes 8 --poll 60 --timeout 7200 --out results/senselab-dev-deploy-agent

    # any loop, from its ledger (see ledger.py for the schema)
    python -m benchmarks.gate score results/my-loop
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import runner
from .adapters import resolve
from .protocol import ProbeKind


def _opts(pairs: Sequence[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"--opt expects key=value, got {p!r}")
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive a loop through the three self-tests")
    run.add_argument("--loop", required=True,
                     help="reference[:replay|:accept-all] or senselab")
    run.add_argument("--agent", help="agent id (senselab)")
    run.add_argument("--url", help="API base URL (senselab; AMFS_PRO_URL)")
    run.add_argument("--api-key", dest="api_key", help="API key (senselab; AMFS_EVAL_API_KEY)")
    run.add_argument("--mode", choices=["dry_run", "live"],
                     help="probe mode to request (senselab; the agent's setting by default)")
    run.add_argument("--probes", type=int, default=8, help="probes to inject (default 8)")
    run.add_argument("--kinds", default=None,
                     help="comma-separated subset of placebo,poison,reward_hack,revert")
    run.add_argument("--poll", type=float, default=30.0, help="seconds between polls")
    run.add_argument("--timeout", type=float, default=24 * 3600.0,
                     help="seconds before an open probe is marked expired (0 = never)")
    run.add_argument("--busy-wait", type=float, default=6 * 3600.0,
                     help="seconds to wait for a busy gate before giving up")
    run.add_argument("--aa-trials", type=int, default=40)
    run.add_argument("--aa-window-days", type=int, default=14)
    run.add_argument("--skip-aa", action="store_true")
    run.add_argument("--skip-reward", action="store_true")
    run.add_argument("--opt", action="append", metavar="KEY=VALUE",
                     help="loop option (reference: seed, false_accept, canary_catch, "
                          "remembers_verified, minutes_to_reject)")
    run.add_argument("--out", type=Path, default=None, help="results directory")
    run.add_argument("-q", "--quiet", action="store_true")

    score = sub.add_parser("score", help="score a results directory a loop wrote itself")
    score.add_argument("results_dir", type=Path)
    score.add_argument("--loop", default=None, help="loop name for the card")
    score.add_argument("--no-write", action="store_true", help="print only")
    score.add_argument("-q", "--quiet", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "score":
        card = runner.score(args.results_dir, loop_name=args.loop, write=not args.no_write)
        _print(card)
        return 0
    options = _opts(args.opt)
    for k in ("agent", "url", "api_key", "mode"):
        if getattr(args, k, None):
            options[k] = getattr(args, k)
    kinds = ([ProbeKind(k.strip()) for k in args.kinds.split(",") if k.strip()]
             if args.kinds else None)
    cfg = runner.RunConfig(
        probes=args.probes, kinds=kinds, poll_interval=args.poll, timeout=args.timeout,
        busy_wait=args.busy_wait, aa_trials=args.aa_trials, aa_window_days=args.aa_window_days,
        skip_aa=args.skip_aa, skip_reward=args.skip_reward,
        meta={"loop_spec": args.loop, **{f"opt_{k}": v for k, v in options.items()
                                         if k != "api_key"}},
    )
    loop = resolve(args.loop, options)
    try:
        result = runner.run(loop, cfg, out_dir=args.out)
    finally:
        loop.close()
    _print(result.card)
    if args.out:
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


def _print(card: dict) -> None:
    from .scorecard import render_markdown

    sys.stdout.write(render_markdown(card))


if __name__ == "__main__":
    raise SystemExit(main())
