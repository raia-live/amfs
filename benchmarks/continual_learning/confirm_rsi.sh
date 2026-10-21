#!/usr/bin/env bash
# The confirmatory runs of the RSI paper (HYPOTHESIS_RSI.md, amendments (w) and (x)): seeds
# 37/41/59/73 on the two set-C families, both agent models, on the `post-canary-ni` cut. Run once,
# after `analysis-freeze-rsi` is tagged. One runner process per (protocol, seed), each with its
# own replay-hub port (8790+), launched in its own session (setsid) so it outlives the shell that
# started it. Never re-run a finished run name: the runner skips finished cells and re-runs
# partial ones from scratch, which is the intended resume.
#
#   ./confirm_rsi.sh plan                 the cells and episode counts, nothing launched
#   ./confirm_rsi.sh launch f1 <seed>     fleet 1, C protocol (100 episodes, change at 30), every headline
#                                         system and the two ablations; run name rsi-conf-f1-s<seed>
#   ./confirm_rsi.sh launch e2b <seed>    fleet 10, E2b protocol (200 episodes, change at 50), SenseLab and
#                                         its two gate ablations; run name rsi-conf-e2b-s<seed>
#   ./confirm_rsi.sh status               episode counts and the last log line of every confirmatory run
#   ./confirm_rsi.sh tunnels              the ngrok agent's tunnels and whether a hub listens on each
#   ./confirm_rsi.sh prune                delete the tunnels whose hub is gone (launch does this too)
#   ./confirm_rsi.sh read                 the confirmatory readout into results/rsi-report-confirmatory
#
# Launch order: the four f1 processes first, then the four e2b ones. The ngrok account allows five
# tunnels on its agent session; launches past the fifth share the pooled URL (see prune_tunnels).
#
# Protocol constants are those of the dev runs `rsi-c5` / `rsi-c6` (fleet 1) and `rsi-e2b-c6`
# (fleet 10): CL_RSI_TIER2=1, CL_RSI_VERDICT_WAIT=240, CL_SENSELAB_TIMEOUT=90, canary timeout and
# window = the cell horizon (amendment (v) rule), --change-focus 0.5 --change-variant C --opaque-ids.
# Overrides: CL_CONF_MODELS (default "gpt-5.5 claude-opus-5"), CL_CONF_F1_ARMS (the eight
# confirmatory arms), CL_CONF_E2B_ARMS (senselab-rsi), CL_CONF_E2B_MODELS (gpt-5.5),
# CL_CONF_WORKERS_F1 (3), CL_CONF_WORKERS_E2B (3), CL_CONF_HUB_PORT (the hub port, when two
# f1 processes share a seed).
#
# Cross-model study (HYPOTHESIS_RSI.md amendment (ac), 2026-10-21): Beam alone and Beam +
# SenseLab against Claude Opus 5.5 alone and Opus 5.5 with its own history in context. Two
# launches per seed so no cell is wasted (Beam x raw-traces would spend a third of the Beam
# daily token allowance on a column the readout does not use):
#   CL_CONF_RUN_TAG=beam   CL_CONF_MODELS=Beam-1.0-501B-A23B CL_CONF_F1_ARMS="none senselab-rsi" \
#     CL_CONF_WORKERS_F1=2 CL_REFLECTION_RPM=14 ./confirm_rsi.sh launch f1 <seed>
#   CL_CONF_RUN_TAG=opus55 CL_CONF_MODELS=claude-opus-5-5   CL_CONF_F1_ARMS="none raw-traces" \
#     CL_CONF_WORKERS_F1=4 CL_CONF_HUB_PORT=$((8800 + i)) ./confirm_rsi.sh launch f1 <seed>
# Only the senselab-rsi arm opens a replay hub, so the opus55 processes never bind their port;
# the distinct number only keeps the launch pre-check honest. CL_REFLECTION_RPM is each Beam
# process's share of Reflection's 60 requests a minute (four processes: 14 each). Readout:
#   CL_CONF_MODELS="Beam-1.0-501B-A23B claude-opus-5-5" ./confirm_rsi.sh read-cross-model
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
PY="$BENCH/../.venv/bin/python"
MODELS="${CL_CONF_MODELS:-gpt-5.5 claude-opus-5}"
SEEDS_ALL="37 41 59 73"
F1_ARMS="${CL_CONF_F1_ARMS:-none raw-traces pgvector pgvector+rerank pgvector-diy+outcomes senselab senselab-rsi senselab-rsi-nocanary}"
E2B_ARMS="${CL_CONF_E2B_ARMS:-senselab-rsi}"   # amendment (y): fleet 10 runs SenseLab only
E2B_MODELS="${CL_CONF_E2B_MODELS:-gpt-5.5}"   # amendment (y): fleet 10 on gpt-5.5 only
# Amendment (aa), 2026-09-30: a tag in the run name, so a protocol relaunched on a later engine cut
# gets its own run (rsi-conf-e2b-<tag>-s<seed>) and the cells already run keep theirs. Empty = the
# original names.
RUN_TAG="${CL_CONF_RUN_TAG:+-$CL_CONF_RUN_TAG}"

port_for() {  # a distinct hub port per (protocol, seed): f1 8787 (the agent's own tunnel), 8791-8793; e2b 8794-8797
  local proto="$1" seed="$2" i=0
  if [[ -n "${CL_CONF_HUB_PORT:-}" ]]; then echo "$CL_CONF_HUB_PORT"; return; fi
  for s in $SEEDS_ALL; do [[ "$s" == "$seed" ]] && break; i=$((i + 1)); done
  if [[ "$proto" == "f1" ]]; then [[ $i -eq 0 ]] && echo 8787 || echo $((8790 + i)); else echo $((8794 + i)); fi
}

common_env() {  # the .env plus the protocol constants; nothing is printed
  set -a; source "$HERE/.env"; set +a
  export CL_RSI_TIER2=1 CL_RSI_VERDICT_WAIT=240 CL_SENSELAB_TIMEOUT=90 CL_REPLAY_SIBLINGS="8787-8797"
}

# The ngrok account runs at most 5 endpoints on one agent session (measured 2026-09-29), and the
# eight runner processes each carry a replay hub. Every tunnel of the account pools on one reserved
# domain, and a hub hands a webhook it cannot verify to the sibling whose secret signed it
# (replay_hub.py), so a hub does not need a tunnel of its own: it needs the pooled URL and one
# tunnelled hub alive. Rule: prune tunnels whose local port no longer listens; the first five
# launches get their own tunnel (the hub opens it); later launches are given the pooled URL
# (CL_REPLAY_PUBLIC_URL). Launch the four f1 processes first — they run longest — so the hubs that
# outlive the rest are the tunnelled ones.
NGROK_API="${NGROK_API:-http://127.0.0.1:4040}"
TUNNEL_CAP=5

tunnels_json() { curl -s -m 5 "$NGROK_API/api/tunnels" 2>/dev/null || echo '{"tunnels":[]}'; }

stale_tunnels() {  # names of tunnels whose local port has no listener
  TJ="$(tunnels_json)" python3 - <<'PY'
import json, os, socket
for t in json.loads(os.environ["TJ"]).get("tunnels", []):
    addr = str(t.get("config", {}).get("addr", ""))
    port = int(addr.rsplit(":", 1)[-1]) if addr.rsplit(":", 1)[-1].isdigit() else 0
    s = socket.socket(); s.settimeout(1)
    alive = port and s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    # the agent's own command-line tunnel is never deleted: it may be the agent session itself
    if not alive and t["name"] != "command_line":
        print(t["name"])
PY
}

prune_tunnels() {  # delete tunnels to ports with no listener (finished runs); prints what it removed
  local name
  while read -r name; do
    [[ -n "$name" ]] || continue
    curl -s -m 5 -o /dev/null -X DELETE "$NGROK_API/api/tunnels/$name" && echo "pruned stale tunnel $name"
  done < <(stale_tunnels)
}

tunnel_count() {
  TJ="$(tunnels_json)" python3 - <<'PY'
import json, os
print(len(json.loads(os.environ["TJ"]).get("tunnels", [])))
PY
}

pooled_url() {  # the reserved domain every tunnel shares (never printed by the caller)
  TJ="$(tunnels_json)" python3 - <<'PY'
import json, os
for t in json.loads(os.environ["TJ"]).get("tunnels", []):
    u = str(t.get("public_url", ""))
    if u.startswith("https://"):
        print(u.rstrip("/")); break
PY
}

runner_cmd() {  # proto seed -> the runner argv (without env)
  local proto="$1" seed="$2"
  if [[ "$proto" == "f1" ]]; then
    echo "-m continual_learning.runner --run rsi-conf-f1$RUN_TAG-s$seed --scenarios support ci-fix --arms $F1_ARMS" \
         "--models $MODELS --seeds $seed --fleet 1 --episodes 100 --change-at 30 --change-focus 0.5" \
         "--change-variant C --opaque-ids --workers ${CL_CONF_WORKERS_F1:-3}"
  else
    echo "-m continual_learning.runner --run rsi-conf-e2b$RUN_TAG-s$seed --scenarios support ci-fix --arms $E2B_ARMS" \
         "--models $E2B_MODELS --seeds $seed --fleet 10 --episodes 200 --change-at 50 --change-focus 0.5" \
         "--change-variant C --opaque-ids --workers ${CL_CONF_WORKERS_E2B:-3}"
  fi
}

cmd="${1:-}"
case "$cmd" in
  plan)
    cd "$BENCH"
    for proto in f1 e2b; do for seed in $SEEDS_ALL; do
      echo "== $proto seed $seed (hub port $(port_for "$proto" "$seed"))"
      # shellcheck disable=SC2046
      "$PY" $(runner_cmd "$proto" "$seed") --dry | head -1
    done; done
    ;;
  launch)
    proto="${2:?f1|e2b}"; seed="${3:?seed}"
    [[ " $SEEDS_ALL " == *" $seed "* ]] || { echo "seed must be one of: $SEEDS_ALL"; exit 1; }
    run="rsi-conf-$proto$RUN_TAG-s$seed"
    cd "$BENCH"; mkdir -p "continual_learning/results/$run"
    if pgrep -f -- "--run $run " > /dev/null; then echo "$run is already running"; exit 1; fi
    horizon=100; [[ "$proto" == "e2b" ]] && horizon=200
    port="$(port_for "$proto" "$seed")"
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)"; then
      echo "port :$port is still in use (an earlier run's hub?); not launching $run"; exit 1
    fi
    prune_tunnels
    shared=""
    if [[ "$(tunnel_count)" -ge "$TUNNEL_CAP" ]]; then
      shared="$(pooled_url)"
      [[ -n "$shared" ]] || { echo "tunnel cap reached and no pooled URL to share"; exit 1; }
      echo "$run: tunnel cap ($TUNNEL_CAP) reached; hub on :$port shares the pooled URL via sibling forwarding"
    fi
    (
      common_env
      export CL_REPLAY_PORT="$port" CL_RSI_CANARY_TIMEOUT="$horizon" CL_RSI_CANARY_WINDOW="$horizon"
      [[ -n "$shared" ]] && export CL_REPLAY_PUBLIC_URL="$shared"
      export PYTHONUNBUFFERED=1
      # shellcheck disable=SC2046
      perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- "$PY" $(runner_cmd "$proto" "$seed") \
        >> "continual_learning/results/$run/run.log" 2>&1 &
    )
    sleep 2
    echo "launched $run at $(date -u +%FT%TZ): $(pgrep -f -- "--run $run " | head -1)"
    ;;
  tunnels)
    # the ngrok agent's tunnels: local port and whether a hub listens there (URLs not printed)
    TJ="$(tunnels_json)" python3 - <<'PY'
import json, os, socket
ts = json.loads(os.environ["TJ"]).get("tunnels", [])
print(f"{len(ts)} tunnel(s) on the agent session (cap 5)")
for t in ts:
    addr = str(t.get("config", {}).get("addr", "")); port = int(addr.rsplit(":", 1)[-1])
    s = socket.socket(); s.settimeout(1); alive = s.connect_ex(("127.0.0.1", port)) == 0; s.close()
    state = "hub listening" if alive else "STALE (no listener; launch prunes it)"
    print(f"  {t['name']:<20} :{port}  {state}")
PY
    ;;
  prune)
    prune_tunnels
    ;;
  status)
    cd "$HERE/results"
    for d in rsi-conf-*/; do
      d="${d%/}"; n=0; [[ -f "$d/episodes.jsonl" ]] && n=$(wc -l < "$d/episodes.jsonl")
      alive="stopped"; pgrep -f -- "--run $d " > /dev/null && alive="running"
      printf "%-20s %5s episodes  %-8s  %s\n" "$d" "$n" "$alive" "$(tail -1 "$d/run.log" 2>/dev/null | cut -c1-110)"
    done
    ;;
  read)
    # The whole confirmatory read: tables, the fleet-10 canary readout (P12–P14; the trace join
    # needs PGPASS_DSN in the environment, otherwise P14 is left undefined), the blind coding sheet
    # for P11(a′) over every confirmatory run, then the paper fill. Set CL_CONF_FIGURES=1 to redraw
    # the figures with the confirmatory seeds as well.
    cd "$BENCH/.."
    runs=$(cd "$HERE/results" && ls -d rsi-conf-* 2>/dev/null | tr '\n' ' ')
    # shellcheck disable=SC2086
    "$PY" -m benchmarks.continual_learning.analysis.rsi --seeds $SEEDS_ALL --models $MODELS --runs $runs \
      --report-dir benchmarks/continual_learning/results/rsi-report-confirmatory --paper
    for r in $(cd "$HERE/results" && ls -d rsi-conf-e2b-* 2>/dev/null); do
      if [ -n "${PGPASS_DSN:-}" ]; then
        "$PY" -m benchmarks.continual_learning.analysis.rsi_canary_readout --run "$r"
      else
        "$PY" -m benchmarks.continual_learning.analysis.rsi_canary_readout --run "$r" --no-db
      fi
    done
    # shellcheck disable=SC2086
    "$PY" -m benchmarks.continual_learning.analysis.rsi_coding_sheet make --runs $runs \
      --out benchmarks/continual_learning/results/rsi-coding-confirmatory
    "$PY" -m benchmarks.continual_learning.analysis.fill_rsi_paper
    ;;
  read-cross-model)
    # Amendment (ac): the cross-model readout, keyed on (family, seed), each system from its own
    # model, into results/rsi-report-beam. Runs: every rsi-conf-f1-beam-* and rsi-conf-f1-opus55-*.
    cd "$BENCH/.."
    runs=$(cd "$HERE/results" && ls -d rsi-conf-f1-beam-* rsi-conf-f1-opus55-* 2>/dev/null | tr '\n' ' ')
    [[ -n "$runs" ]] || { echo "no rsi-conf-f1-beam-* / rsi-conf-f1-opus55-* runs yet"; exit 1; }
    # shellcheck disable=SC2086
    "$PY" -m benchmarks.continual_learning.analysis.rsi --cross-model --seeds $SEEDS_ALL --models $MODELS \
      --runs $runs --report-dir benchmarks/continual_learning/results/rsi-report-beam
    ;;
  *)
    sed -n '2,24p' "$0"; exit 1
    ;;
esac
