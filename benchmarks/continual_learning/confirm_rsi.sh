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
#   ./confirm_rsi.sh read                 the confirmatory readout into results/rsi-report-confirmatory
#
# Protocol constants are those of the dev runs `rsi-c5` / `rsi-c6` (fleet 1) and `rsi-e2b-c6`
# (fleet 10): CL_RSI_TIER2=1, CL_RSI_VERDICT_WAIT=240, CL_SENSELAB_TIMEOUT=90, canary timeout and
# window = the cell horizon (amendment (v) rule), --change-focus 0.5 --change-variant C --opaque-ids.
# Overrides: CL_CONF_MODELS (default "gpt-5.5 claude-opus-5"), CL_CONF_WORKERS_F1 (3),
# CL_CONF_WORKERS_E2B (3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
PY="$BENCH/../.venv/bin/python"
MODELS="${CL_CONF_MODELS:-gpt-5.5 claude-opus-5}"
SEEDS_ALL="37 41 59 73"
F1_ARMS="none raw-traces pgvector pgvector+rerank pgvector-diy+outcomes senselab senselab-rsi senselab-rsi-nocanary"
E2B_ARMS="senselab-rsi senselab-rsi-nocanary senselab-rsi-noreplay"

port_for() {  # a distinct hub port per (protocol, seed): f1 8790-8793, e2b 8794-8797
  local proto="$1" seed="$2" i=0
  for s in $SEEDS_ALL; do [[ "$s" == "$seed" ]] && break; i=$((i + 1)); done
  [[ "$proto" == "f1" ]] && echo $((8790 + i)) || echo $((8794 + i))
}

common_env() {  # the .env plus the protocol constants; nothing is printed
  set -a; source "$HERE/.env"; set +a
  export CL_RSI_TIER2=1 CL_RSI_VERDICT_WAIT=240 CL_SENSELAB_TIMEOUT=90 CL_REPLAY_SIBLINGS="8787-8797"
}

runner_cmd() {  # proto seed -> the runner argv (without env)
  local proto="$1" seed="$2"
  if [[ "$proto" == "f1" ]]; then
    echo "-m continual_learning.runner --run rsi-conf-f1-s$seed --scenarios support ci-fix --arms $F1_ARMS" \
         "--models $MODELS --seeds $seed --fleet 1 --episodes 100 --change-at 30 --change-focus 0.5" \
         "--change-variant C --opaque-ids --workers ${CL_CONF_WORKERS_F1:-3}"
  else
    echo "-m continual_learning.runner --run rsi-conf-e2b-s$seed --scenarios support ci-fix --arms $E2B_ARMS" \
         "--models $MODELS --seeds $seed --fleet 10 --episodes 200 --change-at 50 --change-focus 0.5" \
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
    run="rsi-conf-$proto-s$seed"
    cd "$BENCH"; mkdir -p "continual_learning/results/$run"
    if pgrep -f -- "--run $run " > /dev/null; then echo "$run is already running"; exit 1; fi
    horizon=100; [[ "$proto" == "e2b" ]] && horizon=200
    (
      common_env
      export CL_REPLAY_PORT="$(port_for "$proto" "$seed")" CL_RSI_CANARY_TIMEOUT="$horizon" CL_RSI_CANARY_WINDOW="$horizon"
      export PYTHONUNBUFFERED=1
      # shellcheck disable=SC2046
      perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- "$PY" $(runner_cmd "$proto" "$seed") \
        >> "continual_learning/results/$run/run.log" 2>&1 &
    )
    sleep 2
    echo "launched $run at $(date -u +%FT%TZ): $(pgrep -f -- "--run $run " | head -1)"
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
  *)
    sed -n '2,24p' "$0"; exit 1
    ;;
esac
