#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# SenseLab (AMFS) bootstrap for Fly.io Sprites
#
# Wraps install-mcp.sh for a headless, non-interactive environment. Run it ONCE
# while building the Sprite base image so the MCP config + agent instructions
# are baked in. At runtime you only inject two env vars (fast, and keeps the API
# key out of the checkpoint):
#
#   AMFS_API_KEY      — the SenseLab API key (inject at runtime, never bake)
#   AMFS_ENTITY_PATH  — the home entity this Sprite works on (bind per workload)
#
# Base-image build (bake config for all preinstalled agents):
#   AMFS_API_KEY=amfs_sk_xxx AMFS_ENTITY_PATH=sprites/acme/checkout \
#     bash sprite-bootstrap.sh
#
# Or configure a single agent:
#   bash sprite-bootstrap.sh --client gemini
#
# Any extra flags are forwarded verbatim to install-mcp.sh.
# ─────────────────────────────────────────────────────────────────────────────

INSTALLER_URL="${AMFS_INSTALLER_URL:-https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh}"
API_URL="${AMFS_HTTP_URL:-https://amfs-login.sense-lab.ai}"

log() { printf '==> %s\n' "$*"; }

# Locate install-mcp.sh: prefer a repo-local copy (this file lives two levels
# below it), otherwise download it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_INSTALLER="$SCRIPT_DIR/../../../install-mcp.sh"

run_installer() {
    if [[ -f "$LOCAL_INSTALLER" ]]; then
        log "Using repo-local installer: $LOCAL_INSTALLER"
        bash "$LOCAL_INSTALLER" "$@"
    else
        if ! command -v curl >/dev/null 2>&1; then
            echo "curl is required to fetch install-mcp.sh" >&2
            exit 1
        fi
        log "Fetching installer: $INSTALLER_URL"
        curl -sSL "$INSTALLER_URL" | bash -s -- "$@"
    fi
}

# Build the flag set. --yes keeps it non-interactive (no TTY in image builds).
ARGS=(--yes --client all --api-url "$API_URL")

if [[ -n "${AMFS_API_KEY:-}" ]]; then
    ARGS+=(--api-key "$AMFS_API_KEY")
else
    log "AMFS_API_KEY not set — configuring against local storage."
    log "For hosted SenseLab, inject AMFS_API_KEY at runtime and re-run, or bake it now."
fi

if [[ -n "${AMFS_ENTITY_PATH:-}" ]]; then
    ARGS+=(--entity-path "$AMFS_ENTITY_PATH")
    log "Binding this environment to entity_path: $AMFS_ENTITY_PATH"
else
    log "AMFS_ENTITY_PATH not set — agents will not auto-brief a specific entity."
    log "Set it (e.g. sprites/acme/checkout) so fresh Sprites hydrate the right memory."
fi

# Forward any caller-supplied flags (e.g. --client gemini) after the defaults so
# they take precedence.
run_installer "${ARGS[@]}" "$@"

log "SenseLab bootstrap complete."
log "Runtime reminder: export AMFS_API_KEY and AMFS_ENTITY_PATH before starting agents."
