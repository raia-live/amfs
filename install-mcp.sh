#!/usr/bin/env bash
set -euo pipefail

# AMFS MCP Installer
# One-line install: curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
#
# Flags:
#   --client <name|all>   Skip auto-detect; configure a specific client (or "all")
#   --api-key <key>       Use AMFS SaaS with this API key
#   --api-url <url>       SaaS API URL (default: https://amfs-login.sense-lab.ai)
#   --entity-path <path>  Bind this environment to a home entity_path (e.g.
#                         acme/checkout) so agents auto-brief it on boot
#   --saas                Configure hosted mode (pro server) without baking a key
#   --remote              Connect to the hosted MCP endpoint; sign in in a browser
#   --join <link|token>   Room invite to accept once connected (implies --remote)
#   --uninstall           Remove AMFS config from detected/specified clients
#   -y, --yes             Skip confirmation prompts

AMFS_DEFAULT_API_URL="https://amfs-login.sense-lab.ai"

# The hosted MCP endpoint. A different host from the API URL above on purpose:
# it is the identity this resource advertises in its OAuth discovery documents,
# and a client that discovers one issuer and is then sent to another treats the
# mismatch as an attack.
AMFS_DEFAULT_MCP_URL="https://mcp.sense-lab.ai/mcp"

# ── Colours & helpers ────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { printf "${BLUE}==> ${NC}%s\n" "$*"; }
success() { printf "${GREEN}==> ${NC}%s\n" "$*"; }
warn()    { printf "${YELLOW}==> ${NC}%s\n" "$*"; }
error()   { printf "${RED}==> ${NC}%s\n" "$*" >&2; }
fatal()   { error "$@"; exit 1; }

# Hosted install? True when a key or an explicit API URL was given, or --saas
# was passed. Hosted installs use the pro server and bake AMFS_HTTP_URL; the API
# key is written into the config ONLY when supplied here — so `--saas`/`--api-url`
# alone bakes a hosted-ready config whose key is injected at runtime, keeping any
# secret out of an image or checkpoint.
is_saas() { [[ "$SAAS_MODE" == true || -n "$API_KEY" ]]; }

# ── Parse arguments ──────────────────────────────────────────────────────────

CLIENT_FLAG=""
API_KEY=""
API_URL=""
ENTITY_PATH=""
MCP_URL=""
REMOTE=false
JOIN_TOKEN=""
UNINSTALL=false
AUTO_YES=false
SAAS_MODE=false

# What the closing summary is allowed to claim.
#
# In remote mode some clients are configured and some can only be told where to
# paste the endpoint, so "Done, restart your app" is true for one run and a lie
# for another — a run that found only Claude Desktop would report success while
# nothing had been connected. These count which happened.
CONFIGURED_COUNT=0
MANUAL_COUNT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --client)      CLIENT_FLAG="$2"; shift 2 ;;
        --api-key)     API_KEY="$2"; SAAS_MODE=true; shift 2 ;;
        --api-url)     API_URL="$2"; SAAS_MODE=true; shift 2 ;;
        --saas)        SAAS_MODE=true; shift ;;
        --entity-path) ENTITY_PATH="$2"; shift 2 ;;
        --mcp-url)     MCP_URL="$2"; REMOTE=true; shift 2 ;;
        --remote)      REMOTE=true; shift ;;
        --join)        JOIN_TOKEN="$2"; shift 2 ;;
        --uninstall)   UNINSTALL=true; shift ;;
        -y|--yes)      AUTO_YES=true; shift ;;
        -h|--help)
            cat <<'USAGE'
AMFS MCP Installer

Usage:
  curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
  bash install-mcp.sh [OPTIONS]

Options:
  --client <name|all>   Configure a specific client: claude-desktop, cursor,
                        claude-code, codex, gemini, windsurf, vscode, or "all"
  --api-key <key>       Connect to AMFS SaaS with this API key
  --api-url <url>       SaaS API URL (default: https://amfs-login.sense-lab.ai).
                        Implies --saas.
  --saas                Configure hosted mode (the pro server + AMFS_HTTP_URL)
                        WITHOUT baking a key. Inject AMFS_API_KEY at runtime —
                        ideal for a base image/checkpoint shared across tenants,
                        so no secret lands in the image.
  --entity-path <path>  Bind this environment to a home entity_path (e.g.
                        acme/checkout). Sets AMFS_ENTITY_PATH so agents
                        auto-brief that entity on boot — ideal for disposable
                        sandboxes and CI jobs.
  --remote              Connect to the hosted MCP endpoint instead of running a
                        local server. Your client signs in through a browser, so
                        there is no API key to create or paste.
  --join <link|token>   A room invite link to accept once connected. Implies
                        --remote, because redeeming one needs a signed-in user.
  --mcp-url <url>       Hosted MCP endpoint (default: https://mcp.sense-lab.ai/mcp)
  --uninstall           Remove AMFS MCP config from clients
  -y, --yes             Skip confirmation prompts
  -h, --help            Show this help

Notes:
  Without --api-key or --remote, this installs the open-source server backed by
  local files in ~/.amfs/ — which is what it has always done with no flags.
USAGE
            exit 0
            ;;
        *) fatal "Unknown option: $1 (use --help for usage)" ;;
    esac
done

# ── Resolve the connection mode ──────────────────────────────────────────────
#
# Three modes, and exactly one of them applies: the hosted endpoint, SaaS with a
# pasted key, or local files. The default — no flags at all — is local files, and
# nothing below changes that.

if $REMOTE && [[ -n "$API_KEY" ]]; then
    fatal "--remote and --api-key are different ways to authenticate; pass one.
  --remote signs in through a browser and needs no key.
  --api-key runs a local server against the SaaS API with the key you paste."
fi

# --join needs a signed-in user, since accepting an invite admits a person rather
# than a process. A pasted key already identifies one — its creator — so a key
# invocation is left alone; without one, the browser sign-in is how a user gets
# attached to the connection, so joining implies --remote.
if [[ -n "$JOIN_TOKEN" && -z "$API_KEY" ]]; then
    REMOTE=true
fi

if [[ -n "$JOIN_TOKEN" ]] && $UNINSTALL; then
    fatal "--join and --uninstall ask for opposite things."
fi

# ── OS detection ─────────────────────────────────────────────────────────────

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

OS="$(detect_os)"
if [[ "$OS" == "unknown" ]]; then
    fatal "Unsupported operating system: $(uname -s). This installer supports macOS and Linux."
fi

# ── Step 1: Ensure uv is installed ───────────────────────────────────────────

ensure_uv() {
    # The hosted endpoint needs no local process, so it needs no Python
    # toolchain. Installing uv anyway would download a toolchain and edit the
    # user's PATH to run nothing — the opposite of what --remote is for.
    if $REMOTE; then
        return 0
    fi

    if command -v uv &>/dev/null; then
        success "uv is already installed ($(uv --version))"
        return 0
    fi

    info "uv not found — installing..."

    if ! command -v curl &>/dev/null; then
        fatal "curl is required to install uv. Please install curl and retry."
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the env file that uv's installer creates
    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    elif [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.cargo/env"
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        fatal "uv was installed but could not be found on PATH. Try opening a new terminal and running this script again."
    fi

    success "uv installed successfully ($(uv --version))"
}

# ── Step 2: Pre-install amfs-mcp-server ──────────────────────────────────────

ensure_amfs_mcp() {
    # Nothing to install for the hosted endpoint — the client talks to it over
    # HTTP. This is the whole reason --remote exists: no uvx, no Python, and no
    # key to copy, because the client signs in through a browser.
    if $REMOTE; then
        info "Using the hosted SenseLab endpoint — nothing to install"
        return 0
    fi

    # `--refresh` so the warm-up pulls the latest published build, matching the
    # `--refresh` the generated client config uses at launch.
    if is_saas; then
        info "Installing amfs-mcp-server-pro (SaaS)..."
        uvx --refresh --from amfs-mcp-server-pro amfs-mcp-server-pro --help &>/dev/null || true
        success "amfs-mcp-server-pro is ready"
    else
        info "Installing amfs-mcp-server..."
        uvx --refresh --from amfs-mcp-server amfs-mcp-server --help &>/dev/null || true
        success "amfs-mcp-server is ready"
    fi
}

# ── Client config paths ─────────────────────────────────────────────────────

claude_desktop_config_path() {
    if [[ "$OS" == "macos" ]]; then
        echo "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    else
        echo "$HOME/.config/Claude/claude_desktop_config.json"
    fi
}

cursor_config_path() {
    echo "$HOME/.cursor/mcp.json"
}

windsurf_config_path() {
    echo "$HOME/.windsurf/mcp.json"
}

vscode_config_path() {
    echo "$HOME/.vscode/mcp.json"
}

gemini_settings_path() {
    echo "$HOME/.gemini/settings.json"
}

# Whitelist env var NAMES on the Codex senselab server so runtime-injected values
# reach it. Codex does NOT pass the ambient environment to stdio MCP subprocesses;
# only names listed in `env_vars` are forwarded from Codex's launch env (`--env`
# / the `env` table sets literal values only). So a keyless --saas bake, whose
# AMFS_API_KEY is injected at runtime, would 401 without this — the server never
# sees the key. `codex mcp add` can't set env_vars, so we patch config.toml.
codex_whitelist_env_vars() {
    [[ $# -eq 0 ]] && return 0
    local cfg="${CODEX_HOME:-$HOME/.codex}/config.toml"
    [[ -f "$cfg" ]] || { warn "Codex config not found at $cfg — skipping env_vars whitelist"; return 0; }

    # Build a TOML array literal, e.g. ["AMFS_API_KEY", "AMFS_ENTITY_PATH"].
    local list="" name
    for name in "$@"; do
        [[ -n "$list" ]] && list+=", "
        list+="\"$name\""
    done

    # Insert `env_vars = [...]` immediately after the [mcp_servers.senselab] header
    # (so it sits on the main table, before any [mcp_servers.senselab.env] subtable),
    # dropping any prior env_vars line in that block so re-runs stay idempotent.
    awk -v line="env_vars = [$list]" '
        /^\[mcp_servers\.senselab\][[:space:]]*$/ { print; print line; in_block=1; next }
        in_block && /^[[:space:]]*env_vars[[:space:]]*=/ { next }
        /^\[/ { in_block=0 }
        { print }
    ' "$cfg" > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
}

# ── Build MCP config JSON ───────────────────────────────────────────────────

UVX_PATH=""

mcp_url() {
    echo "${MCP_URL:-$AMFS_DEFAULT_MCP_URL}"
}

resolve_uvx_path() {
    if [[ -n "$UVX_PATH" ]]; then return; fi
    UVX_PATH="$(command -v uvx 2>/dev/null || echo "uvx")"
}

build_mcp_json() {
    # The hosted endpoint is a URL, not a command, so this branch returns before
    # uvx is even resolved. Placed ahead of the two below and gated on a variable
    # that is false unless --remote or --join was passed, so neither the bare
    # `curl | bash` nor the --api-key invocation reaches different code than it
    # did before this branch existed.
    if $REMOTE; then
        cat <<REMOTEJSON
{
        "type": "http",
        "url": "$(mcp_url)"
    }
REMOTEJSON
        return 0
    fi

    resolve_uvx_path
    local pkg="amfs-mcp-server"
    local -a env_pairs=()

    if is_saas; then
        pkg="amfs-mcp-server-pro"
        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
        env_pairs+=("\"AMFS_HTTP_URL\": \"$url\"")
        # Only write the key when one was actually supplied. --saas/--api-url
        # alone bakes a hosted-ready config with no secret; the key is injected
        # at runtime (ambient env), keeping it out of any image/checkpoint.
        if [[ -n "$API_KEY" ]]; then
            env_pairs+=("\"AMFS_API_KEY\": \"$API_KEY\"")
        fi
    fi
    # A bound entity_path applies in both local and SaaS mode, so agents in a
    # disposable environment auto-brief the right memory on boot.
    if [[ -n "$ENTITY_PATH" ]]; then
        env_pairs+=("\"AMFS_ENTITY_PATH\": \"$ENTITY_PATH\"")
    fi

    local env_block="{}"
    if [[ ${#env_pairs[@]} -gt 0 ]]; then
        local inner=""
        local p
        for p in "${env_pairs[@]}"; do
            if [[ -n "$inner" ]]; then inner+=$',\n'; fi
            inner+="            $p"
        done
        env_block=$'{\n'"$inner"$'\n        }'
    fi

    # `--refresh` forces uvx to re-resolve from PyPI on each launch instead of
    # reusing a stale cached environment. Without it, a user who ever ran an
    # older build keeps getting it forever — even after we publish a fix — so
    # retrieval can stay broken across releases. The small per-launch re-resolve
    # cost is worth never serving a stale MCP server.
    cat <<MCPJSON
{
        "command": "$UVX_PATH",
        "args": ["--refresh", "$pkg"],
        "env": $env_block
    }
MCPJSON
}

# ── Not throwing away the key that is already on disk ────────────────────────
#
# The `senselab` block holds AMFS_API_KEY, and the service keeps only a hash of
# a key, never the key. So the copy in a client's config file is the only copy
# there is: once this script rewrites or deletes that block, nobody — not the
# dashboard, not support — can give the same key back, and every other client
# still pointing at it stops working with no way to repair them.
#
# Three paths here do exactly that, and only one of them is asked for in those
# words. `--uninstall` at least says "remove". `--remote` clears the old stdio
# entry as a migration. And a plain re-run overwrites, which is the one that
# actually bites: `curl | bash` with no key downgrades a working keyed install
# to the free server and takes the key with it, and so does a re-run with a
# placeholder pasted out of a doc.
#
# So copy the file first — but only when there is something to lose, meaning a
# key on disk that differs from the one going in. A first install, a keyless
# install over nothing, and an idempotent re-run with the same key all leave no
# backup behind, because in none of them does a key stop existing.

# The copy itself: where it goes, and saying so.
#
# 0600 because the file it came from holds a credential and the directory it
# lands in may not be private. A failed copy is reported rather than fatal: it
# almost certainly means the file is read-only, so the write about to follow
# will fail too and abandoning the install here would only make the message
# about the wrong thing.
keep_a_copy_of_the_key() {
    local file="$1" dest
    dest="$file.senselab-backup-$(date -u +%Y%m%d%H%M%S)"

    if cp "$file" "$dest" 2>/dev/null; then
        chmod 600 "$dest" 2>/dev/null || true
        info "Kept a copy of the API key already in that config: $dest"
        echo "    SenseLab stores keys hashed and cannot reissue the same one, so"
        echo "    this copy is the only one left. Delete it once you are sure the"
        echo "    new connection works."
    else
        warn "Could not copy $file before changing it."
        echo "    The API key in it is about to be replaced, and SenseLab cannot"
        echo "    hand the same one back. Copy the file yourself if you need it."
    fi
}

# Before this script rewrites or deletes a `senselab` block it owns.
#
# `incoming` is the key that is about to take its place, empty for a removal or
# a keyless install — which is why the comparison is against a string and not a
# flag: "no key going in" and "a different key going in" both lose what is
# there, and both want a copy.
backup_key_at_risk() {
    local config_file="$1" incoming="${2:-}" existing=""

    [[ -f "$config_file" ]] || return 0

    existing="$(python3 -c '
import json, sys

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
except (json.JSONDecodeError, OSError, ValueError):
    sys.exit(0)

entry = (config.get("mcpServers") or {}).get("senselab")
if isinstance(entry, dict):
    print((entry.get("env") or {}).get("AMFS_API_KEY") or "")
' "$config_file" 2>/dev/null || true)"

    if [[ -n "$existing" && "$existing" != "$incoming" ]]; then
        keep_a_copy_of_the_key "$config_file"
    fi
}

# Before asking a client's own CLI to drop the entry.
#
# Claude Code and Codex are configured through `claude mcp` and `codex mcp`
# rather than by writing their files, so there is no block of ours to read and
# compare. The marker is enough: if AMFS_API_KEY is in the file the CLI keeps,
# a copy of that file genuinely preserves the key, and if it is not, this says
# nothing rather than claiming a rescue it did not perform. That also covers
# being wrong about where a given version of those tools stores it.
backup_cli_store() {
    local file="$1" incoming="${2:-}"

    [[ -f "$file" ]] || return 0
    grep -q "AMFS_API_KEY" "$file" 2>/dev/null || return 0

    # Re-adding the key already in there changes nothing, so there is nothing to
    # keep — but only the whole stored value counts as "already in there".
    #
    # Matching the incoming key as a bare substring reintroduced the bug this
    # function exists to prevent. A placeholder like `amfs_k` is a substring of
    # every real key that happens to begin with those characters, so the copy
    # was skipped and `codex mcp remove` then deleted the only one left. Short
    # keys are exactly the ones people paste by mistake.
    #
    # Both formats these CLIs write quote the value — JSON for Claude Code,
    # TOML for Codex — so requiring the quotes, on the same line as the name,
    # compares the value end to end. A form neither branch recognises falls
    # through to making a copy, which is the direction to be wrong in: a spare
    # file costs nothing against a credential that cannot be reissued.
    if [[ -n "$incoming" ]] && grep -F "AMFS_API_KEY" "$file" 2>/dev/null |
        grep -qF -e "\"$incoming\"" -e "'$incoming'"; then
        return 0
    fi

    keep_a_copy_of_the_key "$file"
}

# ── JSON merge (portable, no jq dependency) ──────────────────────────────────

inject_mcp_config() {
    local config_file="$1"
    local amfs_block
    amfs_block="$(build_mcp_json)"

    backup_key_at_risk "$config_file" "$API_KEY"

    local dir
    dir="$(dirname "$config_file")"
    mkdir -p "$dir"

    if [[ ! -f "$config_file" ]]; then
        cat > "$config_file" <<NEWJSON
{
    "mcpServers": {
        "senselab": $amfs_block
    }
}
NEWJSON
        return 0
    fi

    # File exists — use python (available on macOS and most Linux) for safe JSON merge
    python3 -c "
import json, sys

config_path = sys.argv[1]
amfs_block = json.loads(sys.argv[2])

with open(config_path, 'r') as f:
    try:
        config = json.load(f)
    except json.JSONDecodeError:
        config = {}

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['senselab'] = amfs_block

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
    f.write('\n')
" "$config_file" "$amfs_block"
}

# Whether a config file holds a local stdio `senselab` entry: present, absent, or
# unreadable.
#
# Three answers rather than two, and a word on stdout rather than an exit status.
#
# Three, because `grep -q '"senselab"'` cannot tell "the entry is gone" from "the
# file is unparseable so nothing was removed" — and remove_mcp_config exits 0
# either way, so a caller trusting grep reports a removal that did not happen and
# leaves the stale local server it was removing. Grep also matches the name in a
# path or another server's argument list.
#
# A word, because an exit status has to carry both the answer and whether asking
# worked, and those are different things. If python3 is not on PATH the status is
# 127, which as a number falls outside a three-way branch and silently means
# nothing happens. Here every failure of the check becomes "unreadable", which is
# the answer that warns rather than the one that stays quiet.
senselab_entry_state() {
    local config_file="$1" state=""

    if [[ ! -f "$config_file" ]]; then
        state="absent"
    else
        state="$(python3 -c "
import json, sys

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
except (json.JSONDecodeError, OSError, ValueError):
    sys.exit(1)

entry = (config.get('mcpServers') or {}).get('senselab')
# A url entry is this script's own remote config, not a local server to clear.
print('present' if isinstance(entry, dict) and entry.get('command') else 'absent')
" "$config_file" 2>/dev/null || true)"
    fi

    case "$state" in
        present|absent) echo "$state" ;;
        *) echo "unreadable" ;;
    esac
}

remove_mcp_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        return 0
    fi

    backup_key_at_risk "$config_file" ""

    python3 -c "
import json, sys

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    try:
        config = json.load(f)
    except json.JSONDecodeError:
        sys.exit(0)

if 'mcpServers' in config and 'senselab' in config['mcpServers']:
    del config['mcpServers']['senselab']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        f.write('\n')
" "$config_file"
}

# ── SenseLab memory instructions ─────────────────────────────────────────────
#
# So agents recall proactively (esp. for personal facts) instead of relying only
# on the advisory MCP instructions, we install a short "recall-first" guide into
# the clients that auto-load on-disk instructions: Claude Code (a Skill +
# ~/.claude/CLAUDE.md) and Codex (~/.codex/AGENTS.md). Blocks are delimited by
# HTML-comment markers so they're idempotent (re-runs replace, not duplicate)
# and removable by --uninstall. Claude Desktop has no writable instruction file
# (its "personal preferences" are an app/account profile setting), so it can't
# be auto-configured here — the dashboard onboarding shows a copy/paste snippet.

SENSELAB_MARKER_BEGIN="<!-- >>> senselab-memory >>> -->"
SENSELAB_MARKER_END="<!-- <<< senselab-memory <<< -->"

senselab_instructions_body() {
    cat <<'MD'
## SenseLab memory (recall-first)

You are connected to SenseLab (AMFS), a persistent memory shared across all your
tools and sessions. Use it proactively:

- **Start of session:** call `amfs_set_identity`, then `amfs_briefing(entity_path="repo/module")` for compiled context before working in a codebase.
- **RECALL-FIRST (do not skip):** whenever the user asks what you know / remember / have saved about ANYTHING — including personal facts and preferences ("what food do I like?") — call `amfs_retrieve(query="<the user's words>")` BEFORE answering. `entity_path`/`key` are optional; it searches by meaning across everything you can see. Never tell the user you have no memory of something without running `amfs_retrieve` first.
- **Remember things:** when the user shares a durable fact, preference, or decision, or says "remember…", call `amfs_write(entity_path, key, value)`.
- `amfs_read`/`amfs_recall` need an EXACT key — a miss there means "try `amfs_retrieve`", NOT "nothing is stored".
MD
    # When this environment is bound to a home entity, spell out the concrete
    # path so a fresh agent hydrates the right memory without guessing.
    if [[ -n "$ENTITY_PATH" ]]; then
        printf '\n- **This environment is bound to `%s`.** Right after `amfs_set_identity`, call `amfs_briefing(entity_path="%s")` to load what prior sessions here learned, and default your reads/writes to `%s` unless the task clearly concerns another entity.\n' \
            "$ENTITY_PATH" "$ENTITY_PATH" "$ENTITY_PATH"
    fi
}

upsert_senselab_block() {
    # Idempotently insert/replace the marker-delimited block in a markdown file.
    local file="$1"
    mkdir -p "$(dirname "$file")"
    SENSELAB_BODY="$(senselab_instructions_body)" \
    SENSELAB_BEGIN="$SENSELAB_MARKER_BEGIN" \
    SENSELAB_END="$SENSELAB_MARKER_END" \
    python3 - "$file" <<'PY'
import os, sys
path = sys.argv[1]
begin, end, body = os.environ["SENSELAB_BEGIN"], os.environ["SENSELAB_END"], os.environ["SENSELAB_BODY"]
block = begin + "\n" + body.rstrip("\n") + "\n" + end
try:
    with open(path) as f:
        content = f.read()
except FileNotFoundError:
    content = ""
if begin in content and end in content:
    pre = content.split(begin)[0].rstrip("\n")
    post = content.split(end, 1)[1].lstrip("\n")
    content = (pre + "\n\n" + post).strip("\n")
content = (content.rstrip("\n") + "\n\n" if content.strip() else "") + block + "\n"
with open(path, "w") as f:
    f.write(content)
PY
}

remove_senselab_block() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    SENSELAB_BEGIN="$SENSELAB_MARKER_BEGIN" SENSELAB_END="$SENSELAB_MARKER_END" \
    python3 - "$file" <<'PY'
import os, sys
path = sys.argv[1]
begin, end = os.environ["SENSELAB_BEGIN"], os.environ["SENSELAB_END"]
with open(path) as f:
    content = f.read()
if begin in content and end in content:
    pre = content.split(begin)[0].rstrip("\n")
    post = content.split(end, 1)[1].lstrip("\n")
    content = (pre + "\n\n" + post).strip("\n")
    with open(path, "w") as f:
        f.write(content + ("\n" if content else ""))
PY
}

install_claude_code_skill() {
    local skill_dir="$HOME/.claude/skills/senselab-memory"
    mkdir -p "$skill_dir"
    {
        echo "---"
        echo "name: senselab-memory"
        echo "description: Use SenseLab (AMFS) as persistent memory. When the user asks what you know/remember/have saved about anything (including personal preferences), call amfs_retrieve BEFORE answering; save durable facts with amfs_write."
        echo "---"
        echo ""
        senselab_instructions_body
    } > "$skill_dir/SKILL.md"
}

remove_claude_code_skill() {
    rm -rf "$HOME/.claude/skills/senselab-memory"
}

# ── Client detection ─────────────────────────────────────────────────────────

declare -a DETECTED_CLIENTS=()

detect_clients() {
    DETECTED_CLIENTS=()

    local claude_path
    claude_path="$(claude_desktop_config_path)"
    local claude_dir
    claude_dir="$(dirname "$claude_path")"
    if [[ -d "$claude_dir" ]]; then
        DETECTED_CLIENTS+=("claude-desktop")
    fi

    if [[ -d "$HOME/.cursor" ]]; then
        DETECTED_CLIENTS+=("cursor")
    fi

    if command -v claude &>/dev/null; then
        DETECTED_CLIENTS+=("claude-code")
    fi

    if command -v codex &>/dev/null; then
        DETECTED_CLIENTS+=("codex")
    fi

    if command -v gemini &>/dev/null || [[ -d "$HOME/.gemini" ]]; then
        DETECTED_CLIENTS+=("gemini")
    fi

    local windsurf_dir
    windsurf_dir="$(dirname "$(windsurf_config_path)")"
    if [[ -d "$windsurf_dir" ]]; then
        DETECTED_CLIENTS+=("windsurf")
    fi

    if [[ -d "$HOME/.vscode" ]]; then
        DETECTED_CLIENTS+=("vscode")
    fi
}

# ── Configure a single client ────────────────────────────────────────────────

# Say where to paste the endpoint, for a client whose config file cannot hold it.
#
# `--remote` writes `{"type": "http", "url": …}`, which is what Cursor, Claude
# Code, Codex and VS Code understand. Claude Desktop's config file takes stdio
# servers only — a `url` entry there is ignored — and Windsurf names the same
# thing `serverUrl`, so it is ignored too. Both reach remote servers through
# their own UI instead.
#
# Writing the file anyway is worse than not touching it: the script says
# "Configured", the client silently has no SenseLab, and the person has no reason
# to look at the one place that would have worked. So neither is written to in
# remote mode, and both are told where to go by hand. The stdio path is
# unaffected and still writes both files as it always did.
# Report a client we actually wrote a config for, and count it as one.
configured() {
    CONFIGURED_COUNT=$((CONFIGURED_COUNT + 1))
    success "$1"
}

connector_instructions() {
    local label="$1" where="$2" config_file="${3:-}"
    MANUAL_COUNT=$((MANUAL_COUNT + 1))

    # Take the old stdio entry out on the way past.
    #
    # Not writing was only half of it. Someone switching an existing install to
    # --remote has a `senselab` block already in this file, and leaving it there
    # means the client keeps launching the local server: the script says to add a
    # connector, they add one, and now there are two — or, if they do not, the
    # migration silently did nothing and they are still on the old server. Both
    # are worse than either half alone, because the summary says the hosted
    # endpoint is what they are on.
    #
    # Reported from the state afterwards rather than from the attempt. A file this
    # cannot parse is left untouched by remove_mcp_config, which exits 0 either
    # way, so "we tried" is not evidence and the one case worth catching is
    # precisely the one where the old server is still there.
    if [[ -n "$config_file" ]]; then
        local state
        state="$(senselab_entry_state "$config_file")"

        if [[ "$state" == "present" ]]; then
            # `|| true` because a failure here is not a reason to abandon the
            # install. The file may be read-only, and the connector instructions
            # below are still the right thing to print — under `set -e` a bare
            # call would exit before saying anything at all, including the warning
            # that the old entry is still there.
            remove_mcp_config "$config_file" || true

            # Asked again, because the answer is the only evidence a removal
            # happened. "absent" is the one state that is one; the other two both
            # mean the old server may still start, and they share a message
            # because they share a recovery — open the file and look.
            state="$(senselab_entry_state "$config_file")"
            if [[ "$state" == "absent" ]]; then
                success "Removed the old local $label entry ($config_file)"
            else
                warn "Could not confirm the old local $label entry is gone."
                echo "    Check $config_file for a \"senselab\" block and remove it —"
                echo "    until then $label may keep starting the old local server."
            fi
        elif [[ "$state" == "unreadable" ]]; then
            warn "$config_file could not be read, so it was left alone."
            echo "    If it has a \"senselab\" block, remove it by hand: otherwise"
            echo "    $label keeps starting the old local server."
        fi
    fi

    warn "$label cannot be configured from here in remote mode."
    echo "    Add it once, by hand: $where"
    echo "    Endpoint: $(mcp_url)"
    echo "    It signs in through a browser; there is no key to paste."
}

# What is left to do about the invite, if one was passed.
#
# The invite is not redeemed here, and that is not a shortcoming to fix later:
# accepting one needs a signed-in user, and the sign-in happens inside the client
# after this script has exited. Doing it here would mean minting a credential of
# our own first, which is the copy-and-paste step --remote exists to remove.
#
# What to say differs by mode, which is why this is a function rather than a block
# in main — the two branches are the thing worth testing, and main cannot be
# called without configuring real clients.
join_next_steps() {
    [[ -n "$JOIN_TOKEN" ]] || return 0

    info "One step left: accept the room invitation."
    if $REMOTE; then
        echo "  Once your client has signed in, ask your agent:"
    else
        # The key path has no sign-in to wait for: the key already names its
        # owner, and that owner is who the invite admits. Telling this person to
        # wait for a browser they will never see reads as a stuck install.
        echo "  Ask your agent:"
    fi
    echo ""
    echo "    Join my SenseLab room with amfs_room_redeem, invite ${JOIN_TOKEN}"
    echo ""
    if $REMOTE; then
        echo "  Or open the invite link in a browser, which does the same thing."
    else
        # Not the same thing on this path. A browser admits whoever is logged
        # into it, which need not be the account the key belongs to — so
        # following it can join the room as one account while the agent that is
        # connected belongs to another, and nothing reports the mismatch.
        echo "  Opening the link in a browser also works, but it admits"
        echo "  whoever is signed in there — which may not be the account"
        echo "  this API key belongs to. Asking the agent avoids the question."
    fi
    echo ""
}

configure_client() {
    local client="$1"

    case "$client" in
        claude-desktop)
            local path
            path="$(claude_desktop_config_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                success "Removed AMFS from Claude Desktop config"
            elif $REMOTE; then
                connector_instructions "Claude Desktop" \
                    "Settings → Connectors → Add custom connector" "$path"
            else
                inject_mcp_config "$path"
                configured "Configured Claude Desktop ($path)"
            fi
            ;;
        cursor)
            local path
            path="$(cursor_config_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                success "Removed AMFS from Cursor config"
            else
                inject_mcp_config "$path"
                configured "Configured Cursor ($path)"
            fi
            ;;
        claude-code)
            if $UNINSTALL; then
                if command -v claude &>/dev/null; then
                    backup_cli_store "$HOME/.claude.json"
                    claude mcp remove senselab 2>/dev/null || true
                    success "Removed AMFS from Claude Code"
                fi
                remove_claude_code_skill
                remove_senselab_block "$HOME/.claude/CLAUDE.md"
            else
                if ! command -v claude &>/dev/null; then
                    warn "Claude Code CLI (claude) not found on PATH — skipping"
                    return 1
                fi
                local args
                if $REMOTE; then
                    # Claude Code has native support for an HTTP transport, and
                    # runs the OAuth flow itself on first use.
                    args=("mcp" "add" "--transport" "http" "senselab" "$(mcp_url)")
                else
                    resolve_uvx_path
                    local pkg="amfs-mcp-server"
                    if is_saas; then pkg="amfs-mcp-server-pro"; fi
                    # `--refresh` (a uvx flag, before the package) forces a fresh
                    # re-resolve each launch so users never get stuck on a stale
                    # cached build after we publish a fix. Build a single args array
                    # (avoids empty-array expansion errors under bash 3.2 + set -u).
                    args=("mcp" "add" "senselab")
                    if is_saas; then
                        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                        args+=("-e" "AMFS_HTTP_URL=$url")
                        if [[ -n "$API_KEY" ]]; then args+=("-e" "AMFS_API_KEY=$API_KEY"); fi
                    fi
                    # A bound entity_path applies in local and SaaS mode (not the
                    # hosted endpoint, which binds per connection server-side).
                    if [[ -n "$ENTITY_PATH" ]]; then
                        args+=("-e" "AMFS_ENTITY_PATH=$ENTITY_PATH")
                    fi
                    args+=("--" "$UVX_PATH" "--refresh" "$pkg")
                fi
                # Replace any existing entry so re-runs stay idempotent. Pass the
                # incoming key so a same-key re-run skips the backup (as Codex
                # does) instead of spawning a fresh .senselab-backup-* each time.
                backup_cli_store "$HOME/.claude.json" "$API_KEY"
                claude mcp remove senselab 2>/dev/null || true
                claude "${args[@]}"
                configured "Configured Claude Code"
                install_claude_code_skill
                upsert_senselab_block "$HOME/.claude/CLAUDE.md"
                success "Installed SenseLab recall-first memory guide (skill + ~/.claude/CLAUDE.md)"
            fi
            ;;
        codex)
            if $UNINSTALL; then
                if command -v codex &>/dev/null; then
                    backup_cli_store "$HOME/.codex/config.toml"
                    codex mcp remove senselab 2>/dev/null || true
                    success "Removed AMFS from Codex"
                fi
                remove_senselab_block "$HOME/.codex/AGENTS.md"
            else
                if ! command -v codex &>/dev/null; then
                    warn "Codex CLI (codex) not found on PATH — skipping"
                    return 1
                fi
                local args
                if $REMOTE; then
                    args=("mcp" "add" "senselab" "--url" "$(mcp_url)")
                else
                    resolve_uvx_path
                    local pkg="amfs-mcp-server"
                    if is_saas; then pkg="amfs-mcp-server-pro"; fi
                    # codex mcp add <name> [--env KEY=VAL]... -- <command> [args...]
                    # `--refresh` (uvx flag, before the package) forces a fresh
                    # re-resolve each launch so a stale cache can't pin users to an
                    # old build after we publish a fix.
                    args=("mcp" "add" "senselab")
                    if is_saas; then
                        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                        args+=("--env" "AMFS_HTTP_URL=$url")
                        if [[ -n "$API_KEY" ]]; then args+=("--env" "AMFS_API_KEY=$API_KEY"); fi
                    fi
                    if [[ -n "$ENTITY_PATH" ]]; then
                        args+=("--env" "AMFS_ENTITY_PATH=$ENTITY_PATH")
                    fi
                    args+=("--" "$UVX_PATH" "--refresh" "$pkg")
                fi
                # Replace any existing entry so re-runs stay idempotent — which
                # means a re-run with a different key, or none, drops the old one.
                backup_cli_store "$HOME/.codex/config.toml" "$API_KEY"
                codex mcp remove senselab 2>/dev/null || true
                codex "${args[@]}"
                # Forward names that arrive at runtime rather than being baked as
                # literals above — Codex drops ambient env for stdio servers, so
                # without this the injected AMFS_API_KEY (keyless --saas) and an
                # unbaked AMFS_ENTITY_PATH never reach the server. The hosted
                # endpoint runs no local subprocess, so skip it under --remote.
                if ! $REMOTE; then
                    local fwd_names=()
                    if is_saas && [[ -z "$API_KEY" ]]; then fwd_names+=("AMFS_API_KEY"); fi
                    if [[ -z "$ENTITY_PATH" ]]; then fwd_names+=("AMFS_ENTITY_PATH"); fi
                    if [[ ${#fwd_names[@]} -gt 0 ]]; then
                        codex_whitelist_env_vars "${fwd_names[@]}"
                    fi
                fi
                configured "Configured Codex"
                upsert_senselab_block "$HOME/.codex/AGENTS.md"
                success "Installed SenseLab recall-first memory guide (~/.codex/AGENTS.md)"
            fi
            ;;
        gemini)
            # Gemini CLI reads MCP servers from ~/.gemini/settings.json under the
            # same "mcpServers" key as the other JSON clients, so the shared
            # merge helper applies. Its recall-first guide goes in ~/.gemini/GEMINI.md.
            local path
            path="$(gemini_settings_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                remove_senselab_block "$HOME/.gemini/GEMINI.md"
                success "Removed AMFS from Gemini CLI"
            else
                inject_mcp_config "$path"
                configured "Configured Gemini CLI ($path)"
                upsert_senselab_block "$HOME/.gemini/GEMINI.md"
                success "Installed SenseLab recall-first memory guide (~/.gemini/GEMINI.md)"
            fi
            ;;
        windsurf)
            local path
            path="$(windsurf_config_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                success "Removed AMFS from Windsurf config"
            elif $REMOTE; then
                connector_instructions "Windsurf" \
                    "Settings → Cascade → MCP servers → Add server" "$path"
            else
                inject_mcp_config "$path"
                configured "Configured Windsurf ($path)"
            fi
            ;;
        vscode)
            local path
            path="$(vscode_config_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                success "Removed AMFS from VS Code config"
            else
                inject_mcp_config "$path"
                configured "Configured VS Code ($path)"
            fi
            ;;
        *)
            error "Unknown client: $client"
            return 1
            ;;
    esac
}

# ── Interactive menu ─────────────────────────────────────────────────────────

prompt_clients() {
    detect_clients

    if [[ ${#DETECTED_CLIENTS[@]} -eq 0 ]]; then
        warn "No supported MCP clients detected."
        echo ""
        echo "You can manually add this to your MCP client config:"
        echo ""
        resolve_uvx_path
        printf '  "senselab": '
        build_mcp_json | sed 's/^/  /'
        echo ""
        return 1
    fi

    echo ""
    printf "${BOLD}Detected MCP clients:${NC}\n"
    echo ""
    local i=1
    for client in "${DETECTED_CLIENTS[@]}"; do
        printf "  ${GREEN}%d)${NC} %s\n" "$i" "$client"
        ((i++))
    done
    echo ""

    if $AUTO_YES; then
        info "Configuring all detected clients (--yes)"
        for client in "${DETECTED_CLIENTS[@]}"; do
            configure_client "$client" || true
        done
        return 0
    fi

    local action="Configure"
    if $UNINSTALL; then action="Remove AMFS from"; fi

    # Said before the question, because after it the key is already gone. Anyone
    # uninstalling to reinstall cleanly is about to find out that "cleanly" costs
    # them a credential, and this is the last moment that is useful to know.
    if $UNINSTALL; then
        warn "This takes your API key with it."
        echo "    SenseLab stores keys hashed, so the same one cannot be reissued."
        echo "    A copy of any config holding it is kept next to the original."
        echo ""
    fi

    printf "${BOLD}$action all detected clients? [Y/n/list]:${NC} "
    read -r answer </dev/tty 2>/dev/null || answer="y"
    answer="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"

    case "$answer" in
        ""|y|yes)
            for client in "${DETECTED_CLIENTS[@]}"; do
                configure_client "$client" || true
            done
            ;;
        n|no)
            info "Skipped client configuration."
            echo ""
            echo "You can manually add this to your MCP client config:"
            echo ""
            resolve_uvx_path
            printf '  "senselab": '
            build_mcp_json | sed 's/^/  /'
            echo ""
            ;;
        *)
            echo ""
            printf "Enter client numbers separated by spaces (e.g. 1 3): "
            read -r selections </dev/tty 2>/dev/null || selections=""
            for sel in $selections; do
                local idx=$((sel - 1))
                if [[ $idx -ge 0 && $idx -lt ${#DETECTED_CLIENTS[@]} ]]; then
                    configure_client "${DETECTED_CLIENTS[$idx]}" || true
                else
                    warn "Invalid selection: $sel"
                fi
            done
            ;;
    esac
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    echo ""
    printf "${BOLD}AMFS MCP Installer${NC}\n"
    echo "──────────────────"
    echo ""

    if $UNINSTALL; then
        info "Uninstall mode"
        if [[ -n "$CLIENT_FLAG" ]]; then
            if [[ "$CLIENT_FLAG" == "all" ]]; then
                detect_clients
                for client in "${DETECTED_CLIENTS[@]}"; do
                    configure_client "$client" || true
                done
            else
                configure_client "$CLIENT_FLAG"
            fi
        else
            prompt_clients
        fi
        echo ""
        success "AMFS MCP config removed. Restart your IDE to apply."
        return 0
    fi

    # Step 1: uv
    ensure_uv
    echo ""

    # Step 2: amfs-mcp-server
    ensure_amfs_mcp
    echo ""

    # Step 3: Configure clients
    if [[ -n "$CLIENT_FLAG" ]]; then
        if [[ "$CLIENT_FLAG" == "all" ]]; then
            detect_clients
            for client in "${DETECTED_CLIENTS[@]}"; do
                configure_client "$client" || true
            done
        else
            configure_client "$CLIENT_FLAG"
        fi
    else
        prompt_clients || true
    fi

    echo ""
    echo "──────────────────"
    # Only claim a restart will do it if something was written for a client to
    # pick up. A remote run that found only Claude Desktop has written nothing,
    # and telling that person to restart sends them to look for a connection
    # that will not be there instead of at the instructions just printed.
    if (( CONFIGURED_COUNT > 0 )); then
        success "Done! Restart your IDE/app to connect to AMFS."
    elif (( MANUAL_COUNT > 0 )); then
        warn "Nothing was configured automatically — see the step above."
    else
        warn "No clients were configured."
    fi
    echo ""

    if $REMOTE && (( CONFIGURED_COUNT > 0 )); then
        info "Configured the hosted SenseLab endpoint ($(mcp_url))"
        echo "  Your client will open a browser to sign in the first time it"
        echo "  connects. There is no API key to create."
        echo ""
        # Named per client because the reload step differs and getting it wrong
        # looks like the install failed: the config is on disk and the client is
        # still running without it.
        echo "  Claude Code:  run /mcp and approve the sign-in"
        echo "  Cursor:       Settings → MCP, then sign in"
        echo "  Codex:        restart codex"
        echo "  Others:       restart the app"
    elif $REMOTE; then
        info "The hosted SenseLab endpoint is $(mcp_url)"
        echo "  Signing in happens in a browser; there is no API key to create."
    elif [[ -n "$API_KEY" ]]; then
        info "Connected to AMFS SaaS (${API_URL:-$AMFS_DEFAULT_API_URL})"
    elif is_saas; then
        info "Configured hosted mode (${API_URL:-$AMFS_DEFAULT_API_URL}) — no key baked."
        echo "  Inject AMFS_API_KEY in the environment at runtime to connect."
    else
        info "Using local filesystem storage (~/.amfs/)"
        echo "  To connect to AMFS SaaS, re-run with --api-key <key> (or --saas)"
    fi
    echo ""

    join_next_steps

    # Claude Desktop has no writable instructions file — surface the one manual
    # step so it recalls proactively like the file-based clients now do.
    if [[ -d "$(dirname "$(claude_desktop_config_path)")" ]]; then
        warn "Claude Desktop: paste this into Settings → General → \"Instructions for Claude\" for proactive recall:"
        echo "  SenseLab is my personal + work memory, connected as the \"senselab\" MCP connector"
        echo "  (tools start with amfs_). It stores everything I ask it to remember — personal facts,"
        echo "  preferences, people, plans — not just code. Whenever I ask what I like, prefer, know,"
        echo "  remember, or have saved about anything, you MUST call amfs_retrieve with my question"
        echo "  first, then answer from what it returns. Never answer from your own memory, and never"
        echo "  say something \"hasn't come up\" or that SenseLab \"isn't for that\" until you've called"
        echo "  amfs_retrieve. When I say \"remember…\" or share a durable fact, call amfs_write."
        echo ""
    fi
}

main
