#!/usr/bin/env bash
set -euo pipefail

# AMFS MCP Installer
# One-line install: curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
#
# Flags:
#   --client <name|all>   Skip auto-detect; configure a specific client (or "all")
#   --api-key <key>       Use AMFS SaaS with this API key
#   --api-url <url>       SaaS API URL (default: https://amfs-login.sense-lab.ai)
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

# ── Parse arguments ──────────────────────────────────────────────────────────

CLIENT_FLAG=""
API_KEY=""
API_URL=""
MCP_URL=""
REMOTE=false
JOIN_TOKEN=""
UNINSTALL=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --client)     CLIENT_FLAG="$2"; shift 2 ;;
        --api-key)    API_KEY="$2"; shift 2 ;;
        --api-url)    API_URL="$2"; shift 2 ;;
        --mcp-url)    MCP_URL="$2"; REMOTE=true; shift 2 ;;
        --remote)     REMOTE=true; shift ;;
        --join)       JOIN_TOKEN="$2"; shift 2 ;;
        --uninstall)  UNINSTALL=true; shift ;;
        -y|--yes)     AUTO_YES=true; shift ;;
        -h|--help)
            cat <<'USAGE'
AMFS MCP Installer

Usage:
  curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
  bash install-mcp.sh [OPTIONS]

Options:
  --client <name|all>   Configure a specific client: claude-desktop, cursor,
                        claude-code, codex, windsurf, vscode, or "all"
  --api-key <key>       Connect to AMFS SaaS with this API key
  --api-url <url>       SaaS API URL (default: https://amfs-login.sense-lab.ai)
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
    if [[ -n "$API_KEY" ]]; then
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
    local env_block="{}"
    local pkg="amfs-mcp-server"

    if [[ -n "$API_KEY" ]]; then
        pkg="amfs-mcp-server-pro"
        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
        env_block=$(cat <<ENVJSON
{
            "AMFS_HTTP_URL": "$url",
            "AMFS_API_KEY": "$API_KEY"
        }
ENVJSON
)
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

# ── JSON merge (portable, no jq dependency) ──────────────────────────────────

inject_mcp_config() {
    local config_file="$1"
    local amfs_block
    amfs_block="$(build_mcp_json)"

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

remove_mcp_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        return 0
    fi

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
connector_instructions() {
    local label="$1" where="$2"
    warn "$label cannot be configured from here in remote mode."
    echo "    Add it once, by hand: $where"
    echo "    Endpoint: $(mcp_url)"
    echo "    It signs in through a browser; there is no key to paste."
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
                    "Settings → Connectors → Add custom connector"
            else
                inject_mcp_config "$path"
                success "Configured Claude Desktop ($path)"
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
                success "Configured Cursor ($path)"
            fi
            ;;
        claude-code)
            if $UNINSTALL; then
                if command -v claude &>/dev/null; then
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
                    # Re-runs stay idempotent; adding a name that exists fails.
                    claude mcp remove senselab 2>/dev/null || true
                else
                    resolve_uvx_path
                    local pkg="amfs-mcp-server"
                    if [[ -n "$API_KEY" ]]; then pkg="amfs-mcp-server-pro"; fi
                    # `--refresh` (a uvx flag, before the package) forces a fresh
                    # re-resolve each launch so users never get stuck on a stale
                    # cached build after we publish a fix.
                    args=("mcp" "add" "senselab" "--" "$UVX_PATH" "--refresh" "$pkg")
                    if [[ -n "$API_KEY" ]]; then
                        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                        args=("mcp" "add" "senselab" "-e" "AMFS_HTTP_URL=$url" "-e" "AMFS_API_KEY=$API_KEY" "--" "$UVX_PATH" "--refresh" "$pkg")
                    fi
                fi
                claude "${args[@]}"
                success "Configured Claude Code"
                install_claude_code_skill
                upsert_senselab_block "$HOME/.claude/CLAUDE.md"
                success "Installed SenseLab recall-first memory guide (skill + ~/.claude/CLAUDE.md)"
            fi
            ;;
        codex)
            if $UNINSTALL; then
                if command -v codex &>/dev/null; then
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
                    if [[ -n "$API_KEY" ]]; then pkg="amfs-mcp-server-pro"; fi
                    # codex mcp add <name> [--env KEY=VAL]... -- <command> [args...]
                    # `--refresh` (uvx flag, before the package) forces a fresh
                    # re-resolve each launch so a stale cache can't pin users to an
                    # old build after we publish a fix.
                    args=("mcp" "add" "senselab" "--" "$UVX_PATH" "--refresh" "$pkg")
                    if [[ -n "$API_KEY" ]]; then
                        local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                        args=("mcp" "add" "senselab" "--env" "AMFS_HTTP_URL=$url" "--env" "AMFS_API_KEY=$API_KEY" "--" "$UVX_PATH" "--refresh" "$pkg")
                    fi
                fi
                # Replace any existing entry so re-runs stay idempotent.
                codex mcp remove senselab 2>/dev/null || true
                codex "${args[@]}"
                success "Configured Codex"
                upsert_senselab_block "$HOME/.codex/AGENTS.md"
                success "Installed SenseLab recall-first memory guide (~/.codex/AGENTS.md)"
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
                    "Settings → Cascade → MCP servers → Add server"
            else
                inject_mcp_config "$path"
                success "Configured Windsurf ($path)"
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
                success "Configured VS Code ($path)"
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
    success "Done! Restart your IDE/app to connect to AMFS."
    echo ""

    if $REMOTE; then
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
    elif [[ -n "$API_KEY" ]]; then
        info "Connected to AMFS SaaS (${API_URL:-$AMFS_DEFAULT_API_URL})"
    else
        info "Using local filesystem storage (~/.amfs/)"
        echo "  To connect to AMFS SaaS, re-run with --api-key <key>"
    fi
    echo ""

    if [[ -n "$JOIN_TOKEN" ]]; then
        # Not redeemed here, and this is not a shortcoming to fix later. Accepting
        # an invite needs a signed-in user, and the sign-in has not happened yet
        # — it happens inside the client, after this script has exited. Doing it
        # here would mean minting a credential of our own first, which is the
        # copy-and-paste step --remote exists to remove.
        info "One step left: accept the room invitation."
        echo "  Once your client has signed in, ask your agent:"
        echo ""
        echo "    Join my SenseLab room with amfs_room_redeem, invite ${JOIN_TOKEN}"
        echo ""
        echo "  Or open the invite link in a browser, which does the same thing."
        echo ""
    fi

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
