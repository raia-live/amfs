#!/usr/bin/env bash
set -euo pipefail

# AMFS MCP Installer
# One-line install: curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
#
# Flags:
#   --client <name|all>   Skip auto-detect; configure a specific client (or "all")
#   --api-key <key>       Use AMFS SaaS with this API key
#   --api-url <url>       SaaS API URL (default: https://amfs-login.sense-lab.ai)
#   --uninstall           Remove AMFS config from detected/specified clients
#   -y, --yes             Skip confirmation prompts

AMFS_DEFAULT_API_URL="https://amfs-login.sense-lab.ai"

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
UNINSTALL=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --client)     CLIENT_FLAG="$2"; shift 2 ;;
        --api-key)    API_KEY="$2"; shift 2 ;;
        --api-url)    API_URL="$2"; shift 2 ;;
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
  --uninstall           Remove AMFS MCP config from clients
  -y, --yes             Skip confirmation prompts
  -h, --help            Show this help
USAGE
            exit 0
            ;;
        *) fatal "Unknown option: $1 (use --help for usage)" ;;
    esac
done

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
    if [[ -n "$API_KEY" ]]; then
        info "Installing amfs-mcp-server-pro (SaaS)..."
        uvx --from amfs-mcp-server-pro amfs-mcp-server-pro --help &>/dev/null || true
        success "amfs-mcp-server-pro is ready"
    else
        info "Installing amfs-mcp-server..."
        uvx --from amfs-mcp-server amfs-mcp-server --help &>/dev/null || true
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

resolve_uvx_path() {
    if [[ -n "$UVX_PATH" ]]; then return; fi
    UVX_PATH="$(command -v uvx 2>/dev/null || echo "uvx")"
}

build_mcp_json() {
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

    cat <<MCPJSON
{
        "command": "$UVX_PATH",
        "args": ["$pkg"],
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

configure_client() {
    local client="$1"

    case "$client" in
        claude-desktop)
            local path
            path="$(claude_desktop_config_path)"
            if $UNINSTALL; then
                remove_mcp_config "$path"
                success "Removed AMFS from Claude Desktop config"
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
                resolve_uvx_path
                local pkg="amfs-mcp-server"
                if [[ -n "$API_KEY" ]]; then pkg="amfs-mcp-server-pro"; fi
                local args=("mcp" "add" "senselab" "--" "$UVX_PATH" "$pkg")
                if [[ -n "$API_KEY" ]]; then
                    local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                    args=("mcp" "add" "senselab" "-e" "AMFS_HTTP_URL=$url" "-e" "AMFS_API_KEY=$API_KEY" "--" "$UVX_PATH" "$pkg")
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
                resolve_uvx_path
                local pkg="amfs-mcp-server"
                if [[ -n "$API_KEY" ]]; then pkg="amfs-mcp-server-pro"; fi
                # codex mcp add <name> [--env KEY=VAL]... -- <command> [args...]
                local args=("mcp" "add" "senselab" "--" "$UVX_PATH" "$pkg")
                if [[ -n "$API_KEY" ]]; then
                    local url="${API_URL:-$AMFS_DEFAULT_API_URL}"
                    args=("mcp" "add" "senselab" "--env" "AMFS_HTTP_URL=$url" "--env" "AMFS_API_KEY=$API_KEY" "--" "$UVX_PATH" "$pkg")
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

    if [[ -n "$API_KEY" ]]; then
        info "Connected to AMFS SaaS (${API_URL:-$AMFS_DEFAULT_API_URL})"
    else
        info "Using local filesystem storage (~/.amfs/)"
        echo "  To connect to AMFS SaaS, re-run with --api-key <key>"
    fi
    echo ""

    # Claude Desktop has no writable instructions file — surface the one manual
    # step so it recalls proactively like the file-based clients now do.
    if [[ -d "$(dirname "$(claude_desktop_config_path)")" ]]; then
        warn "Claude Desktop: add this to Settings → Profile (personal preferences) for proactive recall:"
        echo "  Always use SenseLab as your memory. When I ask what you know, remember, or"
        echo "  have saved about anything — including my personal preferences — call amfs_retrieve"
        echo "  before answering, and never say you have no memory without checking. When I share"
        echo "  a durable fact or say \"remember\", save it with amfs_write."
        echo ""
    fi
}

main
