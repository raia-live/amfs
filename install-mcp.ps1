# AMFS / SenseLab MCP Installer (Windows)
#
# One-line install:
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.ps1 | iex"
#
# With a SaaS API key (simplest form — set the env var first, the script picks it up):
#   $env:AMFS_API_KEY="amfs_sk_your_key"
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.ps1 | iex"
#
# Or pass parameters explicitly (needed because `irm | iex` cannot forward args):
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.ps1))) -ApiKey amfs_sk_your_key
#
# This is the Windows counterpart of install-mcp.sh and is kept behaviourally in
# sync with it: same "senselab" server name, same client list, same
# marker-delimited instruction blocks.

param(
    [string]$Client = "",
    [string]$ApiKey = "",
    [string]$ApiUrl = "",
    [switch]$Uninstall,
    [switch]$Yes,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$AmfsDefaultApiUrl = "https://amfs-login.sense-lab.ai"
$AmfsServerName    = "senselab"

# Env-var fallbacks so the plain `irm | iex` form can still reach SaaS.
if ([string]::IsNullOrWhiteSpace($ApiKey)) { $ApiKey = "$env:AMFS_API_KEY" }
if ([string]::IsNullOrWhiteSpace($ApiUrl)) { $ApiUrl = "$env:AMFS_HTTP_URL" }
if ([string]::IsNullOrWhiteSpace($ApiUrl)) { $ApiUrl = $AmfsDefaultApiUrl }

# ── Output helpers ───────────────────────────────────────────────────────────

function Write-Info    { param([string]$Message) Write-Host "==> " -ForegroundColor Blue   -NoNewline; Write-Host $Message }
function Write-Ok      { param([string]$Message) Write-Host "==> " -ForegroundColor Green  -NoNewline; Write-Host $Message }
function Write-Warn    { param([string]$Message) Write-Host "==> " -ForegroundColor Yellow -NoNewline; Write-Host $Message }
function Write-Err     { param([string]$Message) Write-Host "==> " -ForegroundColor Red    -NoNewline; Write-Host $Message }
function Stop-WithError { param([string]$Message) Write-Err $Message; exit 1 }

if ($Help) {
    Write-Host @"
AMFS / SenseLab MCP Installer (Windows)

Usage:
  powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.ps1 | iex"
  .\install-mcp.ps1 [OPTIONS]

Options:
  -Client <name|all>   Configure a specific client: claude-desktop, cursor,
                       claude-code, codex, windsurf, vscode, or "all"
  -ApiKey <key>        Connect to AMFS SaaS with this API key
                       (or set `$env:AMFS_API_KEY)
  -ApiUrl <url>        SaaS API URL (default: $AmfsDefaultApiUrl)
  -Uninstall           Remove AMFS MCP config from clients
  -Yes                 Skip confirmation prompts
  -Help                Show this help
"@
    exit 0
}

if ($PSVersionTable.PSVersion.Major -lt 5) {
    Stop-WithError "PowerShell 5.1 or newer is required (found $($PSVersionTable.PSVersion))."
}

$HasApiKey = -not [string]::IsNullOrWhiteSpace($ApiKey)
$AmfsPackage = if ($HasApiKey) { "amfs-mcp-server-pro" } else { "amfs-mcp-server" }

$KnownClients = @("claude-desktop", "cursor", "claude-code", "codex", "windsurf", "vscode")
if (-not [string]::IsNullOrWhiteSpace($Client) -and $Client -ne "all" -and $KnownClients -notcontains $Client) {
    Stop-WithError "Unknown client '$Client'. Valid values: $($KnownClients -join ', '), all"
}

# ── Step 1: ensure uv is installed ───────────────────────────────────────────

function Update-SessionPath {
    # uv's installer writes to the *user* PATH in the registry; the running
    # process won't see it without this refresh.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machine, $user, "$env:USERPROFILE\.local\bin") | Where-Object { $_ }) -join ";"
}

function Get-UvxPath {
    $cmd = Get-Command uvx -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    # Fall back to the default install location in case PATH is still stale.
    $fallback = Join-Path $env:USERPROFILE ".local\bin\uvx.exe"
    if (Test-Path $fallback) { return $fallback }

    return $null
}

function Install-UvIfMissing {
    if (Get-UvxPath) {
        Write-Ok "uv is already installed ($(& (Get-UvxPath) --version 2>$null))"
        return
    }

    Write-Info "uv not found - installing..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Stop-WithError @"
Failed to install uv automatically: $($_.Exception.Message)

Install it manually with one of these, then re-run this script:
  winget install --id=astral-sh.uv -e
  scoop install uv
"@
    }

    Update-SessionPath

    if (-not (Get-UvxPath)) {
        Stop-WithError "uv was installed but uvx could not be found on PATH. Open a new PowerShell window and run this script again."
    }

    Write-Ok "uv installed successfully ($(& (Get-UvxPath) --version 2>$null))"
}

# ── Step 2: pre-warm the amfs-mcp-server package ─────────────────────────────

function Install-AmfsPackage {
    $uvx = Get-UvxPath
    Write-Info "Installing $AmfsPackage..."
    try {
        # --refresh so the warm-up pulls the latest published build, matching the
        # --refresh the generated client config uses at launch.
        & $uvx --refresh --from $AmfsPackage $AmfsPackage --help *> $null
    } catch {
        # Non-fatal: the client will fetch it on first launch. Warn only.
        Write-Warn "Could not pre-install $AmfsPackage - it will be fetched on first use."
        return
    }
    Write-Ok "$AmfsPackage is ready"
}

# ── Client config paths ──────────────────────────────────────────────────────

function Get-ClaudeDesktopConfigPath { Join-Path $env:APPDATA "Claude\claude_desktop_config.json" }
function Get-CursorConfigPath        { Join-Path $env:USERPROFILE ".cursor\mcp.json" }
function Get-WindsurfConfigPath      { Join-Path $env:USERPROFILE ".windsurf\mcp.json" }
function Get-VSCodeConfigPath        { Join-Path $env:USERPROFILE ".vscode\mcp.json" }

# ── MCP server block ─────────────────────────────────────────────────────────

function Get-McpServerBlock {
    $uvx = Get-UvxPath
    if (-not $uvx) { $uvx = "uvx" }

    # --refresh forces uvx to re-resolve from PyPI on each launch instead of
    # reusing a stale cached environment. Without it, a user who ever ran an
    # older build keeps getting it forever - even after we publish a fix - so
    # retrieval can stay broken across releases. The small per-launch re-resolve
    # cost is worth never serving a stale MCP server.
    $block = [ordered]@{
        command = $uvx
        args    = @("--refresh", $AmfsPackage)
    }

    if ($HasApiKey) {
        $block["env"] = [ordered]@{
            AMFS_HTTP_URL = $ApiUrl
            AMFS_API_KEY  = $ApiKey
        }
    } else {
        $block["env"] = [ordered]@{}
    }

    return $block
}

# ── JSON merge ───────────────────────────────────────────────────────────────

function ConvertTo-OrderedHashtable {
    # ConvertFrom-Json returns PSCustomObjects and PS 5.1 has no -AsHashtable,
    # so convert manually to keep existing config keys editable and ordered.
    param($InputObject)

    if ($null -eq $InputObject) { return $null }

    if ($InputObject -is [System.Management.Automation.PSCustomObject]) {
        $result = [ordered]@{}
        foreach ($property in $InputObject.PSObject.Properties) {
            $result[$property.Name] = ConvertTo-OrderedHashtable $property.Value
        }
        return $result
    }

    if ($InputObject -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $InputObject.Keys) {
            $result[$key] = ConvertTo-OrderedHashtable $InputObject[$key]
        }
        return $result
    }

    if ($InputObject -is [object[]]) {
        $items = @(foreach ($item in $InputObject) { ConvertTo-OrderedHashtable $item })
        # The comma stops PowerShell unrolling a one-element array back into a
        # scalar on return, which would rewrite a neighbouring server's
        # "args": ["x"] as "args": "x" and break it.
        return ,$items
    }

    return $InputObject
}

function Read-JsonFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return [ordered]@{} }

    $raw = Get-Content -Path $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) { return [ordered]@{} }

    $parsed = $null
    try {
        $parsed = ConvertTo-OrderedHashtable (ConvertFrom-Json $raw)
    } catch {
        $parsed = $null
    }

    if ($parsed -isnot [System.Collections.IDictionary]) {
        Write-Warn "$Path is not a valid JSON object - backing it up to $Path.amfs-backup and starting fresh."
        Copy-Item -Path $Path -Destination "$Path.amfs-backup" -Force
        return [ordered]@{}
    }

    return $parsed
}

function Write-JsonFile {
    param([string]$Path, $Data)

    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    $json = ConvertTo-Json $Data -Depth 20
    # Write UTF-8 without BOM: some MCP clients reject a BOM'd config file.
    [System.IO.File]::WriteAllText($Path, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
}

function Set-McpConfig {
    param([string]$Path)

    $config = Read-JsonFile -Path $Path
    if (-not $config.Contains("mcpServers") -or $config["mcpServers"] -isnot [System.Collections.IDictionary]) {
        $config["mcpServers"] = [ordered]@{}
    }
    $config["mcpServers"][$AmfsServerName] = Get-McpServerBlock
    Write-JsonFile -Path $Path -Data $config
}

function Remove-McpConfig {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return }

    $config = Read-JsonFile -Path $Path
    if ($config.Contains("mcpServers") -and
        $config["mcpServers"] -is [System.Collections.IDictionary] -and
        $config["mcpServers"].Contains($AmfsServerName)) {
        $config["mcpServers"].Remove($AmfsServerName)
        Write-JsonFile -Path $Path -Data $config
    }
}

# ── SenseLab memory instructions ─────────────────────────────────────────────
#
# Mirrors install-mcp.sh: a recall-first guide is written into the clients that
# auto-load on-disk instructions (Claude Code skill + ~/.claude/CLAUDE.md, and
# ~/.codex/AGENTS.md). Marker-delimited so re-runs replace rather than duplicate.

$SenseLabMarkerBegin = "<!-- >>> senselab-memory >>> -->"
$SenseLabMarkerEnd   = "<!-- <<< senselab-memory <<< -->"

$SenseLabInstructions = @'
## SenseLab memory (recall-first)

You are connected to SenseLab (AMFS), a persistent memory shared across all your
tools and sessions. Use it proactively:

- **Start of session:** call `amfs_set_identity`, then `amfs_briefing(entity_path="repo/module")` for compiled context before working in a codebase.
- **RECALL-FIRST (do not skip):** whenever the user asks what you know / remember / have saved about ANYTHING - including personal facts and preferences ("what food do I like?") - call `amfs_retrieve(query="<the user's words>")` BEFORE answering. `entity_path`/`key` are optional; it searches by meaning across everything you can see. Never tell the user you have no memory of something without running `amfs_retrieve` first.
- **Remember things:** when the user shares a durable fact, preference, or decision, or says "remember...", call `amfs_write(entity_path, key, value)`.
- `amfs_read`/`amfs_recall` need an EXACT key - a miss there means "try `amfs_retrieve`", NOT "nothing is stored".
'@

function Remove-SenseLabBlockText {
    param([string]$Content)

    if ($Content.Contains($SenseLabMarkerBegin) -and $Content.Contains($SenseLabMarkerEnd)) {
        $before = $Content.Substring(0, $Content.IndexOf($SenseLabMarkerBegin))
        $endIndex = $Content.IndexOf($SenseLabMarkerEnd) + $SenseLabMarkerEnd.Length
        $after = $Content.Substring($endIndex)
        return ($before.TrimEnd("`r", "`n") + "`n`n" + $after.TrimStart("`r", "`n")).Trim("`r", "`n")
    }
    return $Content
}

function Set-SenseLabBlock {
    param([string]$Path)

    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    $content = ""
    if (Test-Path $Path) {
        $content = Get-Content -Path $Path -Raw -Encoding UTF8
        if ($null -eq $content) { $content = "" }
        $content = Remove-SenseLabBlockText -Content $content
    }

    $block = $SenseLabMarkerBegin + "`n" + $SenseLabInstructions.TrimEnd() + "`n" + $SenseLabMarkerEnd
    $prefix = if ($content.Trim()) { $content.TrimEnd("`r", "`n") + "`n`n" } else { "" }

    [System.IO.File]::WriteAllText($Path, $prefix + $block + "`n", (New-Object System.Text.UTF8Encoding($false)))
}

function Remove-SenseLabBlock {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return }

    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    if ($null -eq $content) { return }

    $cleaned = Remove-SenseLabBlockText -Content $content
    if ($cleaned -ne $content) {
        $suffix = if ($cleaned) { "`n" } else { "" }
        [System.IO.File]::WriteAllText($Path, $cleaned + $suffix, (New-Object System.Text.UTF8Encoding($false)))
    }
}

function Install-ClaudeCodeSkill {
    $skillDirectory = Join-Path $env:USERPROFILE ".claude\skills\senselab-memory"
    if (-not (Test-Path $skillDirectory)) {
        New-Item -ItemType Directory -Force -Path $skillDirectory | Out-Null
    }

    $frontmatter = @'
---
name: senselab-memory
description: Use SenseLab (AMFS) as persistent memory. When the user asks what you know/remember/have saved about anything (including personal preferences), call amfs_retrieve BEFORE answering; save durable facts with amfs_write.
---
'@

    $skillPath = Join-Path $skillDirectory "SKILL.md"
    $body = $frontmatter + "`n`n" + $SenseLabInstructions.TrimEnd() + "`n"
    [System.IO.File]::WriteAllText($skillPath, $body, (New-Object System.Text.UTF8Encoding($false)))
}

function Remove-ClaudeCodeSkill {
    $skillDirectory = Join-Path $env:USERPROFILE ".claude\skills\senselab-memory"
    if (Test-Path $skillDirectory) {
        Remove-Item -Path $skillDirectory -Recurse -Force
    }
}

# ── CLI client helpers (Claude Code / Codex) ─────────────────────────────────

function Invoke-Cli {
    # `claude` and `codex` are usually .cmd shims on Windows, which need to be
    # invoked through cmd.exe rather than spawned directly.
    param([string]$Exe, [string[]]$CliArgs)

    $command = Get-Command $Exe -ErrorAction SilentlyContinue
    if (-not $command) { return $false }

    $target = $command.Source
    if ($target -match '\.(cmd|bat)$') {
        & cmd.exe /c $target @CliArgs
    } else {
        & $target @CliArgs
    }
    return ($LASTEXITCODE -eq 0)
}

function Test-CliExists {
    param([string]$Exe)
    return [bool](Get-Command $Exe -ErrorAction SilentlyContinue)
}

# ── Client detection ─────────────────────────────────────────────────────────

function Get-DetectedClients {
    $detected = @()

    if (Test-Path (Split-Path -Parent (Get-ClaudeDesktopConfigPath))) { $detected += "claude-desktop" }
    if (Test-Path (Join-Path $env:USERPROFILE ".cursor"))             { $detected += "cursor" }
    if (Test-CliExists "claude")                                      { $detected += "claude-code" }
    if (Test-CliExists "codex")                                       { $detected += "codex" }
    if (Test-Path (Split-Path -Parent (Get-WindsurfConfigPath)))       { $detected += "windsurf" }
    if (Test-Path (Join-Path $env:USERPROFILE ".vscode"))             { $detected += "vscode" }

    return $detected
}

# ── Configure a single client ────────────────────────────────────────────────

function Set-ClientConfig {
    param([string]$Name)

    switch ($Name) {
        "claude-desktop" {
            $path = Get-ClaudeDesktopConfigPath
            if ($Uninstall) { Remove-McpConfig $path; Write-Ok "Removed AMFS from Claude Desktop config" }
            else            { Set-McpConfig $path;    Write-Ok "Configured Claude Desktop ($path)" }
        }
        "cursor" {
            $path = Get-CursorConfigPath
            if ($Uninstall) { Remove-McpConfig $path; Write-Ok "Removed AMFS from Cursor config" }
            else            { Set-McpConfig $path;    Write-Ok "Configured Cursor ($path)" }
        }
        "claude-code" {
            if ($Uninstall) {
                if (Test-CliExists "claude") {
                    Invoke-Cli "claude" @("mcp", "remove", $AmfsServerName, "--scope", "user") | Out-Null
                    Write-Ok "Removed AMFS from Claude Code"
                }
                Remove-ClaudeCodeSkill
                Remove-SenseLabBlock (Join-Path $env:USERPROFILE ".claude\CLAUDE.md")
                break
            }

            if (-not (Test-CliExists "claude")) {
                Write-Warn "Claude Code CLI (claude) not found on PATH - skipping"
                break
            }

            $uvx = Get-UvxPath
            if (-not $uvx) { $uvx = "uvx" }

            # Replace any existing entry so re-runs stay idempotent.
            Invoke-Cli "claude" @("mcp", "remove", $AmfsServerName, "--scope", "user") | Out-Null

            $cliArgs = @("mcp", "add", $AmfsServerName, "--scope", "user")
            if ($HasApiKey) {
                $cliArgs += @("-e", "AMFS_HTTP_URL=$ApiUrl", "-e", "AMFS_API_KEY=$ApiKey")
            }
            # --refresh is a uvx flag, so it goes before the package name.
            $cliArgs += @("--", $uvx, "--refresh", $AmfsPackage)

            if (Invoke-Cli "claude" $cliArgs) {
                Write-Ok "Configured Claude Code (user scope)"
            } else {
                Write-Warn "claude mcp add failed - check 'claude mcp list' and retry"
            }

            Install-ClaudeCodeSkill
            Set-SenseLabBlock (Join-Path $env:USERPROFILE ".claude\CLAUDE.md")
            Write-Ok "Installed SenseLab recall-first memory guide (skill + ~\.claude\CLAUDE.md)"
        }
        "codex" {
            if ($Uninstall) {
                if (Test-CliExists "codex") {
                    Invoke-Cli "codex" @("mcp", "remove", $AmfsServerName) | Out-Null
                    Write-Ok "Removed AMFS from Codex"
                }
                Remove-SenseLabBlock (Join-Path $env:USERPROFILE ".codex\AGENTS.md")
                break
            }

            if (-not (Test-CliExists "codex")) {
                Write-Warn "Codex CLI (codex) not found on PATH - skipping"
                break
            }

            $uvx = Get-UvxPath
            if (-not $uvx) { $uvx = "uvx" }

            $cliArgs = @("mcp", "add", $AmfsServerName)
            if ($HasApiKey) {
                $cliArgs += @("--env", "AMFS_HTTP_URL=$ApiUrl", "--env", "AMFS_API_KEY=$ApiKey")
            }
            # --refresh is a uvx flag, so it goes before the package name.
            $cliArgs += @("--", $uvx, "--refresh", $AmfsPackage)

            Invoke-Cli "codex" @("mcp", "remove", $AmfsServerName) | Out-Null
            if (Invoke-Cli "codex" $cliArgs) {
                Write-Ok "Configured Codex"
            } else {
                Write-Warn "codex mcp add failed - check 'codex mcp list' and retry"
            }

            Set-SenseLabBlock (Join-Path $env:USERPROFILE ".codex\AGENTS.md")
            Write-Ok "Installed SenseLab recall-first memory guide (~\.codex\AGENTS.md)"
        }
        "windsurf" {
            $path = Get-WindsurfConfigPath
            if ($Uninstall) { Remove-McpConfig $path; Write-Ok "Removed AMFS from Windsurf config" }
            else            { Set-McpConfig $path;    Write-Ok "Configured Windsurf ($path)" }
        }
        "vscode" {
            $path = Get-VSCodeConfigPath
            if ($Uninstall) { Remove-McpConfig $path; Write-Ok "Removed AMFS from VS Code config" }
            else            { Set-McpConfig $path;    Write-Ok "Configured VS Code ($path)" }
        }
        default {
            Write-Err "Unknown client: $Name"
        }
    }
}

function Show-ManualSnippet {
    Write-Host ""
    Write-Host "Add this to your MCP client config manually:"
    Write-Host ""
    Write-Host (ConvertTo-Json ([ordered]@{ mcpServers = [ordered]@{ $AmfsServerName = (Get-McpServerBlock) } }) -Depth 20)
    Write-Host ""
}

# ── Interactive selection ────────────────────────────────────────────────────

function Read-Answer {
    param([string]$Prompt)
    try   { return (Read-Host -Prompt $Prompt) }
    catch { return "" }   # Non-interactive host: fall through to the default.
}

function Invoke-ClientSelection {
    $detected = @(Get-DetectedClients)

    if ($detected.Count -eq 0) {
        Write-Warn "No supported MCP clients detected."
        Show-ManualSnippet
        return
    }

    Write-Host ""
    Write-Host "Detected MCP clients:" -ForegroundColor White
    Write-Host ""
    for ($i = 0; $i -lt $detected.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $detected[$i]) -ForegroundColor Green
    }
    Write-Host ""

    if ($Yes) {
        Write-Info "Configuring all detected clients (-Yes)"
        foreach ($name in $detected) { Set-ClientConfig $name }
        return
    }

    $action = if ($Uninstall) { "Remove AMFS from" } else { "Configure" }
    $answer = (Read-Answer "$action all detected clients? [Y/n/list]").Trim().ToLower()

    switch ($answer) {
        { $_ -in @("", "y", "yes") } {
            foreach ($name in $detected) { Set-ClientConfig $name }
        }
        { $_ -in @("n", "no") } {
            Write-Info "Skipped client configuration."
            Show-ManualSnippet
        }
        default {
            $selections = (Read-Answer "Enter client numbers separated by spaces (e.g. 1 3)").Split(" ")
            foreach ($selection in $selections) {
                $index = 0
                if ([int]::TryParse($selection.Trim(), [ref]$index) -and $index -ge 1 -and $index -le $detected.Count) {
                    Set-ClientConfig $detected[$index - 1]
                } elseif ($selection.Trim()) {
                    Write-Warn "Invalid selection: $selection"
                }
            }
        }
    }
}

function Invoke-RequestedClients {
    if ([string]::IsNullOrWhiteSpace($Client)) {
        Invoke-ClientSelection
        return
    }

    if ($Client -eq "all") {
        foreach ($name in @(Get-DetectedClients)) { Set-ClientConfig $name }
    } else {
        Set-ClientConfig $Client
    }
}

# ── Main ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "AMFS MCP Installer" -ForegroundColor White
Write-Host "------------------"
Write-Host ""

if ($Uninstall) {
    Write-Info "Uninstall mode"
    Invoke-RequestedClients
    Write-Host ""
    Write-Ok "AMFS MCP config removed. Restart your IDE to apply."
    exit 0
}

Install-UvIfMissing
Write-Host ""

Install-AmfsPackage
Write-Host ""

Invoke-RequestedClients

Write-Host ""
Write-Host "------------------"
Write-Ok "Done! Restart your IDE/app to connect to AMFS."
Write-Host ""

if ($HasApiKey) {
    Write-Info "Connected to AMFS SaaS ($ApiUrl)"
} else {
    Write-Info "Using local filesystem storage (~\.amfs\)"
    Write-Host "  To connect to AMFS SaaS, re-run with -ApiKey <key>"
}
Write-Host ""

# Claude Desktop has no writable instructions file, so surface the one manual
# step that makes it recall proactively like the file-based clients do.
if (Test-Path (Split-Path -Parent (Get-ClaudeDesktopConfigPath))) {
    Write-Warn "Claude Desktop: paste this into Settings -> General -> `"Instructions for Claude`" for proactive recall:"
    Write-Host @'
  SenseLab is my personal + work memory, connected as the "senselab" MCP connector
  (tools start with amfs_). It stores everything I ask it to remember - personal facts,
  preferences, people, plans - not just code. Whenever I ask what I like, prefer, know,
  remember, or have saved about anything, you MUST call amfs_retrieve with my question
  first, then answer from what it returns. Never answer from your own memory, and never
  say something "hasn't come up" or that SenseLab "isn't for that" until you have called
  amfs_retrieve. When I say "remember..." or share a durable fact, call amfs_write.
'@
    Write-Host ""
}
