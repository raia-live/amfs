# Tests for install-mcp.ps1
#
# Runs the installer against throwaway sandbox directories with stubbed uvx /
# claude / codex executables, so nothing touches a real MCP client config.
#
# Usage (from the repo root):
#   pwsh -NoProfile -File tests/install-mcp.Tests.ps1
#
# The installer is Windows-targeted but the logic under test (JSON merging,
# idempotency, instruction blocks, exit codes) is platform independent, so this
# suite also runs on Linux/macOS PowerShell in CI. On non-Windows the Windows
# path separators become literal characters in file names, which is harmless
# here because only file *contents* are asserted on.

param([string]$ScriptPath = (Join-Path $PSScriptRoot ".." | Join-Path -ChildPath "install-mcp.ps1"))

$ErrorActionPreference = "Stop"
$ScriptPath = (Resolve-Path $ScriptPath).Path

$script:pass = 0
$script:fail = 0
$script:failures = @()

function Check {
    param([string]$Name, $Condition, [string]$Detail = "")
    if ($Condition) {
        Write-Host "  PASS  $Name" -ForegroundColor Green
        $script:pass++
    } else {
        Write-Host "  FAIL  $Name $Detail" -ForegroundColor Red
        $script:fail++
        $script:failures += $Name
    }
}

function Write-Section { param([string]$Name) Write-Host "`n=== $Name ===" -ForegroundColor Cyan }

# ── Stub executables ────────────────────────────────────────────────────────

$stubBin = Join-Path ([System.IO.Path]::GetTempPath()) "amfs-test-bin"
New-Item -ItemType Directory -Force -Path $stubBin | Out-Null

function Set-Stub {
    param([string]$Name, [string]$Body)
    $path = Join-Path $stubBin $Name
    Set-Content -Path $path -Value $Body -NoNewline
    if (-not $IsWindows) { & chmod +x $path }
}

Set-Stub -Name "uvx" -Body "#!/bin/sh`necho 'uv 0.9.9'`nexit 0`n"
$env:PATH = $stubBin + [System.IO.Path]::PathSeparator + $env:PATH

# Mimics the real CLI: chatty on stdout, and `mcp remove` fails when there is
# nothing to remove (the state of every fresh machine).
function Set-ClaudeStub {
    param([int]$AddExit = 0)
    Set-Stub -Name "claude" -Body @"
#!/bin/sh
if [ "`$2" = "remove" ]; then
  echo "No MCP server named senselab found" >&2
  exit 1
fi
if [ "`$2" = "add" ]; then
  echo "Added stdio MCP server senselab to user config"
  exit $AddExit
fi
exit 0
"@
}

function Set-CodexStub {
    Set-Stub -Name "codex" -Body "#!/bin/sh`necho 'codex output'`nexit 0`n"
}

function New-Sandbox {
    $path = Join-Path ([System.IO.Path]::GetTempPath()) ("amfs-sandbox-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Force -Path (Join-Path $path ".cursor") | Out-Null
    $env:USERPROFILE = $path
    $env:APPDATA = Join-Path $path "AppData"
    $env:AMFS_API_KEY = ""
    $env:AMFS_HTTP_URL = ""
    return $path
}

function Get-CursorRaw {
    param([string]$Sandbox)
    $file = Get-ChildItem -Path (Join-Path $Sandbox ".cursor") -File -Force | Select-Object -First 1
    if (-not $file) { return $null }
    return (Get-Content $file.FullName -Raw)
}

function Invoke-Installer {
    # Named parameters must be splatted from a hashtable; splatting an array
    # binds the values positionally, so "-Client" would arrive as a value.
    # 6>&1 captures Write-Host (information stream) so output can be asserted on.
    param([hashtable]$Parameters)
    return (& $ScriptPath @Parameters 6>&1 2>&1 | Out-String)
}

# ── Configuration writing ───────────────────────────────────────────────────

Write-Section "SaaS install selects the pro package and injects credentials"
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "cursor"; ApiKey = "amfs_sk_test"; Yes = $true } | Out-Null
$config = Get-CursorRaw $sandbox | ConvertFrom-Json
$server = $config.mcpServers.senselab
Check "senselab server written" ($null -ne $server)
Check "uses pro package for SaaS" ($server.args -contains "amfs-mcp-server-pro")
Check "--refresh precedes the package" ($server.args[0] -eq "--refresh" -and $server.args[1] -eq "amfs-mcp-server-pro") "got $($server.args -join ' ')"
Check "API key injected" ($server.env.AMFS_API_KEY -eq "amfs_sk_test")
Check "default API url used" ($server.env.AMFS_HTTP_URL -eq "https://amfs-login.sense-lab.ai")
Check "command resolves to uvx" ($server.command -like "*uvx*") "got $($server.command)"

Write-Section "Local install selects the free package with an empty env"
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "cursor"; Yes = $true } | Out-Null
$raw = Get-CursorRaw $sandbox
Check "uses free package" ((($raw | ConvertFrom-Json).mcpServers.senselab.args) -contains "amfs-mcp-server")
Check "env serialized as an empty object" ($raw -match '"env":\s*\{\s*\}') "got $raw"

Write-Section "API key is read from the environment"
$sandbox = New-Sandbox
$env:AMFS_API_KEY = "amfs_sk_fromenv"
Invoke-Installer @{ Client = "cursor"; Yes = $true } | Out-Null
$server = (Get-CursorRaw $sandbox | ConvertFrom-Json).mcpServers.senselab
Check "env var key used" ($server.env.AMFS_API_KEY -eq "amfs_sk_fromenv")
Check "env var key selects pro package" ($server.args -contains "amfs-mcp-server-pro")
$env:AMFS_API_KEY = ""

Write-Section "Custom API url overrides the default"
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "cursor"; ApiKey = "k"; ApiUrl = "https://amfs.internal"; Yes = $true } | Out-Null
Check "custom url used" (((Get-CursorRaw $sandbox | ConvertFrom-Json).mcpServers.senselab.env.AMFS_HTTP_URL) -eq "https://amfs.internal")

# ── Merging into an existing config ─────────────────────────────────────────

Write-Section "Merging preserves unrelated servers and keys"
$sandbox = New-Sandbox
$existing = @'
{
    "someTopLevelSetting": true,
    "mcpServers": {
        "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": { "TOKEN": "abc" } },
        "postgres": { "command": "uvx", "args": ["postgres-mcp"] }
    }
}
'@
$cursorConfig = Join-Path (Join-Path $sandbox ".cursor") "mcp.json"
Set-Content -Path $cursorConfig -Value $existing -NoNewline
Invoke-Installer @{ Client = "cursor"; ApiKey = "amfs_sk_merge"; Yes = $true } | Out-Null
$raw = Get-Content $cursorConfig -Raw
$config = $raw | ConvertFrom-Json
Check "multi-element args preserved" ($config.mcpServers.github.args.Count -eq 2) "got $($config.mcpServers.github.args.Count)"
Check "nested env preserved" ($config.mcpServers.github.env.TOKEN -eq "abc")
Check "single-element args stays an array" ($config.mcpServers.postgres.args -is [array]) "collapsed to $($config.mcpServers.postgres.args.GetType().Name)"
Check "single-element args value intact" ($config.mcpServers.postgres.args[0] -eq "postgres-mcp")
Check "unrelated top-level key preserved" ($config.someTopLevelSetting -eq $true)
Check "senselab added alongside" ($config.mcpServers.senselab.env.AMFS_API_KEY -eq "amfs_sk_merge")

Write-Section "Re-running is idempotent"
Invoke-Installer @{ Client = "cursor"; ApiKey = "amfs_sk_merge2"; Yes = $true } | Out-Null
$config = Get-Content $cursorConfig -Raw | ConvertFrom-Json
Check "server count unchanged" ((@($config.mcpServers.PSObject.Properties).Count) -eq 3) "got $(@($config.mcpServers.PSObject.Properties).Count)"
Check "credentials updated in place" ($config.mcpServers.senselab.env.AMFS_API_KEY -eq "amfs_sk_merge2")
Check "neighbours still intact" ($config.mcpServers.github.env.TOKEN -eq "abc")

Write-Section "Uninstall removes only the senselab entry"
Invoke-Installer @{ Client = "cursor"; Uninstall = $true; Yes = $true } | Out-Null
$config = Get-Content $cursorConfig -Raw | ConvertFrom-Json
Check "senselab removed" ($null -eq $config.mcpServers.senselab)
Check "github survived" ($config.mcpServers.github.command -eq "npx")
Check "postgres survived" ($null -ne $config.mcpServers.postgres)
Check "top-level key survived" ($config.someTopLevelSetting -eq $true)

Write-Section "Malformed config is backed up rather than lost"
$sandbox = New-Sandbox
$cursorConfig = Join-Path (Join-Path $sandbox ".cursor") "mcp.json"
Set-Content -Path $cursorConfig -Value "{ not json ,,," -NoNewline
Invoke-Installer @{ Client = "cursor"; ApiKey = "amfs_sk_bad"; Yes = $true } | Out-Null
Check "backup written" (Test-Path "$cursorConfig.amfs-backup")
Check "config rebuilt" (((Get-Content $cursorConfig -Raw | ConvertFrom-Json).mcpServers.senselab.env.AMFS_API_KEY) -eq "amfs_sk_bad")

Write-Section "Config is written without a BOM"
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "cursor"; ApiKey = "k"; Yes = $true } | Out-Null
$file = Get-ChildItem -Path (Join-Path $sandbox ".cursor") -File -Force | Select-Object -First 1
$bytes = [System.IO.File]::ReadAllBytes($file.FullName)
Check "no UTF-8 BOM" (-not ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF))

# ── Instruction blocks ──────────────────────────────────────────────────────

Write-Section "Instruction block is idempotent and preserves user content"
Set-CodexStub
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "codex"; ApiKey = "k"; Yes = $true } | Out-Null
$agents = Get-ChildItem -Path $sandbox -Recurse -File -Force | Where-Object { $_.Name -like "*AGENTS.md" } | Select-Object -First 1
Check "AGENTS.md created" ($null -ne $agents)
if ($agents) {
    $marker = "<!-- >>> senselab-memory >>> -->"
    $content = Get-Content $agents.FullName -Raw
    Check "exactly one marker" (([regex]::Matches($content, [regex]::Escape($marker))).Count -eq 1)

    Set-Content -Path $agents.FullName -Value ("# My own notes`nkeep this line`n`n" + $content) -NoNewline
    Invoke-Installer @{ Client = "codex"; ApiKey = "k"; Yes = $true } | Out-Null
    $content = Get-Content $agents.FullName -Raw
    Check "still one marker after re-run" (([regex]::Matches($content, [regex]::Escape($marker))).Count -eq 1)
    Check "user content preserved" ($content -match "keep this line")

    Invoke-Installer @{ Client = "codex"; Uninstall = $true; Yes = $true } | Out-Null
    $content = Get-Content $agents.FullName -Raw
    Check "block removed on uninstall" (-not ($content -match "senselab-memory"))
    Check "user content survived uninstall" ($content -match "keep this line")
}

Write-Section "Claude Code skill is written with valid frontmatter"
Set-ClaudeStub
$sandbox = New-Sandbox
Invoke-Installer @{ Client = "claude-code"; ApiKey = "k"; Yes = $true } | Out-Null
$skill = Get-ChildItem -Path $sandbox -Recurse -File -Force | Where-Object { $_.Name -like "*SKILL.md" } | Select-Object -First 1
Check "SKILL.md created" ($null -ne $skill)
if ($skill) {
    $content = Get-Content $skill.FullName -Raw
    Check "frontmatter starts the file" ($content.StartsWith("---"))
    Check "declares its name" ($content -match "name: senselab-memory")
    Check "contains the recall-first rule" ($content -match "RECALL-FIRST")
}

# ── CLI handling and exit codes ─────────────────────────────────────────────

Write-Section "Fresh machine: a failing 'mcp remove' must not abort the install"
Set-ClaudeStub -AddExit 0
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Client = "claude-code"; ApiKey = "k"; Yes = $true }
Check "install completed" ($output -match "Done!")
Check "reported as configured" ($output -match "Configured Claude Code")
Check "exit code 0" ($LASTEXITCODE -eq 0) "got $LASTEXITCODE"

Write-Section "A failing 'mcp add' must be reported, never masked as success"
Set-ClaudeStub -AddExit 1
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Client = "claude-code"; ApiKey = "k"; Yes = $true }
Check "failure surfaced" ($output -match "mcp add failed")
Check "success not claimed" (-not ($output -match "Configured Claude Code"))

Write-Section "CLI stdout must not leak into the success check"
Set-ClaudeStub -AddExit 0
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Client = "claude-code"; ApiKey = "k"; Yes = $true }
Check "CLI chatter not echoed" (-not ($output -match "Added stdio MCP server"))
Check "still reported as configured" ($output -match "Configured Claude Code")

Write-Section "Missing CLI is skipped rather than fatal"
Remove-Item (Join-Path $stubBin "claude") -Force
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Client = "claude-code"; ApiKey = "k"; Yes = $true }
Check "warns CLI is absent" ($output -match "not found on PATH")
Check "install still completes" ($output -match "Done!")
Set-ClaudeStub

Write-Section "Unknown client fails loudly"
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Client = "notaclient"; Yes = $true }
Check "names the bad client" ($output -match "Unknown client")
Check "exit code non-zero" ($LASTEXITCODE -ne 0) "got $LASTEXITCODE"
Check "success not claimed" (-not ($output -match "Done!"))

Write-Section "-Help exits cleanly without writing anything"
$sandbox = New-Sandbox
$output = Invoke-Installer @{ Help = $true }
Check "exit code 0" ($LASTEXITCODE -eq 0) "got $LASTEXITCODE"
Check "shows usage" ($output -match "Usage:")
Check "wrote no config" (-not (Get-ChildItem -Path (Join-Path $sandbox ".cursor") -File -Force))

# ── Portability guards ──────────────────────────────────────────────────────

Write-Section "Script avoids PowerShell 7-only syntax (must run on Windows 5.1)"
# Strip comment-only lines so prose mentioning a construct isn't a false positive.
$source = ((Get-Content $ScriptPath) | Where-Object { $_.TrimStart() -notmatch '^#' }) -join [Environment]::NewLine
Check "no null-coalescing operator" (-not ($source -match '\?\?'))
Check "no ternary operator" (-not ($source -match '\)\s*\?\s*[^\s]+\s*:\s'))
Check "no -AsHashtable usage" (-not ($source -match 'ConvertFrom-Json[^\r\n]*-AsHashtable'))
Check "no utf8NoBOM encoding name" (-not ($source -match 'utf8NoBOM'))
Check "no && or || chaining" (-not ($source -match '(\s&&\s|\s\|\|\s)'))
Check "no #Requires (breaks irm | iex)" (-not ($source -match '#Requires'))

Write-Host "`n----------------------------------------"
if ($script:fail -gt 0) {
    Write-Host "FAILED: $($script:fail)   PASSED: $($script:pass)" -ForegroundColor Red
    $script:failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "ALL PASSED: $($script:pass) assertions" -ForegroundColor Green
