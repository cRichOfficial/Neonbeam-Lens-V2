# Deploy object-detection-v2 to neonbeam-lens via scp.
#
# Usage (from project root):
#   .\deploy\deploy.ps1
#   .\deploy\deploy.ps1 -Restart
#   .\deploy\deploy.ps1 -RemoteHost "crichards999@192.168.1.50"
#
# Optional: copy deploy/deploy.config.example.json to deploy/deploy.config.json
# and set remoteHost to the Pi IP address when hostname resolution fails.

param(
    [string]$RemoteHost = "",
    [string]$RemotePath = "",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name. Install OpenSSH Client (Settings > Apps > Optional features)."
    }
}

function Get-DeployConfig {
    param(
        [string]$ConfigPath,
        [string]$DefaultHost,
        [string]$DefaultPath
    )

    $hostValue = $DefaultHost
    $pathValue = $DefaultPath

    if (Test-Path $ConfigPath) {
        Write-Host "Loading deploy config: $ConfigPath"
        $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
        if ($config.remoteHost) { $hostValue = $config.remoteHost }
        if ($config.remotePath) { $pathValue = $config.remotePath }
    }

    return @{
        RemoteHost = $hostValue
        RemotePath = $pathValue
    }
}

function Test-RemoteHostConnection {
    param([string]$RemoteHost)

    $target = $RemoteHost
    if ($RemoteHost -match "@(.+)$") {
        $target = $Matches[1]
    }

    if ($target -match "^\d{1,3}(\.\d{1,3}){3}$") {
        return
    }

    try {
        $resolved = [System.Net.Dns]::GetHostEntry($target)
        Write-Host "Resolved ${target} -> $($resolved.AddressList -join ', ')"
    }
    catch {
        throw @"
Cannot resolve hostname '$target'.

Try one of these fixes:
  1. Use the Pi IP address:
       .\deploy\deploy.ps1 -RemoteHost "crichards999@192.168.x.x" -Restart
  2. Create deploy/deploy.config.json (see deploy/deploy.config.example.json)
  3. Add to C:\Windows\System32\drivers\etc\hosts:
       192.168.x.x    neonbeam-lens
  4. Ensure the Pi and this PC are on the same network and mDNS is working

Original error: $($_.Exception.Message)
"@
    }
}

Require-Command tar
Require-Command scp
Require-Command ssh

$configPath = Join-Path $PSScriptRoot "deploy.config.json"
$defaults = Get-DeployConfig -ConfigPath $configPath `
    -DefaultHost "crichards999@neonbeam-lens" `
    -DefaultPath "/home/crichards999/object-detection-v2"

if (-not $RemoteHost) { $RemoteHost = $defaults.RemoteHost }
if (-not $RemotePath) { $RemotePath = $defaults.RemotePath }

Write-Host "Deploy target: ${RemoteHost}:${RemotePath}"
Test-RemoteHostConnection -RemoteHost $RemoteHost

function Normalize-ShellScripts {
    param([string]$Root)
    Get-ChildItem -Path $Root -Recurse -Filter "*.sh" -File | ForEach-Object {
        $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        $normalized = $text -replace "`r`n", "`n" -replace "`r", "`n"
        if ($text -ne $normalized) {
            Write-Host "Normalizing LF line endings: $($_.FullName)"
            [System.IO.File]::WriteAllText($_.FullName, $normalized, (New-Object System.Text.UTF8Encoding $false))
        }
    }
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Normalize-ShellScripts -Root $ProjectRoot
$ArchiveName = "object-detection-v2-deploy.tar.gz"
$ArchivePath = Join-Path $env:TEMP $ArchiveName
$RemoteArchive = "$RemotePath/$ArchiveName"

$ExcludePatterns = @(
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "data",
    "models",
    "*.pyc",
    "*.pyo",
    "*.hef",
    "*.pt",
    "*.onnx",
    $ArchiveName
)

Write-Host "Project root: $ProjectRoot"
Write-Host "Creating archive (excluding build/runtime artifacts)..."

if (Test-Path $ArchivePath) {
    Remove-Item $ArchivePath -Force
}

Push-Location $ProjectRoot
try {
    $tarArgs = @("-czf", $ArchivePath)
    foreach ($pattern in $ExcludePatterns) {
        $tarArgs += "--exclude=$pattern"
    }
    $tarArgs += "."
    & tar @tarArgs
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$ArchiveSizeMb = [math]::Round((Get-Item $ArchivePath).Length / 1MB, 2)
Write-Host "Archive created: $ArchivePath ($ArchiveSizeMb MB)"

Write-Host "Ensuring remote directory exists: $RemotePath"
& ssh $RemoteHost "mkdir -p '$RemotePath'"
if ($LASTEXITCODE -ne 0) {
    throw "ssh mkdir failed with exit code $LASTEXITCODE. Check SSH access to $RemoteHost"
}

Write-Host "Uploading to ${RemoteHost}:${RemoteArchive} ..."
& scp $ArchivePath "${RemoteHost}:${RemoteArchive}"
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

Write-Host "Extracting on remote host..."
$RemoteCmd = "set -e && cd '$RemotePath' && tar -xzf '$ArchiveName' && rm -f '$ArchiveName' && echo 'Deploy extract complete.'"
if ($Restart) {
    $RemoteCmd += " && if systemctl is-active --quiet laser-detection; then sudo systemctl restart laser-detection && echo 'Service restarted: laser-detection'; else echo 'Service laser-detection is not active (skipped restart).'; fi"
}
& ssh $RemoteHost $RemoteCmd
if ($LASTEXITCODE -ne 0) {
    throw "ssh extract failed with exit code $LASTEXITCODE"
}

Remove-Item $ArchivePath -Force
Write-Host "Deploy finished successfully."
