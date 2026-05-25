# Deploy object-detection-v2 to neonbeam-lens.richwerks.local via scp.
#
# Usage (from project root):
#   .\deploy\deploy.ps1
#   .\deploy\deploy.ps1 -Restart
#   .\deploy\deploy.ps1 -RemoteHost "crichards999@neonbeam-lens.richwerks.local"
#
# Optional: copy deploy/deploy.config.example.json to deploy/deploy.config.json
# and tune SSH settings (timeouts, retries, preferIpv4).

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

function Get-RemoteTargetHost {
    param([string]$RemoteHost)

    if ($RemoteHost -match "@(.+)$") {
        return $Matches[1]
    }
    return $RemoteHost
}

function Get-DeployConfig {
    param(
        [string]$ConfigPath,
        [string]$DefaultHost,
        [string]$DefaultPath
    )

    $config = @{
        RemoteHost = $DefaultHost
        RemotePath = $DefaultPath
        SshConnectTimeout = 15
        SshRetryCount = 3
        SshRetryDelaySeconds = 5
        PreferIpv4 = $false
        SshClient = "auto"
    }

    if (Test-Path $ConfigPath) {
        Write-Host "Loading deploy config: $ConfigPath"
        $fileConfig = Get-Content $ConfigPath -Raw | ConvertFrom-Json
        if ($fileConfig.remoteHost) { $config.RemoteHost = $fileConfig.remoteHost }
        if ($fileConfig.remotePath) { $config.RemotePath = $fileConfig.remotePath }
        if ($null -ne $fileConfig.sshConnectTimeout) { $config.SshConnectTimeout = [int]$fileConfig.sshConnectTimeout }
        if ($null -ne $fileConfig.sshRetryCount) { $config.SshRetryCount = [int]$fileConfig.sshRetryCount }
        if ($null -ne $fileConfig.sshRetryDelaySeconds) { $config.SshRetryDelaySeconds = [int]$fileConfig.sshRetryDelaySeconds }
        if ($null -ne $fileConfig.preferIpv4) { $config.PreferIpv4 = [bool]$fileConfig.preferIpv4 }
        if ($fileConfig.sshClient) { $config.SshClient = [string]$fileConfig.sshClient }
    }

    return $config
}

function Resolve-SshClients {
    param([hashtable]$DeployConfig)

    $windowsSsh = (Get-Command ssh -ErrorAction SilentlyContinue).Source
    $windowsScp = (Get-Command scp -ErrorAction SilentlyContinue).Source
    $gitSsh = "C:\Program Files\Git\usr\bin\ssh.exe"
    $gitScp = "C:\Program Files\Git\usr\bin\scp.exe"

    $choice = $DeployConfig.SshClient.ToLowerInvariant()
    if ($choice -eq "git") {
        if (-not (Test-Path $gitSsh) -or -not (Test-Path $gitScp)) {
            throw "sshClient is set to 'git' but Git for Windows ssh/scp was not found at $gitSsh"
        }
        Write-Host "Using Git for Windows SSH: $gitSsh"
        return @{ Ssh = $gitSsh; Scp = $gitScp }
    }

    if ($choice -eq "windows") {
        if (-not $windowsSsh -or -not $windowsScp) {
            throw "sshClient is set to 'windows' but OpenSSH Client (ssh/scp) is not installed."
        }
        Write-Host "Using Windows OpenSSH: $windowsSsh"
        return @{ Ssh = $windowsSsh; Scp = $windowsScp }
    }

    if ((Test-Path $gitSsh) -and (Test-Path $gitScp)) {
        Write-Host "Using Git for Windows SSH (auto): $gitSsh"
        return @{ Ssh = $gitSsh; Scp = $gitScp }
    }

    if (-not $windowsSsh -or -not $windowsScp) {
        throw "Required command not found: ssh/scp. Install OpenSSH Client or Git for Windows."
    }

    Write-Host "Using Windows OpenSSH (auto): $windowsSsh"
    return @{ Ssh = $windowsSsh; Scp = $windowsScp }
}

function ConvertTo-GitUnixPath {
    param([string]$Path)

    if ($Path -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/$drive/$rest"
    }

    return $Path -replace '\\', '/'
}

function Get-NormalizedExitCode {
    param($ExitCode)

    if ($null -eq $ExitCode -or "$ExitCode" -eq "") {
        return 255
    }

    return [int]$ExitCode
}

function Invoke-ExternalCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $previousEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $FilePath @ArgumentList 2>&1
        $exitCode = Get-NormalizedExitCode -ExitCode $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousEap
    }

    return @{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object { "$_" })
    }
}

function Test-ScpCommandSucceeded {
    param(
        [int]$ExitCode,
        [string[]]$Output
    )

    if (-not (Test-SshCommandSucceeded -ExitCode $ExitCode -Output $Output)) {
        return $false
    }

    $combined = ($Output | ForEach-Object { "$_" }) -join "`n"
    if ($combined -match "No such file or directory|failed to upload|dest open") {
        return $false
    }

    return $true
}

function Get-SshBaseArgs {
    param([hashtable]$DeployConfig)

    $args = @(
        "-o", "ConnectTimeout=$($DeployConfig.SshConnectTimeout)",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3"
    )
    if ($DeployConfig.PreferIpv4) {
        $args += "-4"
    }
    return $args
}

function Test-SshCommandSucceeded {
    param(
        [int]$ExitCode,
        [string[]]$Output,
        [string]$ExpectedOutput = ""
    )

    if ($ExitCode -ne 0) {
        return $false
    }

    $combined = ($Output | ForEach-Object { "$_" }) -join "`n"
    if ($combined -match "Connection reset|Host key verification failed|Permission denied|Operation timed out|No route to host") {
        return $false
    }

    if ($ExpectedOutput -and ($combined -notmatch [regex]::Escape($ExpectedOutput))) {
        return $false
    }

    return $true
}

function Format-SshFailureMessage {
    param(
        [string]$RemoteHost,
        [int]$ExitCode,
        [string[]]$Output,
        [string]$Step
    )

    $combined = ($Output | ForEach-Object { "$_" }) -join "`n"
    $targetHost = Get-RemoteTargetHost -RemoteHost $RemoteHost

    $hints = @(
        "Verify the Pi is powered on and reachable on your LAN.",
        "Test manually: ssh $RemoteHost `"echo ok`"",
        "If scp fails, ensure deploy.config.json has `"sshClient`": `"git`" (Git for Windows scp is more reliable than Windows OpenSSH here)."
    )

    if ($ExitCode -eq 124 -or $combined -match "Command timed out") {
        $hints += "SSH hung and was killed after the per-attempt timeout. The Pi may be overloaded or sshd unresponsive; reboot the Pi and retry."
    }
    elseif ($combined -match "Host key verification failed") {
        if ($targetHost -match "^\d{1,3}(\.\d{1,3}){3}$") {
            $hints = @(
                "SSH host key is stored under the hostname, not the IP.",
                "Prefer the hostname in deploy config: crichards999@neonbeam-lens.richwerks.local",
                "Or add the IP to known_hosts: ssh-keyscan -H $targetHost >> `$env:USERPROFILE\.ssh\known_hosts"
            )
        }
        else {
            $hints += "Remove stale known_hosts entries for $targetHost and reconnect once interactively."
        }
    }
    elseif ($combined -match "Connection reset|Connection timed out|Operation timed out|No route to host") {
        $hints += @(
            "Connection reset/timeouts are often transient (Pi busy, Wi-Fi, sshd restarting).",
            "Re-run deploy; increase sshRetryCount or sshConnectTimeout in deploy.config.json if needed."
        )
    }

    return @"
$Step failed with exit code $ExitCode for $RemoteHost.

SSH output:
$combined

Try:
$($hints | ForEach-Object { "  - $_" } | Out-String)
"@
}

function Invoke-RemoteSsh {
    param(
        [string]$RemoteHost,
        [string]$Command,
        [hashtable]$DeployConfig,
        [string]$SshExe,
        [string]$Step = "ssh",
        [int]$MaxAttempts = 0
    )

    if ($MaxAttempts -le 0) {
        $MaxAttempts = $DeployConfig.SshRetryCount
    }

    $sshArgs = Get-SshBaseArgs -DeployConfig $DeployConfig
    $attempt = 0
    $lastOutput = @()
    $exitCode = 255

    while ($attempt -lt $MaxAttempts) {
        $attempt++
        if ($attempt -gt 1) {
            $delay = $DeployConfig.SshRetryDelaySeconds
            Write-Host "Retrying $Step ($attempt/$MaxAttempts) after ${delay}s..."
            Start-Sleep -Seconds $delay
        }

        $result = Invoke-ExternalCommand -FilePath $SshExe -ArgumentList ($sshArgs + @($RemoteHost, $Command))
        $lastOutput = $result.Output
        $exitCode = $result.ExitCode

        if (Test-SshCommandSucceeded -ExitCode $exitCode -Output $lastOutput) {
            if ($lastOutput) {
                $lastOutput | ForEach-Object { Write-Host $_ }
            }
            return
        }
    }

    throw (Format-SshFailureMessage -RemoteHost $RemoteHost -ExitCode $(if ($exitCode -eq 0) { 255 } else { $exitCode }) -Output $lastOutput -Step $Step)
}

function Invoke-RemoteScp {
    param(
        [string]$LocalPath,
        [string]$RemoteDestination,
        [hashtable]$DeployConfig,
        [string]$ScpExe,
        [string]$Step = "scp"
    )

    $scpArgs = Get-SshBaseArgs -DeployConfig $DeployConfig
    $localPathForScp = $LocalPath
    if ($ScpExe -match '\\Git\\usr\\bin\\scp\.exe$') {
        $localPathForScp = ConvertTo-GitUnixPath -Path $LocalPath
    }

    $attempt = 0
    $lastOutput = @()
    $exitCode = 255

    while ($attempt -lt $DeployConfig.SshRetryCount) {
        $attempt++
        if ($attempt -gt 1) {
            $delay = $DeployConfig.SshRetryDelaySeconds
            Write-Host "Retrying $Step ($attempt/$($DeployConfig.SshRetryCount)) after ${delay}s..."
            Start-Sleep -Seconds $delay
        }

        $result = Invoke-ExternalCommand -FilePath $ScpExe -ArgumentList ($scpArgs + @($localPathForScp, $RemoteDestination))
        $lastOutput = $result.Output
        $exitCode = $result.ExitCode

        if (Test-ScpCommandSucceeded -ExitCode $exitCode -Output $lastOutput) {
            if ($lastOutput) {
                $lastOutput | ForEach-Object { Write-Host $_ }
            }
            return
        }
    }

    throw (Format-SshFailureMessage -RemoteHost $RemoteDestination -ExitCode $(if ($exitCode -eq 0) { 255 } else { $exitCode }) -Output $lastOutput -Step $Step)
}

function Test-RemoteHostConnection {
    param([string]$RemoteHost)

    $target = Get-RemoteTargetHost -RemoteHost $RemoteHost

    if ($target -match "^\d{1,3}(\.\d{1,3}){3}$") {
        Write-Host "Using Pi IP address: $target"
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
  1. Use the Pi hostname (recommended):
       .\deploy\deploy.ps1 -RemoteHost "crichards999@neonbeam-lens.richwerks.local" -Restart
  2. Create deploy/deploy.config.json (see deploy/deploy.config.example.json)
  3. Add to C:\Windows\System32\drivers\etc\hosts:
       192.168.x.x    neonbeam-lens.richwerks.local
  4. Ensure the Pi and this PC are on the same network

Original error: $($_.Exception.Message)
"@
    }
}

function Test-SshConnection {
    param(
        [string]$RemoteHost,
        [hashtable]$DeployConfig,
        [string]$SshExe
    )

    Write-Host "Testing SSH to $RemoteHost (timeout $($DeployConfig.SshConnectTimeout)s, up to $($DeployConfig.SshRetryCount) attempts)..."
    Invoke-RemoteSsh -RemoteHost $RemoteHost -Command "echo deploy-ssh-ok" -DeployConfig $DeployConfig -SshExe $SshExe -Step "SSH connectivity test" -MaxAttempts $DeployConfig.SshRetryCount
    Write-Host "SSH connectivity OK."
}

Require-Command tar
Require-Command scp
Require-Command ssh

$configPath = Join-Path $PSScriptRoot "deploy.config.json"
$deployConfig = Get-DeployConfig -ConfigPath $configPath `
    -DefaultHost "crichards999@neonbeam-lens.richwerks.local" `
    -DefaultPath "/home/crichards999/object-detection-v2"

if (-not $RemoteHost) { $RemoteHost = $deployConfig.RemoteHost }
if (-not $RemotePath) { $RemotePath = $deployConfig.RemotePath }

$sshClients = Resolve-SshClients -DeployConfig $deployConfig

Write-Host "Deploy target: ${RemoteHost}:${RemotePath}"
Test-RemoteHostConnection -RemoteHost $RemoteHost
Test-SshConnection -RemoteHost $RemoteHost -DeployConfig $deployConfig -SshExe $sshClients.Ssh

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
    "web/node_modules",
    "*.pyc",
    "*.pyo",
    "*.hef",
    "*.pt",
    "*.onnx",
    $ArchiveName
)

Write-Host "Project root: $ProjectRoot"

$WebDir = Join-Path $ProjectRoot "web"
$PackageJson = Join-Path $WebDir "package.json"
if (Test-Path -LiteralPath $PackageJson) {
    Require-Command npm
    Write-Host "Building annotation UI (web/dist)..."
    Push-Location $WebDir
    try {
        if (Test-Path "package-lock.json") {
            npm ci
        } else {
            npm install
        }
        if ($LASTEXITCODE -ne 0) { throw "npm install failed with exit code $LASTEXITCODE" }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}

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
Invoke-RemoteSsh -RemoteHost $RemoteHost -Command "mkdir -p '$RemotePath'" -DeployConfig $deployConfig -SshExe $sshClients.Ssh -Step "ssh mkdir"

Write-Host "Uploading to ${RemoteHost}:${RemoteArchive} ..."
Invoke-RemoteScp -LocalPath $ArchivePath -RemoteDestination "${RemoteHost}:${RemoteArchive}" -DeployConfig $deployConfig -ScpExe $sshClients.Scp -Step "scp upload"

Write-Host "Extracting on remote host..."
$RemoteCmd = "set -e && cd '$RemotePath' && tar -xzf '$ArchiveName' && rm -f '$ArchiveName' && find . -name '*.sh' -type f -exec chmod +x {} + && echo 'Deploy extract complete; shell scripts chmod +x.'"
if ($Restart) {
    $RemoteCmd += " && if systemctl is-active --quiet laser-detection; then sudo systemctl restart laser-detection && echo 'Service restarted: laser-detection'; else echo 'Service laser-detection is not active (skipped restart).'; fi"
}
Invoke-RemoteSsh -RemoteHost $RemoteHost -Command $RemoteCmd -DeployConfig $deployConfig -SshExe $sshClients.Ssh -Step "ssh extract"

Remove-Item $ArchivePath -Force
Write-Host "Deploy finished successfully."
