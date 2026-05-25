# Start the FastAPI service using HOST and PORT from the project .env file.
#
# Usage (from project root):
#   .\scripts\start.ps1
#
# Copy .env.example to .env and edit PORT as needed (e.g. 8100 for manual testing).

$ErrorActionPreference = "Stop"

function Read-DotEnv {
    param([string]$Path)

    $vars = @{}
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $line.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) {
            continue
        }
        $key = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $vars[$key] = $value
    }
    return $vars
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$HostName = "0.0.0.0"
$Port = "8000"
$EnvFile = Join-Path $Root ".env"

if (Test-Path -LiteralPath $EnvFile) {
    $envVars = Read-DotEnv -Path $EnvFile
    if ($envVars.ContainsKey("HOST")) { $HostName = $envVars["HOST"] }
    if ($envVars.ContainsKey("PORT")) { $Port = $envVars["PORT"] }
} else {
    Write-Host "No .env found; using HOST=$HostName PORT=$Port (copy .env.example to .env to customize)"
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Write-Host "Starting uvicorn on http://${HostName}:$Port"
& $Python -m uvicorn app.main:app --host $HostName --port $Port
