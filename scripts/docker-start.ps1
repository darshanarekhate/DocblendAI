<#
.SYNOPSIS
  Start DocBlendAI in Docker on Windows and open it in the browser.

.DESCRIPTION
  1. Checks that Docker Desktop is installed and running.
  2. Creates .env from .env.example if it is missing, asking for your Gemini API key
     (typed hidden, never printed; leave it empty to run without Gemini).
  3. Runs "docker compose up -d" (add -Build to build the image here, -Pull to use the
     pre-built image from the registry instead).
  4. Waits until http://localhost:<port>/health answers, then opens the app.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\docker-start.ps1
  powershell -ExecutionPolicy Bypass -File scripts\docker-start.ps1 -Build
  powershell -ExecutionPolicy Bypass -File scripts\docker-start.ps1 -Pull -NoBrowser
#>
param(
    [switch]$Build,      # build the image from this checkout (docker compose up -d --build)
    [switch]$Pull,       # pull the pre-built image (DOCBLENDAI_IMAGE) and do not build
    [switch]$NoBrowser,  # do not open the browser at the end
    [int]$Port = 8000,
    [int]$TimeoutMinutes = 20
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # the repository root

function Fail($message) { Write-Host "ERROR: $message" -ForegroundColor Red; exit 1 }

# 1. Docker installed and running?
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "Docker is not installed. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker is not running. Start Docker Desktop, wait until it says 'Engine running', then run this again." }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { Fail "'docker compose' is missing. Update Docker Desktop." }

# 2. .env with the Gemini key (never printed)
if (-not (Test-Path ".env")) {
    if (-not (Test-Path ".env.example")) { Fail ".env.example is missing: run this from a DocBlendAI checkout." }
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example."
    $secure = Read-Host "Gemini API key (https://aistudio.google.com/apikey; Enter to skip)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if ($key) {
        $lines = Get-Content ".env" | ForEach-Object { if ($_ -match '^GEMINI_API_KEY=') { "GEMINI_API_KEY=$key" } else { $_ } }
        # UTF-8 without BOM: python-dotenv / Docker read the first key correctly.
        [IO.File]::WriteAllLines((Join-Path (Get-Location) ".env"), [string[]]$lines, (New-Object Text.UTF8Encoding $false))
        Write-Host "Saved the key in .env (not shown)."
    } else {
        Write-Host "No key: answers and LLM refinement stay off until you set GEMINI_API_KEY in .env." -ForegroundColor Yellow
    }
    $key = $null
}

# 3. Start
$env:DOCBLENDAI_PORT = "$Port"
if ($Pull) {
    docker compose pull
    if ($LASTEXITCODE -ne 0) { Fail "docker compose pull failed (is the image published? see README: Run with Docker)." }
    docker compose up -d --no-build
} elseif ($Build) {
    Write-Host "Building the image (first time: 10-20 minutes)..."
    docker compose up -d --build
} else {
    docker compose up -d
}
if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed. If port $Port is busy, run again with -Port 8001." }

# 4. Wait for /health
$url = "http://localhost:$Port"
Write-Host "Waiting for $url/health (the first start loads the app; up to $TimeoutMinutes min)..."
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 3
}
if (-not $ready) { Fail "DocBlendAI did not answer in time. Check: docker compose logs --tail 100" }
Write-Host "DocBlendAI is running: $url  (Experience Center: $url/studio)" -ForegroundColor Green
Write-Host "Stop it with: docker compose down"
if (-not $NoBrowser) { Start-Process $url }
