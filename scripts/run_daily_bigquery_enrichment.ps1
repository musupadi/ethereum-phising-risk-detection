$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'ml\enrich_real_addresses_bigquery.py'
$rawDir = Join-Path $root 'data\raw'
$inputCsv = Join-Path $rawDir 'addresses.csv'
$logDir = Join-Path $rawDir 'logs'

if (-not (Test-Path $python)) {
    Write-Error ".venv python not found at $python"
}

if (-not (Test-Path $inputCsv)) {
    Write-Error "Input CSV not found at $inputCsv. Create addresses.csv with column: address"
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "bigquery_enrichment_$timestamp.log"

Write-Host "Running daily BigQuery enrichment..."
Write-Host "Log: $logFile"
Write-Host "Note: BIGQUERY_* and GOOGLE_APPLICATION_CREDENTIALS are loaded from .env by the Python script"

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python $script 2>&1 | Tee-Object -FilePath $logFile
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference

if ($exitCode -ne 0) {
    Write-Error "Daily BigQuery enrichment failed with exit code $exitCode"
}

Write-Host "Finished."
