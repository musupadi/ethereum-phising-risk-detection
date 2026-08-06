# Menjalankan halaman DApp RiskRegistry (pembaca on-chain).
#
# Disajikan dari folder `chain/` - bukan `chain/dapp/` - supaya halaman bisa
# membaca ../artifacts/deployment.json dan mengisi alamat kontrak otomatis.
#
# Prasyarat (terminal terpisah):
#   1) cd chain ; npx hardhat node
#   2) cd chain ; npx hardhat run scripts/deploy.ts --network localhost
#
# Jalankan:  .\chain\dapp\run_dapp.ps1
#
# CATATAN: file ini sengaja dijaga murni ASCII. PowerShell 5.1 membaca .ps1 tanpa
# BOM sebagai ANSI, sehingga karakter non-ASCII (mis. em dash) berubah menjadi
# tanda kutip keriting dan merusak parser.

$ErrorActionPreference = "Stop"

$chainDir = Split-Path -Parent $PSScriptRoot
$root     = Split-Path -Parent $chainDir
$python   = Join-Path $root ".venv\Scripts\python.exe"
$port     = 8503

if (-not (Test-Path $python)) { $python = "python" }

$deployment = Join-Path $chainDir "artifacts\deployment.json"
if (Test-Path $deployment) {
    $addr = (Get-Content $deployment -Raw | ConvertFrom-Json).address
    Write-Host "Kontrak terdeteksi : $addr" -ForegroundColor Green
} else {
    Write-Host "deployment.json belum ada. Jalankan deploy.ts lebih dulu:" -ForegroundColor Yellow
    Write-Host "  cd chain ; npx hardhat run scripts/deploy.ts --network localhost" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  DApp RiskRegistry  ->  http://localhost:$port/dapp/" -ForegroundColor Cyan
Write-Host "  (Ctrl+C untuk berhenti)"
Write-Host ""

Set-Location $chainDir
& $python -m http.server $port
