@echo off
cd /d %~dp0
powershell -ExecutionPolicy Bypass -File scripts\run_daily_bigquery_enrichment.ps1
pause
