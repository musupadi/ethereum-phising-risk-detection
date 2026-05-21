@echo off
cd /d %~dp0
.venv\Scripts\python.exe ml\extract_addresses_from_graph.py
pause
