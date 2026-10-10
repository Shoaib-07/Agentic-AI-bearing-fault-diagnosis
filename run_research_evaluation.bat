@echo off
cd /d %~dp0
where py >nul 2>&1
if %errorlevel%==0 (
    py research_evaluation.py
) else (
    python research_evaluation.py
)
pause
