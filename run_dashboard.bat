@echo off
cd /d %~dp0
where py >nul 2>&1
if %errorlevel%==0 (
    py -m streamlit run dashboard\app.py
) else (
    python -m streamlit run dashboard\app.py
)
