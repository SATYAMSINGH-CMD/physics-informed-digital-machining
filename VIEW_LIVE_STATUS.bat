@echo off
title Tony Dataset - Live Milling AI Monitor
cd /d "D:\tony dataset"
echo ========================================================
echo   STARTING LIVE DIGITAL TWIN EXTRACTION DASHBOARD...
echo ========================================================
echo.
streamlit run live_dashboard.py
pause
