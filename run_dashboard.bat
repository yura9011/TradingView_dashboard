@echo off
echo ====================================
echo   Trading Analysis Dashboard
echo ====================================
echo.
echo Starting dashboard at http://localhost:8080
echo Press Ctrl+C to stop
echo.
cd /d "%~dp0"
python -m dashboard.app
pause
