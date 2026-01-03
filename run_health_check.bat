@echo off
echo ========================================
echo   TradingAI - System Health Check
echo ========================================
echo.

REM Try venv311 first, then venv
if exist "venv311\Scripts\python.exe" (
    set PYTHON=venv311\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    echo ERROR: No se encontro entorno virtual (venv o venv311)
    echo Por favor ejecute: python -m venv venv311
    pause
    exit /b 1
)

echo Usando: %PYTHON%
echo.

%PYTHON% main_multiagent_local.py --health-check

echo.
pause
