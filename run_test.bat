@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo    Testing Sistema de Analisis
echo ============================================================
echo.

:: Activate virtual environment (check venv first, then venv311 for compatibility)
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
) else if exist "%~dp0venv311\Scripts\activate.bat" (
    call "%~dp0venv311\Scripts\activate.bat"
) else (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta primero: install_local.bat
    pause
    exit /b 1
)

:: Run integration tests
echo Ejecutando tests de integracion...
python -m pytest tests/test_chart_analysis_integration.py -v --tb=short

echo.
pause
