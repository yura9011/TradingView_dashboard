@echo off
chcp 65001 >nul
setlocal

:: Activate virtual environment
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo ERROR: Entorno virtual no encontrado. Ejecuta install_local.bat primero.
    pause
    exit /b 1
)

:: Get symbol from argument or use default
set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=MELI

echo.
echo ============================================================
echo    Analizando: %SYMBOL%
echo ============================================================
echo.

python main_multiagent_local.py --symbol %SYMBOL%

echo.
pause
