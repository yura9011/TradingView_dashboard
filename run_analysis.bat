@echo off
chcp 65001 >nul

:: Get the directory where this bat file is located
cd /d "%~dp0"

:: Activate virtual environment
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo ERROR: Virtual environment not found. Run install_local.bat first.
    pause
    exit /b 1
)

:: Get symbol from argument or use default
set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=MELI

echo.
echo ============================================================
echo    Analyzing: %SYMBOL%
echo ============================================================
echo.

python "%~dp0main_multiagent_local.py" --symbol %SYMBOL%

echo.
pause
