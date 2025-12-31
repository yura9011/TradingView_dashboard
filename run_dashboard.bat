@echo off
chcp 65001 >nul

:: Get the directory where this bat file is located
cd /d "%~dp0"

echo ============================================
echo    Trading Analysis Dashboard
echo ============================================
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

:: Check if Flask is installed, if not install it
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    pip install flask pandas openpyxl
)

echo Starting dashboard at http://localhost:8080
echo Press Ctrl+C to stop
echo.

python "%~dp0dashboard\app.py"

pause
