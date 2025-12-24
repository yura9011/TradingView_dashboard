@echo off
chcp 65001 >nul
setlocal

echo ============================================
echo    Trading Analysis Dashboard
echo ============================================
echo.

:: Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    echo Run install_local.bat first.
    pause
    exit /b 1
)

:: Check if Flask is installed, if not install it
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    pip install flask pandas openpyxl --quiet
)

echo Starting dashboard at http://localhost:8080
echo Press Ctrl+C to stop
echo.

python dashboard/app.py

pause
