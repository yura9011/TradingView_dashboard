@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo    Testing Qwen2-VL Implementation
echo ============================================================
echo.

:: Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual
    echo Ejecuta primero: install_local.bat
    pause
    exit /b 1
)

:: Run test
python test_qwen_implementation.py

echo.
pause
