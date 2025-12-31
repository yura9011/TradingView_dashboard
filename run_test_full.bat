@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo    Verificacion Completa del Sistema (Capture + Analysis)
echo ============================================================
echo.

:: Check for virtual environment (venv first, then venv311 for compatibility)
if exist "%~dp0venv\Scripts\activate.bat" (
    echo Activando entorno virtual...
    call "%~dp0venv\Scripts\activate.bat"
) else if exist "%~dp0venv311\Scripts\activate.bat" (
    echo Activando entorno virtual (venv311)...
    call "%~dp0venv311\Scripts\activate.bat"
) else (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta primero: install_local.bat
    pause
    exit /b 1
)

echo.
echo Ejecutando prueba completa en MELI (Auto-detect Exchange)...
echo Esto puede tomar unos segundos...
echo.

python "%~dp0scripts\run_full_system_test.py" --symbol MELI

echo.
echo ============================================================
if errorlevel 1 (
    echo [ERROR] La prueba fallo. Revisa los logs arriba.
) else (
    echo [EXITO] Prueba completada correctamente.
    echo Revisa la carpeta "data/charts" para ver la imagen anotada.
)
echo ============================================================
pause
