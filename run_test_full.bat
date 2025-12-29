@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo    Verificacion Completa del Sistema (Capture + Analysis)
echo ============================================================
echo.

if not exist "%~dp0venv" (
    echo ERROR: Entorno virtual no encontrado. Ejecuta install_local.bat primero.
    pause
    exit /b 1
)

echo Activando entorno virtual...
call "%~dp0venv\Scripts\activate.bat"

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
