@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo    Actualizando Trading Analysis desde GitHub
echo ============================================================
echo.

:: Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git no encontrado. Instala Git desde https://git-scm.com
    pause
    exit /b 1
)

:: Try normal pull first
echo [1/3] Descargando actualizaciones...
git pull origin main
if errorlevel 1 (
    echo.
    echo       git pull fallo, hay conflictos locales.
    echo       Descartando cambios locales y forzando actualizacion...
    git fetch origin
    git reset --hard origin/main
    if errorlevel 1 (
        echo ERROR: No se pudo actualizar. Verifica tu conexion a internet.
        pause
        exit /b 1
    )
)
echo       ✅ Codigo actualizado

:: Check if venv exists and offer to reinstall
echo.
echo [2/3] Verificando entorno virtual...
if exist "%~dp0venv" (
    echo       Entorno virtual encontrado: venv
    echo.
    set /p REINSTALL="¿Reinstalar dependencias? (S/N): "
    if /i "!REINSTALL!"=="S" (
        echo       Eliminando entorno virtual viejo...
        rmdir /s /q "%~dp0venv" 2>nul
        goto :install
    ) else (
        echo       Manteniendo entorno existente.
        echo       Si hay errores, ejecuta: install_local.bat
    )
) else if exist "%~dp0venv311" (
    echo       Entorno virtual viejo encontrado: venv311
    echo       Eliminando y creando nuevo...
    rmdir /s /q "%~dp0venv311" 2>nul
    goto :install
) else (
    echo       No hay entorno virtual, instalando...
    goto :install
)
goto :done

:install
echo.
echo [3/3] Ejecutando instalacion...
call "%~dp0install_local.bat"
goto :end

:done
echo.
echo ============================================================
echo    ACTUALIZACION COMPLETADA
echo ============================================================
echo.
echo Para ejecutar un analisis:
echo    run_analysis.bat AAPL
echo.

:end
pause
