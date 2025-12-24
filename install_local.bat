@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo    Trading Analysis - Instalacion Automatica (Modelo Local)
echo ============================================================
echo.

:: Check Python
echo [1/6] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado. Instala Python 3.10 o 3.11 desde python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo       Python %PYVER% encontrado

:: Create virtual environment
echo.
echo [2/6] Creando entorno virtual...
if exist venv (
    echo       Entorno virtual ya existe, saltando...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
    echo       Entorno virtual creado
)

:: Activate virtual environment
echo.
echo [3/6] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)
echo       Entorno activado

:: Detect CUDA and install PyTorch
echo.
echo [4/6] Instalando PyTorch...
echo       Detectando GPU...

python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    :: PyTorch not installed, detect CUDA and install
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo       No se detecto GPU NVIDIA, instalando version CPU...
        echo       ADVERTENCIA: El analisis sera MUY lento sin GPU
        pip install torch torchvision --quiet
    ) else (
        echo       GPU NVIDIA detectada, instalando PyTorch con CUDA 12.1...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    )
) else (
    echo       PyTorch ya instalado
)

:: Verify PyTorch installation
python -c "import torch; print('       PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())"

:: Install dependencies
echo.
echo [5/6] Instalando dependencias...
pip install -r requirements_local.txt --quiet
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias
    pause
    exit /b 1
)
echo       Dependencias instaladas

:: Create directories and config
echo.
echo [6/6] Configurando proyecto...
if not exist data\charts mkdir data\charts
if not exist data\reports mkdir data\reports
if not exist logs mkdir logs
if not exist config\config.yaml (
    copy config\config.example.yaml config\config.yaml >nul 2>&1
)
echo       Directorios creados

:: Done
echo.
echo ============================================================
echo    INSTALACION COMPLETADA
echo ============================================================
echo.
echo Para ejecutar un analisis:
echo    1. Abre una terminal en esta carpeta
echo    2. Ejecuta: venv\Scripts\activate
echo    3. Ejecuta: python main_multiagent_local.py --symbol AAPL
echo.
echo O simplemente ejecuta: run_analysis.bat AAPL
echo.
echo NOTA: La primera ejecucion descargara el modelo (~8GB)
echo.
pause
