# 📊 Tutorial: Trading Analysis con Modelo Local Phi-3.5

Este tutorial te guiará paso a paso para configurar y ejecutar el sistema de análisis de trading usando el modelo local **Phi-3.5-vision-instruct** de Microsoft, sin necesidad de APIs externas.

---

## 📋 Tabla de Contenidos

1. [Requisitos del Sistema](#-requisitos-del-sistema)
2. [Instalación](#-instalación)
3. [Configuración](#-configuración)
4. [Ejecución del Análisis](#-ejecución-del-análisis)
5. [Uso del Dashboard](#-uso-del-dashboard)
6. [Solución de Problemas](#-solución-de-problemas)
7. [Preguntas Frecuentes](#-preguntas-frecuentes)

---

## 💻 Requisitos del Sistema

### Hardware Mínimo
| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 12+ GB |
| Almacenamiento | 20 GB libres | 50 GB SSD |
| CPU | 4 cores | 8+ cores |

### GPUs Compatibles
- **NVIDIA**: RTX 3060 (12GB), RTX 3070, RTX 3080, RTX 4060, RTX 4070, RTX 4080, RTX 4090
- **AMD**: ROCm compatible (experimental)
- **CPU**: Funciona pero es muy lento (10-30 minutos por análisis)

### Software Requerido
- **Python**: 3.10 o 3.11 (recomendado)
- **CUDA Toolkit**: 11.8 o 12.1 (para GPU NVIDIA)
- **Git**: Para clonar el repositorio
- **Chrome/Chromium**: Para captura de gráficos

---

## 🔧 Instalación

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/yura9011/TradingView_dashboard.git
cd TradingView_dashboard
```

### Paso 2: Cambiar al Branch del Modelo Local

```bash
git checkout feature/local-phi-model
```

### Paso 3: Crear Entorno Virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Paso 4: Instalar PyTorch

⚠️ **IMPORTANTE**: Instala PyTorch ANTES de las otras dependencias.

**Para GPU NVIDIA con CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Para GPU NVIDIA con CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Solo CPU (muy lento):**
```bash
pip install torch torchvision
```

### Paso 5: Verificar Instalación de PyTorch

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Deberías ver algo como:
```
PyTorch: 2.1.0+cu118
CUDA disponible: True
GPU: NVIDIA GeForce RTX 3080
```

### Paso 6: Instalar Dependencias

```bash
pip install -r requirements_local.txt
```

### Paso 7: (Opcional) Instalar Flash Attention 2

Flash Attention acelera la inferencia significativamente. Requiere CUDA toolkit instalado.

```bash
pip install flash-attn --no-build-isolation
```

Si falla, el sistema funcionará sin Flash Attention (un poco más lento).

### Paso 8: Instalar ChromeDriver

El sistema usa Selenium para capturar gráficos de TradingView.

**Windows:**
1. Descarga ChromeDriver de: https://chromedriver.chromium.org/downloads
2. Asegúrate de que la versión coincida con tu Chrome
3. Coloca `chromedriver.exe` en el PATH o en la carpeta del proyecto

**Linux:**
```bash
sudo apt install chromium-chromedriver
```

**Mac:**
```bash
brew install chromedriver
```

---

## ⚙️ Configuración

### Paso 1: Crear Archivo de Configuración

```bash
cp config/config.example.yaml config/config.yaml
```

### Paso 2: Editar Configuración

Abre `config/config.yaml` y ajusta según necesites:

```yaml
# Configuración del modelo local
local_model:
  name: "microsoft/Phi-3.5-vision-instruct"
  device: "auto"  # auto, cuda, cpu
  
# Configuración de captura de gráficos
chart_capture:
  timeframe: "W"  # D=diario, W=semanal, M=mensual
  width: 1920
  height: 1080
  
# Base de datos
database:
  path: "data/signals.db"
```

### Paso 3: Crear Directorios Necesarios

```bash
mkdir -p data/charts data/reports logs
```

---

## 🚀 Ejecución del Análisis

### Análisis Básico

```bash
python main_multiagent_local.py --symbol AAPL
```

### Análisis con Opciones

```bash
# Especificar exchange
python main_multiagent_local.py --symbol MELI --exchange NASDAQ

# Usar modelo específico
python main_multiagent_local.py --symbol TSLA --model microsoft/Phi-3.5-vision-instruct

# Saltar verificación del sistema
python main_multiagent_local.py --symbol GOOGL --skip-check
```

### Primera Ejecución

⚠️ La primera ejecución descargará el modelo (~8GB). Esto puede tomar 10-30 minutos dependiendo de tu conexión.

```
🔍 SYSTEM CHECK
============================================================
  CUDA Available: ✅ Yes
  GPU: NVIDIA GeForce RTX 3080
  VRAM: 10.0 GB
============================================================

🚀 Local Multi-Agent Analysis: NASDAQ:MELI
📦 Model: microsoft/Phi-3.5-vision-instruct
============================================================
📸 Capturing chart (weekly timeframe)...
   Chart saved: data/charts/MELI_20241224_123456.png

🤖 Running Local Multi-Agent Analysis...
   (First run will download the model ~8GB)
Loading local model: microsoft/Phi-3.5-vision-instruct
Using CUDA: NVIDIA GeForce RTX 3080
Model loaded successfully

🔍 Step 1/3: Pattern Detection (local)...
📈 Step 2/3: Trend Analysis (local)...
📊 Step 3/3: Levels Calculation (local)...
🧠 Synthesizing findings...

============================================================
📊 ANALYSIS RESULTS (Local Model)
============================================================
  SIGNAL TYPE: CANDIDATE
  OVERALL CONFIDENCE: 75%
------------------------------------------------------------
  PATTERN:
    Name: head and shoulders
    Confidence: 75%
...
```

---

## 📱 Uso del Dashboard

### Iniciar el Dashboard

**Windows:**
```cmd
run_dashboard.bat
```

**O manualmente:**
```bash
python dashboard/app.py
```

### Acceder al Dashboard

Abre tu navegador en: **http://localhost:5000**

### Funcionalidades del Dashboard

1. **Lista de Señales**: Ver todas las señales analizadas
2. **Detalle de Señal**: Click en cualquier señal para ver:
   - Gráfico anotado
   - Análisis de patrón
   - Análisis de tendencia (Wyckoff/Elliott)
   - Niveles de soporte/resistencia
   - Razonamiento detallado del modelo
3. **Filtros**: Filtrar por tipo de señal, patrón, fecha

---

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"

**Causa**: GPU sin suficiente VRAM.

**Soluciones**:
1. Cierra otras aplicaciones que usen la GPU
2. Reduce el tamaño del batch (ya está en 1)
3. Usa CPU (muy lento):
   ```bash
   python main_multiagent_local.py --symbol AAPL
   # El sistema detectará automáticamente si no hay GPU
   ```

### Error: "No module named 'torch'"

**Solución**: Instala PyTorch correctamente:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "ChromeDriver not found"

**Solución**: 
1. Verifica que Chrome esté instalado
2. Descarga ChromeDriver de la versión correcta
3. Agrega al PATH o coloca en la carpeta del proyecto

### Error: "Model download failed"

**Causa**: Problema de conexión o espacio en disco.

**Soluciones**:
1. Verifica conexión a internet
2. Asegúrate de tener 20GB+ libres
3. Intenta descargar manualmente:
   ```python
   from transformers import AutoModelForCausalLM, AutoProcessor
   AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
   AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
   ```

### El análisis es muy lento

**Causas y soluciones**:
1. **Sin GPU**: Instala CUDA y PyTorch con soporte CUDA
2. **Sin Flash Attention**: Intenta instalar flash-attn
3. **GPU antigua**: Considera usar una GPU más potente

### El modelo no detecta patrones correctamente

**Sugerencias**:
1. Asegúrate de que el gráfico tenga buena resolución
2. Usa timeframes más largos (semanal mejor que diario)
3. El modelo funciona mejor con patrones claros y definidos

---

## ❓ Preguntas Frecuentes

### ¿Puedo usar otro modelo?

Sí, pero debe ser un modelo de visión. Opciones compatibles:
- `microsoft/Phi-3.5-vision-instruct` (recomendado)
- `llava-hf/llava-1.5-7b-hf`
- Otros modelos VLM de HuggingFace

### ¿Cuánto tarda un análisis?

| Hardware | Tiempo aproximado |
|----------|-------------------|
| RTX 4090 | 30-60 segundos |
| RTX 3080 | 1-2 minutos |
| RTX 3060 | 2-4 minutos |
| CPU | 10-30 minutos |

### ¿Necesito internet para ejecutar?

- **Primera vez**: Sí, para descargar el modelo
- **Después**: Solo para capturar gráficos de TradingView

### ¿Dónde se guardan los datos?

- **Gráficos**: `data/charts/`
- **Reportes**: `data/reports/`
- **Base de datos**: `data/signals.db`
- **Logs**: `logs/agent.log`

### ¿Cómo actualizo el modelo?

```bash
pip install --upgrade transformers accelerate
# El modelo se re-descargará si hay nueva versión
```

---

## 📞 Soporte

Si tienes problemas:
1. Revisa los logs en `logs/agent.log`
2. Verifica los requisitos del sistema
3. Abre un issue en GitHub con:
   - Descripción del error
   - Output del comando
   - Especificaciones de tu sistema

---

## 📄 Licencia

Este proyecto usa el modelo Phi-3.5-vision-instruct de Microsoft bajo la licencia MIT.
