# 🤖 Trading Analysis - Modelo Local

Sistema de análisis técnico con IA usando **Qwen2-VL-7B-Instruct** de Alibaba. No requiere API externa.

## ⚡ Instalación Rápida

```bash
git clone https://github.com/yura9011/TradingView_dashboard.git
cd TradingView_dashboard
git checkout feature/local-phi-model
```

Luego doble click en **`install_local.bat`**

## 🚀 Uso

### Analizar un símbolo
```bash
run_analysis.bat AAPL
```

### Abrir el Dashboard
```bash
run_dashboard.bat
```
Luego abrir http://localhost:8080

### Análisis masivo (268 símbolos)
1. Abrir dashboard
2. Click en "Bulk Analysis"
3. Click en "Start Analysis"

## 💻 Requisitos

- Windows 10/11
- Python 3.10 o 3.11
- GPU NVIDIA con 8GB+ VRAM (recomendado)
- 20GB espacio en disco

## 📁 Estructura

```
data/
  charts/     → Capturas de gráficos
  reports/    → Reportes generados
  signals.db  → Base de datos
```

## ❓ Problemas comunes

**"CUDA out of memory"** → Cerrar otras apps que usen la GPU

**"Model download failed"** → Verificar conexión a internet y espacio en disco

**Análisis muy lento** → Sin GPU el análisis tarda ~20 min por símbolo

---

📖 Ver [TUTORIAL.md](TUTORIAL.md) para documentación completa.
