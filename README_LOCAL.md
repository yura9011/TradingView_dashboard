# QuantAgents Local - Multi-Agent Trading Analysis

Sistema de análisis técnico multi-agente usando **YOLO** para patrones + **Qwen2-VL** para tendencia/niveles.

## 🚀 Instalación Rápida

```bash
# Windows - Ejecutar instalador automático
install_local.bat
```

### Requisitos
- **Python 3.10 o 3.11** (recomendado)
- **GPU NVIDIA** (opcional, recomendado para velocidad)
- **8GB+ VRAM** para Qwen2-VL-2B

## 📊 Uso

### Ejecutar análisis
```bash
run_analysis.bat AAPL
```

### Ver dashboard
```bash
run_dashboard.bat
```

## 🧠 Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATOR (Otto)                            │
│                    Final Synthesis + Veto Logic                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────────┐
    │                   │                       │
    ▼                   ▼                       ▼
┌─────────────┐   ┌─────────────┐        ┌─────────────┐
│   YOLO      │   │  Qwen2-VL   │        │  Qwen2-VL   │
│  Pattern    │   │   Trend     │        │   Levels    │
│  Detector   │   │  Analyst    │        │   Calc      │
│  (93% acc)  │   │             │        │             │
└─────────────┘   └─────────────┘        └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────────┐       ┌─────────────┐        ┌─────────────┐
│    Dave     │       │   Emily     │        │   TRIPLE    │
│    Risk     │       │  Sentiment  │        │    VETO     │
│   Manager   │       │   Analyst   │        │   SYSTEM    │
│(rule-based) │       │(rule-based) │        │             │
└─────────────┘       └─────────────┘        └─────────────┘
```

## 🎯 Patrones Detectados (YOLO)

| Patrón | Clase YOLO | Descripción |
|--------|------------|-------------|
| Double Top | M_Head | Bearish reversal |
| Double Bottom | W_Bottom | Bullish reversal |
| Head & Shoulders Top | Head and shoulders top | Bearish reversal |
| Head & Shoulders Bottom | Head and shoulders bottom | Bullish reversal |
| Triangle | Triangle | Continuation |

**Accuracy reportada:** 93% mAP @ IoU 0.5

## 🛡️ Sistema de Veto

1. **RISK VETO (Dave):** ATR% > 5% → DANGEROUS → Veto automático
2. **SENTIMENT VETO (Emily):** Sentiment < -0.5 + setup bullish → Veto
3. **FAKEOUT VETO:** Breakout + RVOL < 1.5 → Veto por bajo volumen

## ⚙️ Configuración

### Usar YOLO (por defecto)
```python
from src.agents.coordinator_local import get_coordinator_local

coordinator = get_coordinator_local(use_yolo=True)  # YOLO para patrones
```

### Usar VLM solo (sin YOLO)
```python
coordinator = get_coordinator_local(use_yolo=False)  # Qwen2-VL para todo
```

## 📁 Estructura

```
src/agents/
├── coordinator_local.py      # Orquestador principal
├── specialists/
│   ├── pattern_detector_yolo.py  # YOLO (nuevo!)
│   ├── pattern_detector_local.py # VLM (fallback)
│   ├── trend_analyst_local.py    # Qwen2-VL
│   ├── levels_calculator_local.py# Qwen2-VL
│   ├── risk_manager_local.py     # Dave (rule-based)
│   └── news_analyst_local.py     # Emily (rule-based)
```

## ⏱️ Tiempos de Ejecución

| Componente | Hardware | Tiempo |
|------------|----------|--------|
| YOLO Pattern Detection | CPU | ~2s |
| Qwen2-VL (Trend+Levels) | RTX 3070 | ~30-60s |
| Qwen2-VL (Trend+Levels) | CPU | ~10-15 min |
| Risk/Sentiment Analysis | CPU | <1s |

## 📤 Output

Cada análisis genera:
- Signal en DB (`data/signals.db`)
- Chart capturado (`data/charts/`)
- Chart anotado con YOLO (`*_yolo.png`)
- Report markdown (`data/reports/`)
