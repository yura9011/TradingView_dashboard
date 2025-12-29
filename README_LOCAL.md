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

## 🧠 Arquitectura (VSA Upgrade)

El sistema ha evolucionado de Análisis Técnico Básico a **VSA (Volume Spread Analysis)** profesional.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATOR (Otto)                            │
│           Calcula Spread, RVOL y Close Position (VSA)           │
│                    Final Synthesis + Veto Logic                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────────┐
    │                   │                       │
    ▼                   ▼                       ▼
┌─────────────┐   ┌─────────────┐        ┌─────────────┐
│   YOLO      │   │ VSA Analyst │        │ Supply/Demand │
│  Pattern    │   │ (Prompt 3.0)│        │   Calc      │
│  Detector   │   │  Climaxes   │        │  Imbalance  │
│  (93% acc)  │   │  Traps      │        │  Zones      │
└─────────────┘   └─────────────┘        └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────────┐       ┌─────────────┐        ┌─────────────┐
│    Dave     │       │   Emily     │        │   TRIPLE    │
│    Risk     │       │ Psychology  │        │    VETO     │
│   Manager   │       │ Contrarian  │        │   SYSTEM    │
│(rule-based) │       │ Logic       │        │             │
└─────────────┘       └─────────────┘        └─────────────┘
```

## 💎 Nueva Lógica VSA ("Smart Money")
1. **Trend Analyst**: Ya no busca solo "tendencias". Busca Huellas Institucionales:
   - **Climaxes**: Buying/Selling Climax.
   - **Traps**: Shakeouts (Trampa bajista) y Upthrusts (Trampa alcista).
   - **Effort vs Result**: Anomalías entre Volumen y Precio.
2. **Levels Calculator**: Busca Zonas de **Oferta y Demanda** (Desequilibrio), no soportes estáticos.
3. **Psychology Analyst**: Aplica lógica contraria. Si hay **Euforia (RSI > 70)** y **Venta Institucional**, emite señal de VENTA.

## 🎯 Patrones Detectados (YOLO)

| Patrón | Clase YOLO | Descripción |
|--------|------------|-------------|
| Double Top | M_Head | Bearish reversal |
| Double Bottom | W_Bottom | Bullish reversal |
| Head & Shoulders Top | Head and shoulders top | Bearish reversal |
| Head & Shoulders Bottom | Head and shoulders bottom | Bullish reversal |
| Triangle | Triangle | Continuation |

**Accuracy reportada:** 93% mAP @ IoU 0.5

## 🛡️ Sistema de Veto Professional

1. **RISK VETO (Dave):** ATR% > 5% → DANGEROUS → Veto automático
2. **SMART MONEY VETO (Emily):** Euforia + Venta Institucional → Veto compra
3. **FAKEOUT VETO:** Breakout + RVOL < 1.5 → Veto por falta de interés profesional

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
├── coordinator_local.py      # Orquestador con métricas VSA (Spread/RVOL)
├── specialists/
│   ├── pattern_detector_yolo.py  # YOLO (Visual)
│   ├── pattern_detector_local.py # VLM (Fallback)
│   ├── trend_analyst_local.py    # VSA Specialist (Qwen2-VL)
│   ├── levels_calculator_local.py# Supply/Demand (Qwen2-VL)
│   ├── risk_manager_local.py     # Dave (Risk)
│   └── news_analyst_local.py     # Emily (Psychology/Contrarian)
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
