# QuantAgents Local - Multi-Agent Trading Analysis

Sistema de análisis técnico multi-agente usando modelos de visión local (Qwen2-VL).

## Requisitos

### Hardware
- **GPU recomendada:** RTX 3070 o superior (8GB+ VRAM)
- **Modelo:** Qwen2-VL-2B-Instruct (~4GB VRAM)

### Software
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow
pip install tradingview-screener
pip install selenium webdriver-manager
```

## Uso Rápido

### Análisis con modelo local (GPU)
```bash
python main_multiagent_local.py --symbol AAPL
```

### Análisis con Gemini API (cloud)
```bash
python main_multiagent.py --symbol AAPL
```

## Arquitectura

```
┌──────────────────────────────────────────────────────┐
│                  Coordinator (Otto)                   │
│                  Final Synthesis                      │
└───────────────────────┬──────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Pattern │      │  Trend  │      │ Levels  │
│ Detector│      │ Analyst │      │ Calc    │
│ (Bob)   │      │         │      │         │
└─────────┘      └─────────┘      └─────────┘
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│  Risk   │      │Sentiment│      │  VETO   │
│ Manager │      │ Analyst │      │ System  │
│ (Dave)  │      │ (Emily) │      │         │
└─────────┘      └─────────┘      └─────────┘
```

## Sistema de Veto

El sistema incluye 3 niveles de veto para proteger el capital:

1. **RISK VETO (Dave):** Si ATR% > 5% → DANGEROUS → Veto automático
2. **SENTIMENT VETO (Emily):** Si sentiment < -0.5 + setup bullish → Veto
3. **FAKEOUT VETO:** Si breakout + RVOL < 1.5 → Veto por bajo volumen

## Tests

```bash
# Test de integración (sin modelo)
python test_integration_flow.py

# Test completo E2E (requiere GPU o usa moondream2 en CPU)
python test_full_flow_e2e.py --symbol TSLA
```

## Estructura de Archivos

```
src/agents/
├── coordinator_local.py      # Orquestador principal
├── specialists/
│   ├── pattern_detector*.py  # Detección de patrones
│   ├── trend_analyst*.py     # Análisis de tendencia
│   ├── levels_calculator*.py # Cálculo de niveles
│   ├── risk_manager_local.py # Dave (rule-based)
│   └── news_analyst_local.py # Emily (rule-based)
```

## Output

Cada análisis genera:
- Signal en DB (`data/signals.db`)
- Chart capturado (`data/charts/`)
- Report markdown (`data/reports/`)

## Tiempos de Ejecución

| Hardware | Tiempo por análisis |
|----------|---------------------|
| RTX 3070 | ~30-60 segundos |
| RTX 4090 | ~15-20 segundos |
| CPU only | ~10-15 minutos |
