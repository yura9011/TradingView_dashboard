# 📦 Source Code - AI Trading Analysis

## Estructura de Módulos

```
src/
├── agents/              # Agentes de IA
│   ├── specialists/     # Agentes especializados
│   ├── coordinator.py   # Orquestador (Gemini API)
│   ├── coordinator_local.py  # Orquestador (Modelo Local)
│   ├── gemini_client.py # Cliente Gemini API
│   └── phi_client.py    # Cliente Phi-3.5 Local
├── database/            # Persistencia SQLite
├── models/              # Schemas Pydantic
├── screener/            # Captura de charts TradingView
├── visual/              # Anotación de imágenes
└── notifier/            # Notificaciones (Telegram)
```

## 🤖 Agentes (`agents/`)

### Arquitectura Multi-Agente

El sistema usa 3 agentes especializados coordinados:

| Agente | Archivo | Función |
|--------|---------|---------|
| **Pattern Detector** | `pattern_detector.py` | Detecta patrones chartistas (H&S, Double Top, etc.) |
| **Trend Analyst** | `trend_analyst.py` | Analiza tendencia, Wyckoff y Elliott Wave |
| **Levels Calculator** | `levels_calculator.py` | Calcula S/R, Fibonacci, niveles clave |

### Versiones

- **Gemini API** (`*_agent.py`): Usa Google Gemini Flash
- **Local** (`*_local.py`): Usa Phi-3.5-vision-instruct

### Coordinadores

```python
# Gemini API
from src.agents.coordinator import get_coordinator
coordinator = get_coordinator()
result = coordinator.analyze("chart.png", "AAPL")

# Modelo Local
from src.agents.coordinator_local import get_coordinator_local
coordinator = get_coordinator_local()
result = coordinator.analyze("chart.png", "AAPL")
```

### LocalModelManager

Singleton thread-safe que comparte el modelo entre agentes:

```python
from src.agents.specialists.base_agent_local import LocalModelManager

manager = LocalModelManager.get_instance()
model, processor = manager.load_model("microsoft/Phi-3.5-vision-instruct")
```

## 💾 Database (`database/`)

SQLite con las siguientes tablas:

- `signals`: Resultados de análisis

Campos principales:
- `symbol`, `signal_type`, `pattern_detected`, `pattern_confidence`
- `trend`, `trend_strength`, `market_phase`, `elliott_wave`
- `support_level`, `resistance_level`, `fibonacci_level`
- `analysis_summary`, `detailed_reasoning`

## 📊 Models (`models/`)

Schemas Pydantic:

- `Signal`: Resultado de análisis
- `SignalType`: Enum (candidate, pending, not_candidate)
- `PatternType`: Enum de patrones detectables

## 📸 Screener (`screener/`)

Captura screenshots de TradingView usando Selenium:

```python
from src.screener.chart_capture import get_chart_capture

capture = get_chart_capture()
path = capture.capture_sync("AAPL", "NASDAQ")
```

## 🎨 Visual (`visual/`)

Anotación de charts con PIL:

```python
from src.visual import get_annotator, get_report_generator

# Anotar pattern box
annotator = get_annotator()
annotator.draw_pattern_box(image, (x1, y1, x2, y2), "Head & Shoulders")

# Generar reporte completo
report_gen = get_report_generator()
report_path = report_gen.generate(signal, chart_path, annotate=True)
```

## 📱 Notifier (`notifier/`)

Envío de alertas por Telegram (opcional).

---

## Flujo de Datos

```
TradingView → Screener → Chart Image
                              ↓
                    ┌─────────┴─────────┐
                    ↓         ↓         ↓
              Pattern    Trend     Levels
              Detector   Analyst   Calculator
                    ↓         ↓         ↓
                    └─────────┬─────────┘
                              ↓
                        Coordinator
                              ↓
                    Signal + Summary
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
                Database            Visual
                (SQLite)         (Annotated Chart)
                              ↓
                          Dashboard
```
