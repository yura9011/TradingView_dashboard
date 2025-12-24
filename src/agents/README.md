# 🤖 Agents Module

Sistema multi-agente para análisis técnico de charts financieros.

## Arquitectura

```
agents/
├── coordinator.py          # Orquestador Gemini API
├── coordinator_local.py    # Orquestador Modelo Local
├── gemini_client.py        # Cliente Google Gemini
├── phi_client.py           # Cliente Phi-3.5-vision
├── chart_analyzer.py       # Analizador legacy (deprecated)
└── specialists/
    ├── base_agent.py           # Clase base Gemini
    ├── base_agent_local.py     # Clase base Local + ModelManager
    ├── pattern_detector.py     # Detector de patrones (Gemini)
    ├── pattern_detector_local.py
    ├── trend_analyst.py        # Analista de tendencia (Gemini)
    ├── trend_analyst_local.py
    ├── levels_calculator.py    # Calculador de niveles (Gemini)
    └── levels_calculator_local.py
```

## Agentes Especializados

### 1. Pattern Detector

Detecta patrones chartistas en la imagen.

**Output:**
- `pattern`: Nombre del patrón (head and shoulders, double bottom, etc.)
- `confidence`: 0.0 - 1.0
- `pattern_box`: Coordenadas (x1, y1, x2, y2) como % de imagen
- `components`: Descripción de componentes del patrón
- `target`: Precio objetivo teórico
- `invalidation`: Nivel de invalidación

### 2. Trend Analyst

Analiza dirección de tendencia, fase Wyckoff y onda Elliott.

**Output:**
- `trend`: up / down / sideways
- `strength`: strong / moderate / weak
- `phase`: accumulation / markup / distribution / markdown
- `wyckoff_event`: Spring, UTAD, SOW, etc.
- `wave`: Onda Elliott actual
- `wave_count`: Conteo de ondas

### 3. Levels Calculator

Calcula niveles técnicos de soporte/resistencia.

**Output:**
- `support` / `resistance`: Niveles primarios
- `support_reason` / `resistance_reason`: Justificación
- `support_secondary` / `resistance_secondary`: Niveles secundarios
- `fibonacci`: Nivel Fib relevante
- `fibonacci_confluence`: Confluencia con S/R
- `key_level`: Nivel más importante actual
- `key_level_reason`: Por qué es clave

## Coordinadores

### CoordinatorAgent (Gemini)

```python
from src.agents.coordinator import get_coordinator

coordinator = get_coordinator()
analysis = coordinator.analyze("path/to/chart.png", "AAPL")

# Resultado: CoordinatedAnalysis
print(analysis.signal_type)      # candidate / pending / not_candidate
print(analysis.pattern)          # head and shoulders
print(analysis.phase)            # distribution
print(analysis.summary)          # Resumen completo
print(analysis.detailed_reasoning)  # JSON con todo el razonamiento
```

### CoordinatorAgentLocal (Phi-3.5)

```python
from src.agents.coordinator_local import get_coordinator_local

coordinator = get_coordinator_local(model_name="microsoft/Phi-3.5-vision-instruct")
analysis = coordinator.analyze("path/to/chart.png", "AAPL")
```

## LocalModelManager

Singleton thread-safe para compartir el modelo entre agentes:

```python
from src.agents.specialists.base_agent_local import LocalModelManager

# El modelo se carga una sola vez y se comparte
manager = LocalModelManager.get_instance()
model, processor = manager.load_model()

# Los agentes usan el mismo manager internamente
pattern_agent = PatternDetectorAgentLocal()  # Usa el modelo compartido
trend_agent = TrendAnalystAgentLocal()       # Mismo modelo
levels_agent = LevelsCalculatorAgentLocal()  # Mismo modelo
```

## Prompts

Los prompts están en `/prompts/*.yaml`:

- `pattern_detector.yaml`: Instrucciones para detección de patrones
- `trend_analyst.yaml`: Instrucciones para análisis Wyckoff/Elliott
- `levels_calculator.yaml`: Instrucciones para cálculo de niveles

Cada prompt incluye:
- Rol del agente
- Expertise específico
- Formato de output esperado
- Ejemplos

## Flujo de Análisis

```
1. Coordinator recibe imagen + símbolo
2. Carga modelo (si es local) o usa API
3. Ejecuta secuencialmente:
   a. Pattern Detector → patrones
   b. Trend Analyst → tendencia/Wyckoff/Elliott
   c. Levels Calculator → S/R/Fibonacci
4. Sintetiza resultados en CoordinatedAnalysis
5. Genera summary y detailed_reasoning
6. Retorna resultado
```

## Manejo de Errores

Los agentes manejan errores gracefully:

```python
result = agent.analyze(image_path)
if not result.success:
    print(f"Error: {result.error}")
else:
    print(result.parsed)
```

Si un agente falla, el coordinator continúa con los demás y usa valores por defecto.
