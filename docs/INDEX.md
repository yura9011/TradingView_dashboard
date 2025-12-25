# 📚 Documentación - AI Trading Analysis

## Guías de Usuario

| Documento | Descripción |
|-----------|-------------|
| [README_LOCAL.md](../README_LOCAL.md) | Guía rápida de instalación |
| [TUTORIAL.md](../TUTORIAL.md) | Tutorial completo paso a paso |

## Documentación Técnica

| Documento | Descripción |
|-----------|-------------|
| [src/README.md](../src/README.md) | Arquitectura del código fuente |
| [src/agents/README.md](../src/agents/README.md) | Sistema multi-agente |
| [ROADMAP.md](ROADMAP.md) | Plan de desarrollo |
| [REVIEW_REPORT.md](REVIEW_REPORT.md) | Reporte de revisión de código |

## Research

Documentos de investigación en `docs/research/`:

- Arquitectura de agentes de IA para trading
- Introducción al trading algorítmico con IA

## Estructura del Proyecto

```
TradingView_dashboard/
├── config/              # Configuración
├── dashboard/           # Web UI (Flask)
├── data/                # Datos y DB
│   ├── charts/          # Screenshots capturados
│   ├── reports/         # Reportes generados
│   └── signals.db       # Base de datos SQLite
├── docs/                # Documentación
├── logs/                # Logs de ejecución
├── prompts/             # Prompts de los agentes
├── src/                 # Código fuente
│   ├── agents/          # Agentes de IA
│   ├── database/        # Persistencia
│   ├── models/          # Schemas
│   ├── screener/        # Captura TradingView
│   └── visual/          # Anotación de imágenes
├── tests/               # Tests
├── main_multiagent.py       # Entry point (Gemini)
├── main_multiagent_local.py # Entry point (Local)
├── install_local.bat    # Instalador automático
├── run_analysis.bat     # Ejecutar análisis
└── run_dashboard.bat    # Iniciar dashboard
```

## Branches

| Branch | Descripción |
|--------|-------------|
| `main` | Versión con Gemini API |
| `feature/local-phi-model` | Versión con modelo local Phi-3.5 |
