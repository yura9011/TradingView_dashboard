"""Agents package for AI Trading Analysis Agent.

This package provides both cloud (Gemini) and local (Phi-3.5-vision) agents.
Imports are lazy to avoid requiring google-generativeai when using local models.
"""

# Lazy imports to avoid requiring google-generativeai when using local models
# Use: from src.agents.gemini_client import GeminiClient
# Or:  from src.agents.coordinator_local import get_coordinator_local

__all__ = [
    # Cloud (Gemini) - import directly from submodules
    "GeminiClient",
    "AnalysisResult", 
    "get_gemini_client",
    "ChartAnalyzer",
    "get_chart_analyzer",
    # Local (Phi) - import directly from submodules
    "get_coordinator_local",
]


def __getattr__(name):
    """Lazy import to avoid loading google-generativeai unless needed."""
    if name in ("GeminiClient", "AnalysisResult", "get_gemini_client"):
        from .gemini_client import GeminiClient, AnalysisResult, get_gemini_client
        return {"GeminiClient": GeminiClient, "AnalysisResult": AnalysisResult, "get_gemini_client": get_gemini_client}[name]
    
    if name in ("ChartAnalyzer", "get_chart_analyzer"):
        from .chart_analyzer import ChartAnalyzer, get_chart_analyzer
        return {"ChartAnalyzer": ChartAnalyzer, "get_chart_analyzer": get_chart_analyzer}[name]
    
    if name == "get_coordinator_local":
        from .coordinator_local import get_coordinator_local
        return get_coordinator_local
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
