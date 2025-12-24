"""Specialist agents package.

Lazy imports to avoid requiring google-generativeai or torch unless needed.
Import directly from submodules:
  - Gemini: from src.agents.specialists.pattern_detector import PatternDetectorAgent
  - Local:  from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
"""

__all__ = [
    # Gemini API agents
    "PatternDetectorAgent",
    "TrendAnalystAgent", 
    "LevelsCalculatorAgent",
    # Local model agents
    "PatternDetectorAgentLocal",
    "TrendAnalystAgentLocal",
    "LevelsCalculatorAgentLocal",
]


def __getattr__(name):
    """Lazy import to avoid loading dependencies unless needed."""
    # Gemini-based agents
    if name == "PatternDetectorAgent":
        from src.agents.specialists.pattern_detector import PatternDetectorAgent
        return PatternDetectorAgent
    if name == "TrendAnalystAgent":
        from src.agents.specialists.trend_analyst import TrendAnalystAgent
        return TrendAnalystAgent
    if name == "LevelsCalculatorAgent":
        from src.agents.specialists.levels_calculator import LevelsCalculatorAgent
        return LevelsCalculatorAgent
    
    # Local model agents
    if name == "PatternDetectorAgentLocal":
        from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
        return PatternDetectorAgentLocal
    if name == "TrendAnalystAgentLocal":
        from src.agents.specialists.trend_analyst_local import TrendAnalystAgentLocal
        return TrendAnalystAgentLocal
    if name == "LevelsCalculatorAgentLocal":
        from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal
        return LevelsCalculatorAgentLocal
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Helper function for getting all local agents at once
def get_local_agents():
    """Get local model agent classes. Requires torch to be installed."""
    from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
    from src.agents.specialists.trend_analyst_local import TrendAnalystAgentLocal
    from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal
    from src.agents.specialists.base_agent_local import BaseAgentLocal, LocalModelManager, AgentResponse
    
    return {
        "PatternDetectorAgentLocal": PatternDetectorAgentLocal,
        "TrendAnalystAgentLocal": TrendAnalystAgentLocal,
        "LevelsCalculatorAgentLocal": LevelsCalculatorAgentLocal,
        "BaseAgentLocal": BaseAgentLocal,
        "LocalModelManager": LocalModelManager,
        "AgentResponse": AgentResponse,
    }
