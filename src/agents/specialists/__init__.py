"""Specialist agents package."""

from src.agents.specialists.pattern_detector import PatternDetectorAgent
from src.agents.specialists.trend_analyst import TrendAnalystAgent
from src.agents.specialists.levels_calculator import LevelsCalculatorAgent

__all__ = [
    # Gemini API agents
    "PatternDetectorAgent",
    "TrendAnalystAgent", 
    "LevelsCalculatorAgent",
]

# Local model agents (Phi-3.5-vision) - imported conditionally to avoid torch dependency
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
