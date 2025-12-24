"""Specialist agents package."""

from src.agents.specialists.pattern_detector import PatternDetectorAgent
from src.agents.specialists.trend_analyst import TrendAnalystAgent
from src.agents.specialists.levels_calculator import LevelsCalculatorAgent

# Local model agents (Phi-3.5-vision)
from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
from src.agents.specialists.trend_analyst_local import TrendAnalystAgentLocal
from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal
from src.agents.specialists.base_agent_local import BaseAgentLocal, LocalModelManager, AgentResponse

__all__ = [
    # Gemini API agents
    "PatternDetectorAgent",
    "TrendAnalystAgent", 
    "LevelsCalculatorAgent",
    # Local model agents
    "PatternDetectorAgentLocal",
    "TrendAnalystAgentLocal",
    "LevelsCalculatorAgentLocal",
    "BaseAgentLocal",
    "LocalModelManager",
    "AgentResponse",
]
