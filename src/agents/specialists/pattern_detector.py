"""
Pattern Detector Agent - Identifies chart patterns with coordinates.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple

from src.agents.specialists.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class PatternDetectorAgent(BaseAgent):
    """Specialist agent for detecting chart patterns."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            prompt_file="pattern_detector.yaml",
            api_key=api_key,
        )
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse pattern detection response.
        
        Expected format (v2.0):
        PATTERN: [name]
        CONFIDENCE: [0.0-1.0]
        PATTERN_BOX: [x1,y1,x2,y2]
        COMPONENTS: [description of pattern components]
        TARGET: [theoretical price target]
        INVALIDATION: [price level that invalidates pattern]
        DESCRIPTION: [text]
        """
        result = {
            "pattern": "none",
            "confidence": 0.0,
            "pattern_box": None,
            "components": None,
            "target": None,
            "invalidation": None,
            "description": "",
        }
        
        lines = raw_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("PATTERN:"):
                result["pattern"] = line.replace("PATTERN:", "").strip().lower()
                
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                    result["confidence"] = min(max(conf, 0.0), 1.0)
                except ValueError:
                    pass
                    
            elif line.startswith("PATTERN_BOX:"):
                box_str = line.replace("PATTERN_BOX:", "").strip()
                result["pattern_box"] = self._parse_box(box_str)
            
            elif line.startswith("COMPONENTS:"):
                result["components"] = line.replace("COMPONENTS:", "").strip()
            
            elif line.startswith("TARGET:"):
                result["target"] = line.replace("TARGET:", "").strip()
            
            elif line.startswith("INVALIDATION:"):
                result["invalidation"] = line.replace("INVALIDATION:", "").strip()
                
            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.replace("DESCRIPTION:", "").strip()
        
        # If description spans multiple lines, capture the rest
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Pattern detected: {result['pattern']} (confidence: {result['confidence']})")
        if result["pattern_box"]:
            logger.info(f"Pattern box: {result['pattern_box']}")
        if result["target"]:
            logger.info(f"Target: {result['target']}")
        
        return result
    
    def _parse_box(self, val: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse PATTERN_BOX coordinates."""
        if not val or val.lower() in ["none", "n/a", "-"]:
            return None
        
        try:
            # Remove any non-numeric characters except commas and periods
            cleaned = re.sub(r'[^\d,.]', '', val)
            parts = [int(float(p.strip())) for p in cleaned.split(",") if p.strip()]
            if len(parts) == 4:
                return tuple(parts)
        except (ValueError, TypeError):
            pass
        
        return None
