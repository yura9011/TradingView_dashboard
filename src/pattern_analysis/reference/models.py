"""
Data models for pattern reference comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# Pattern name mapping from various sources to internal names
PATTERN_NAME_MAPPING = {
    # YOLO model class names -> internal names (after YOLO's own mapping)
    "head_shoulders": "head_shoulders",
    "inverse_head_shoulders": "inverse_head_shoulders",
    "double_top": "double_top",
    "double_bottom": "double_bottom",
    "triangle": "symmetric_triangle",  # YOLO generic triangle -> symmetric
    
    # File name patterns -> internal names (for reference images)
    "hch techo": "head_shoulders",
    "hch suelo": "inverse_head_shoulders",
    "head shoulders top": "head_shoulders",
    "head shoulders bottom": "inverse_head_shoulders",
    "dobles techos": "double_top",
    "double tops": "double_top",
    "dobles suelos": "double_bottom",
    "double bottoms": "double_bottom",
    "triangulo ascendente": "ascending_triangle",
    "ascending triangle": "ascending_triangle",
    "triangulo descendente": "descending_triangle",
    "descending triangle": "descending_triangle",
    "triangulo simetrico": "symmetric_triangle",
    "symmetric triangle": "symmetric_triangle",
    "bandera": "flag",
    "flag": "flag",
    "banderin": "pennant",
    "pennant": "pennant",
    "rising wedge": "rising_wedge",
    "falling wedge": "falling_wedge",
    "cup with handle": "cup_and_handle",
    "copa con asa": "cup_and_handle",
    "diamond tops": "diamond_top",
    "diamond bottoms": "diamond_bottom",
    "broadening tops": "broadening_top",
    "broadening bottoms": "broadening_bottom",
    "adam": "double_bottom",  # Adam & Adam, Adam & Eve variants
    "eva": "double_bottom",
}

# Spanish display names for patterns
PATTERN_DISPLAY_NAMES = {
    "head_shoulders": "Hombro-Cabeza-Hombro",
    "inverse_head_shoulders": "HCH Invertido",
    "double_top": "Doble Techo",
    "double_bottom": "Doble Suelo",
    "triple_top": "Triple Techo",
    "triple_bottom": "Triple Suelo",
    "ascending_triangle": "Triángulo Ascendente",
    "descending_triangle": "Triángulo Descendente",
    "symmetric_triangle": "Triángulo Simétrico",
    "flag": "Bandera",
    "pennant": "Banderín",
    "rising_wedge": "Cuña Ascendente",
    "falling_wedge": "Cuña Descendente",
    "cup_and_handle": "Copa con Asa",
    "diamond_top": "Diamante Techo",
    "diamond_bottom": "Diamante Suelo",
    "broadening_top": "Ensanchamiento Techo",
    "broadening_bottom": "Ensanchamiento Suelo",
}


@dataclass
class ReferenceImage:
    """Reference image from trading book."""
    
    pattern_type: str  # Internal pattern type name
    image_path: str  # Path to reference image file
    image_data: Optional[np.ndarray] = None  # Loaded image data (BGR)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional metadata
    
    @property
    def display_name(self) -> str:
        """Get Spanish display name for pattern."""
        return PATTERN_DISPLAY_NAMES.get(self.pattern_type, self.pattern_type)
    
    def __repr__(self) -> str:
        return f"ReferenceImage(type={self.pattern_type}, path={self.image_path})"


@dataclass
class MatchResult:
    """Result of matching a pattern region with a reference image."""
    
    reference: ReferenceImage  # Matched reference image
    similarity_score: float  # 0.0 to 1.0
    pattern_type: str  # Pattern type name
    
    @property
    def similarity_percent(self) -> int:
        """Get similarity as percentage (0-100)."""
        return int(self.similarity_score * 100)
    
    def __repr__(self) -> str:
        return f"MatchResult({self.pattern_type}, {self.similarity_percent}%)"


@dataclass
class PatternMatch:
    """Complete pattern detection with reference matches."""
    
    pattern_type: str  # Detected pattern type
    confidence: float  # YOLO confidence (0.0 to 1.0)
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    region_image: Optional[np.ndarray] = None  # Cropped pattern region
    matches: List[MatchResult] = field(default_factory=list)  # Reference matches
    
    @property
    def best_match(self) -> Optional[MatchResult]:
        """Get the best matching reference (highest similarity)."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity_score)
    
    @property
    def display_name(self) -> str:
        """Get Spanish display name for pattern."""
        return PATTERN_DISPLAY_NAMES.get(self.pattern_type, self.pattern_type)
    
    @property
    def confidence_percent(self) -> int:
        """Get confidence as percentage (0-100)."""
        return int(self.confidence * 100)


def normalize_pattern_name(name: str) -> str:
    """
    Normalize a pattern name to internal format.
    
    Args:
        name: Pattern name from any source (YOLO, filename, etc.)
        
    Returns:
        Normalized internal pattern name
    """
    # Convert to lowercase for matching
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in PATTERN_NAME_MAPPING:
        return PATTERN_NAME_MAPPING[name_lower]
    
    # Partial match - check if any key is contained in the name
    for key, value in PATTERN_NAME_MAPPING.items():
        if key in name_lower:
            return value
    
    # No match found, return cleaned version
    return name_lower.replace(" ", "_").replace("-", "_")
