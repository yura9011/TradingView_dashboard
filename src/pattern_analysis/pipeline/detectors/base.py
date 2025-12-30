"""
Base detector class for pattern detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ...models.dataclasses import BoundingBox, FeatureMap, PatternDetection


class BaseDetector(ABC):
    """Base class for all pattern detectors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def detect(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect patterns and return list of detections."""
        pass
    
    def update_config(self, config: Dict[str, Any]):
        """Update detector configuration."""
        self.config.update(config)
