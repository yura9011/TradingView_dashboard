"""
ML-based pattern detector using YOLO.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ...models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ...models.enums import PatternCategory, PatternType
from .base import BaseDetector


logger = logging.getLogger(__name__)


class MLDetector(BaseDetector):
    """Pattern detector using YOLO ML model."""
    
    # Class name to PatternType mapping
    CLASS_MAPPING = {
        "head and shoulders top": PatternType.HEAD_SHOULDERS,
        "head_shoulders": PatternType.HEAD_SHOULDERS,
        "h&s": PatternType.HEAD_SHOULDERS,
        "head and shoulders bottom": PatternType.INVERSE_HEAD_SHOULDERS,
        "inverse_head_shoulders": PatternType.INVERSE_HEAD_SHOULDERS,
        "m_head": PatternType.DOUBLE_TOP,
        "double_top": PatternType.DOUBLE_TOP,
        "double top": PatternType.DOUBLE_TOP,
        "w_bottom": PatternType.DOUBLE_BOTTOM,
        "double_bottom": PatternType.DOUBLE_BOTTOM,
        "double bottom": PatternType.DOUBLE_BOTTOM,
        "triangle": PatternType.SYMMETRICAL_TRIANGLE,
        "symmetrical_triangle": PatternType.SYMMETRICAL_TRIANGLE,
        "ascending_triangle": PatternType.ASCENDING_TRIANGLE,
        "descending_triangle": PatternType.DESCENDING_TRIANGLE,
        "rising_wedge": PatternType.RISING_WEDGE,
        "falling_wedge": PatternType.FALLING_WEDGE,
        "bull_flag": PatternType.BULL_FLAG,
        "bear_flag": PatternType.BEAR_FLAG,
        "cup_and_handle": PatternType.CUP_AND_HANDLE,
        "channel_up": PatternType.CHANNEL_UP,
        "channel_down": PatternType.CHANNEL_DOWN,
    }
    
    REVERSAL_PATTERNS = {
        PatternType.HEAD_SHOULDERS,
        PatternType.INVERSE_HEAD_SHOULDERS,
        PatternType.DOUBLE_TOP,
        PatternType.DOUBLE_BOTTOM,
        PatternType.TRIPLE_TOP,
        PatternType.TRIPLE_BOTTOM,
    }
    
    BILATERAL_PATTERNS = {
        PatternType.SYMMETRICAL_TRIANGLE,
    }
    
    def __init__(self, model_path: Optional[str] = None, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self._load_model(model_path) if model_path else None
    
    def _load_model(self, path: str) -> Optional[Any]:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            logger.info(f"Loaded ML model from {path}")
            return model
        except ImportError:
            logger.warning("ultralytics not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None
    
    def detect(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Run ML detection on image."""
        if self.model is None:
            return []
        
        try:
            conf_threshold = self.config.get("ml_confidence_threshold", 0.25)
            results = self.model.predict(image, verbose=False, conf=conf_threshold)
            return self._convert_results(results)
        except Exception as e:
            logger.warning(f"ML detection failed: {e}")
            return []
    
    def _convert_results(self, results: Any) -> List[PatternDetection]:
        """Convert YOLO results to PatternDetection objects."""
        detections: List[PatternDetection] = []
        
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue
            
            for box in result.boxes:
                detection = self._convert_box(box, result.names)
                if detection:
                    detections.append(detection)
        
        return detections
    
    def _convert_box(self, box: Any, names: Dict[int, str]) -> Optional[PatternDetection]:
        """Convert single box to PatternDetection."""
        try:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if x1 >= x2 or y1 >= y2:
                return None
            
            pattern_type = self._map_class(cls_id, names)
            if pattern_type is None:
                return None
            
            category = self._get_category(pattern_type)
            
            return PatternDetection(
                pattern_type=pattern_type,
                category=category,
                confidence=conf,
                bounding_box=BoundingBox(x1, y1, x2, y2),
                metadata={
                    "ml_class": names.get(cls_id, "unknown"),
                    "ml_class_id": cls_id
                },
                detector_id="ml_yolo"
            )
        except Exception as e:
            logger.debug(f"Failed to convert box: {e}")
            return None
    
    def _map_class(self, cls_id: int, names: Dict[int, str]) -> Optional[PatternType]:
        """Map class ID to PatternType."""
        class_name = names.get(cls_id, "").lower()
        return self.CLASS_MAPPING.get(class_name)
    
    def _get_category(self, pattern_type: PatternType) -> PatternCategory:
        """Get category for pattern type."""
        if pattern_type in self.REVERSAL_PATTERNS:
            return PatternCategory.REVERSAL
        elif pattern_type in self.BILATERAL_PATTERNS:
            return PatternCategory.BILATERAL
        return PatternCategory.CONTINUATION
