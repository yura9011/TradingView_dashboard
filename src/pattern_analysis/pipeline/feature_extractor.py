"""
Edge-based feature extractor implementation for chart image analysis.

Orchestrates extraction of candlesticks, trendlines, support/resistance,
and volume profile from preprocessed chart images.

Feature: chart-pattern-analysis-framework
Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.dataclasses import BoundingBox, FeatureMap, PreprocessResult
from .interfaces import FeatureExtractor
from .extractors import (
    CandlestickExtractor,
    TrendlineDetector,
    SupportResistanceDetector,
    VolumeExtractor,
)


class EdgeBasedFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using edge detection and Hough transforms.
    
    Composes specialized extractors for each feature type.
    
    Requirements: 2.1-2.6
    """
    
    def __init__(self):
        """Initialize with component extractors."""
        self._candlestick_extractor = CandlestickExtractor()
        self._trendline_detector = TrendlineDetector()
        self._sr_detector = SupportResistanceDetector()
        self._volume_extractor = VolumeExtractor()
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this feature extractor."""
        return "edge_feature_extractor_v1"
    
    def process(
        self, input_data: PreprocessResult, config: Dict[str, Any]
    ) -> FeatureMap:
        """
        Extract features from preprocessed image.
        
        Args:
            input_data: PreprocessResult from preprocessor stage
            config: Configuration dictionary
                
        Returns:
            FeatureMap with extracted features
        """
        if not self.validate_input(input_data):
            return FeatureMap.empty()
        
        image = input_data.image
        
        candlesticks = self.extract_candlesticks(image, config)
        trendlines = self.detect_trendlines(image, config)
        support, resistance = self.find_support_resistance(image, config)
        volume = self._volume_extractor.extract(image, config)
        
        return FeatureMap(
            candlestick_regions=candlesticks,
            trendlines=trendlines,
            support_zones=support,
            resistance_zones=resistance,
            volume_profile=volume,
            quality_score=input_data.quality_score
        )
    
    def extract_candlesticks(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> List[BoundingBox]:
        """Extract candlestick body regions."""
        return self._candlestick_extractor.extract(image, config)
    
    def detect_trendlines(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect trendlines using edge detection."""
        return self._trendline_detector.detect(image, config)
    
    def find_support_resistance(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """Find horizontal support and resistance zones."""
        return self._sr_detector.detect(image, config)
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate that input is a valid PreprocessResult."""
        if not isinstance(input_data, PreprocessResult):
            return False
        
        if not isinstance(input_data.image, np.ndarray):
            return False
        
        if len(input_data.image.shape) != 3 or input_data.image.shape[2] != 3:
            return False
        
        return True
