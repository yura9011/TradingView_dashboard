"""
Chart annotator for visual pattern annotation.

Feature: chart-pattern-analysis-framework
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.dataclasses import (
    AnalysisResult,
    BoundingBox,
    PatternDetection,
    ValidationResult,
)
from ..models.enums import PatternCategory
from .colors import DEFAULT_COLORS, get_color_for_pattern, get_color_for_category
from .drawing import DrawingUtils


logger = logging.getLogger(__name__)


class ChartAnnotator:
    """
    Annotates chart images with detected pattern visualizations.
    
    Requirements: 8.1-8.5
    """
    
    def __init__(
        self,
        colors: Optional[Dict[PatternCategory, Any]] = None,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        box_thickness: int = 2,
        overlay_alpha: float = 0.3
    ):
        """Initialize the chart annotator."""
        self.colors = colors if colors is not None else DEFAULT_COLORS
        self.overlay_alpha = overlay_alpha
        self._drawing = DrawingUtils(
            font_scale=font_scale,
            font_thickness=font_thickness,
            box_thickness=box_thickness,
            overlay_alpha=overlay_alpha
        )
    
    def annotate(
        self,
        image: np.ndarray,
        detections: List[PatternDetection],
        validated_detections: Optional[List[ValidationResult]] = None,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> np.ndarray:
        """Annotate image with pattern detections."""
        annotated = image.copy()
        
        validation_map: Dict[int, ValidationResult] = {}
        if validated_detections:
            for vr in validated_detections:
                if vr.original_detection:
                    key = self._detection_key(vr.original_detection)
                    validation_map[key] = vr
        
        for detection in detections:
            color = get_color_for_pattern(
                detection.category,
                detection.pattern_type,
                self.colors
            )
            
            annotated = self._drawing.draw_bounding_box(
                annotated, detection.bounding_box, color
            )
            annotated = self._drawing.draw_overlay(
                annotated, detection.bounding_box, color
            )
            
            label = self._build_label(
                detection,
                validation_map.get(self._detection_key(detection)),
                show_confidence,
                show_validation
            )
            
            annotated = self._drawing.draw_label(
                annotated, detection.bounding_box, label, color
            )
        
        return annotated

    def annotate_from_result(
        self,
        image: np.ndarray,
        result: AnalysisResult,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> np.ndarray:
        """Annotate image using an AnalysisResult."""
        return self.annotate(
            image,
            result.detections,
            result.validated_detections,
            show_confidence,
            show_validation
        )
    
    def _build_label(
        self,
        detection: PatternDetection,
        validation: Optional[ValidationResult],
        show_confidence: bool,
        show_validation: bool
    ) -> str:
        """Build label text for a detection."""
        pattern_name = detection.pattern_type.value.replace("_", " ").title()
        parts = [pattern_name]
        
        if show_confidence:
            parts.append(f"{detection.confidence:.0%}")
        
        if show_validation and validation:
            parts.append("✓" if validation.is_confirmed else "?")
        
        return " ".join(parts)
    
    def _detection_key(self, detection: PatternDetection) -> int:
        """Generate a hash key for a detection."""
        bbox = detection.bounding_box
        return hash((bbox.x1, bbox.y1, bbox.x2, bbox.y2, detection.pattern_type))
    
    def save(
        self,
        image: np.ndarray,
        output_path: str,
        compression_level: int = 9
    ) -> bool:
        """Save annotated image to PNG file with lossless compression."""
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            success = cv2.imwrite(output_path, image, params)
            
            if success:
                logger.info(f"Saved annotated image to: {output_path}")
            else:
                logger.error(f"Failed to save image to: {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def annotate_and_save(
        self,
        image: np.ndarray,
        detections: List[PatternDetection],
        output_path: str,
        validated_detections: Optional[List[ValidationResult]] = None,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> bool:
        """Annotate image and save to file in one operation."""
        annotated = self.annotate(
            image, detections, validated_detections,
            show_confidence, show_validation
        )
        return self.save(annotated, output_path)
    
    def get_color_for_category(
        self,
        category: PatternCategory
    ) -> Tuple[int, int, int]:
        """Get the default color for a pattern category."""
        return get_color_for_category(category, self.colors)
