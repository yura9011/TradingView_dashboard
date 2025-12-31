"""
Chart annotator for visual pattern annotation.

Feature: chart-pattern-analysis-framework
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
Feature: chart-analysis-improvements
Requirements: 4.3 (draw analyzed region bounds)
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


# Color for analyzed region boundary (cyan/teal)
ANALYZED_REGION_COLOR = (0, 200, 200)


class ChartAnnotator:
    """
    Annotates chart images with detected pattern visualizations.
    
    Requirements: 8.1-8.5, 4.3 (chart-analysis-improvements)
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
        show_validation: bool = True,
        analyzed_region: Optional[BoundingBox] = None,
        show_analyzed_region: bool = True
    ) -> np.ndarray:
        """
        Annotate image with pattern detections.
        
        Args:
            image: Image to annotate
            detections: List of pattern detections
            validated_detections: Optional list of validation results
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            analyzed_region: Optional bounding box of analyzed region (Requirement 4.3)
            show_analyzed_region: Whether to draw the analyzed region boundary
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw analyzed region boundary first (so patterns are drawn on top)
        if show_analyzed_region and analyzed_region is not None:
            annotated = self._draw_analyzed_region(annotated, analyzed_region)
        
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
        show_validation: bool = True,
        show_analyzed_region: bool = True
    ) -> np.ndarray:
        """
        Annotate image using an AnalysisResult.
        
        Args:
            image: Image to annotate
            result: AnalysisResult containing detections and region metadata
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            show_analyzed_region: Whether to draw the analyzed region boundary (Requirement 4.3)
            
        Returns:
            Annotated image
        """
        return self.annotate(
            image,
            result.detections,
            result.validated_detections,
            show_confidence,
            show_validation,
            analyzed_region=result.analyzed_region,
            show_analyzed_region=show_analyzed_region
        )
    
    def _draw_analyzed_region(
        self,
        image: np.ndarray,
        region: BoundingBox
    ) -> np.ndarray:
        """
        Draw the analyzed region boundary on the image.
        
        Draws a dashed rectangle around the analyzed region with a label
        indicating it's the analysis boundary.
        
        Args:
            image: Image to draw on
            region: Bounding box of the analyzed region
            
        Returns:
            Image with analyzed region boundary drawn
            
        Requirements: 4.3 (chart-analysis-improvements)
        """
        # Draw dashed rectangle for analyzed region
        annotated = self._draw_dashed_rectangle(
            image,
            region,
            ANALYZED_REGION_COLOR,
            thickness=2,
            dash_length=10
        )
        
        # Draw label at top-left corner
        label = "Analyzed Region"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Position label inside the region at top-left
        label_x = region.x1 + 5
        label_y = region.y1 + text_height + 5
        
        # Draw background for label
        cv2.rectangle(
            annotated,
            (label_x - 2, label_y - text_height - 2),
            (label_x + text_width + 2, label_y + 2),
            ANALYZED_REGION_COLOR,
            -1
        )
        
        # Draw label text (black for contrast)
        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness
        )
        
        return annotated
    
    def _draw_dashed_rectangle(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10
    ) -> np.ndarray:
        """
        Draw a dashed rectangle on the image.
        
        Args:
            image: Image to draw on
            bbox: Bounding box coordinates
            color: RGB color tuple
            thickness: Line thickness
            dash_length: Length of each dash
            
        Returns:
            Image with dashed rectangle drawn
        """
        # Top edge
        self._draw_dashed_line(
            image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y1),
            color, thickness, dash_length
        )
        # Bottom edge
        self._draw_dashed_line(
            image, (bbox.x1, bbox.y2), (bbox.x2, bbox.y2),
            color, thickness, dash_length
        )
        # Left edge
        self._draw_dashed_line(
            image, (bbox.x1, bbox.y1), (bbox.x1, bbox.y2),
            color, thickness, dash_length
        )
        # Right edge
        self._draw_dashed_line(
            image, (bbox.x2, bbox.y1), (bbox.x2, bbox.y2),
            color, thickness, dash_length
        )
        
        return image
    
    def _draw_dashed_line(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int
    ) -> None:
        """
        Draw a dashed line on the image.
        
        Args:
            image: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: RGB color tuple
            thickness: Line thickness
            dash_length: Length of each dash
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = int(np.sqrt(dx * dx + dy * dy))
        
        if length == 0:
            return
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Draw dashes
        pos = 0
        draw = True
        while pos < length:
            if draw:
                start_x = int(x1 + pos * dx_norm)
                start_y = int(y1 + pos * dy_norm)
                end_pos = min(pos + dash_length, length)
                end_x = int(x1 + end_pos * dx_norm)
                end_y = int(y1 + end_pos * dy_norm)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            
            pos += dash_length
            draw = not draw
    
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
        show_validation: bool = True,
        analyzed_region: Optional[BoundingBox] = None,
        show_analyzed_region: bool = True
    ) -> bool:
        """
        Annotate image and save to file in one operation.
        
        Args:
            image: Image to annotate
            detections: List of pattern detections
            output_path: Path to save the annotated image
            validated_detections: Optional list of validation results
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            analyzed_region: Optional bounding box of analyzed region
            show_analyzed_region: Whether to draw the analyzed region boundary
            
        Returns:
            True if save was successful, False otherwise
        """
        annotated = self.annotate(
            image, detections, validated_detections,
            show_confidence, show_validation,
            analyzed_region, show_analyzed_region
        )
        return self.save(annotated, output_path)
    
    def get_color_for_category(
        self,
        category: PatternCategory
    ) -> Tuple[int, int, int]:
        """Get the default color for a pattern category."""
        return get_color_for_category(category, self.colors)
