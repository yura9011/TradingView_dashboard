"""
Chart annotator for visual pattern annotation.

This module implements the ChartAnnotator class that draws bounding boxes,
labels, and overlays on chart images to visualize detected patterns.

Requirements:
- 8.1: Draw bounding boxes on the original chart image
- 8.2: Use color coding to distinguish pattern types (green for bullish, red for bearish, yellow for neutral)
- 8.3: Include labels with pattern name and confidence score
- 8.4: Use semi-transparent overlays to avoid obscuring original chart data
- 8.5: Save annotated image in PNG format with lossless compression
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
from ..models.enums import PatternCategory, PatternType


logger = logging.getLogger(__name__)


class ChartAnnotator:
    """
    Annotates chart images with detected pattern visualizations.
    
    Draws bounding boxes, labels, and overlays on chart images to
    visualize detected patterns. Uses color coding to distinguish
    between bullish, bearish, and neutral patterns.
    
    Requirements: 8.1-8.5
    
    Attributes:
        colors: Dictionary mapping PatternCategory to BGR color tuples
        font: OpenCV font for text rendering
        font_scale: Scale factor for font size
        font_thickness: Thickness of font strokes
        box_thickness: Thickness of bounding box lines
        overlay_alpha: Transparency level for overlays (0.0-1.0)
    """
    
    # Color scheme: BGR format for OpenCV
    # Green for bullish (reversal to up, continuation up)
    # Red for bearish (reversal to down, continuation down)
    # Yellow for bilateral/neutral
    DEFAULT_COLORS = {
        PatternCategory.REVERSAL: {
            "bullish": (0, 200, 0),      # Green
            "bearish": (0, 0, 200),      # Red
            "default": (0, 200, 200),    # Yellow
        },
        PatternCategory.CONTINUATION: {
            "bullish": (0, 255, 100),    # Light green
            "bearish": (100, 100, 255),  # Light red
            "default": (0, 200, 200),    # Yellow
        },
        PatternCategory.BILATERAL: (0, 255, 255),  # Yellow
    }
    
    # Bullish patterns (typically green)
    BULLISH_PATTERNS = {
        PatternType.INVERSE_HEAD_SHOULDERS,
        PatternType.DOUBLE_BOTTOM,
        PatternType.TRIPLE_BOTTOM,
        PatternType.ASCENDING_TRIANGLE,
        PatternType.BULL_FLAG,
        PatternType.CUP_AND_HANDLE,
        PatternType.FALLING_WEDGE,
        PatternType.CHANNEL_UP,
    }
    
    # Bearish patterns (typically red)
    BEARISH_PATTERNS = {
        PatternType.HEAD_SHOULDERS,
        PatternType.DOUBLE_TOP,
        PatternType.TRIPLE_TOP,
        PatternType.DESCENDING_TRIANGLE,
        PatternType.BEAR_FLAG,
        PatternType.RISING_WEDGE,
        PatternType.CHANNEL_DOWN,
    }
    
    def __init__(
        self,
        colors: Optional[Dict[PatternCategory, Any]] = None,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        box_thickness: int = 2,
        overlay_alpha: float = 0.3
    ):
        """
        Initialize the chart annotator.
        
        Args:
            colors: Optional custom color mapping for categories
            font_scale: Scale factor for font size (default 0.6)
            font_thickness: Thickness of font strokes (default 1)
            box_thickness: Thickness of bounding box lines (default 2)
            overlay_alpha: Transparency for overlays, 0.0-1.0 (default 0.3)
        """
        self.colors = colors if colors is not None else self.DEFAULT_COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.overlay_alpha = max(0.0, min(1.0, overlay_alpha))
    
    def annotate(
        self,
        image: np.ndarray,
        detections: List[PatternDetection],
        validated_detections: Optional[List[ValidationResult]] = None,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> np.ndarray:
        """
        Annotate image with pattern detections.
        
        Draws bounding boxes, labels, and optional overlays for each
        detected pattern on the image.
        
        Args:
            image: Input image as numpy array (RGB or BGR)
            detections: List of pattern detections to annotate
            validated_detections: Optional validation results for detections
            show_confidence: Whether to show confidence scores in labels
            show_validation: Whether to show validation status
            
        Returns:
            Annotated image as numpy array
            
        Requirements: 8.1, 8.2, 8.3, 8.4
        """
        # Create a copy to avoid modifying the original
        annotated = image.copy()
        
        # Build validation lookup if provided
        validation_map: Dict[int, ValidationResult] = {}
        if validated_detections:
            for vr in validated_detections:
                if vr.original_detection:
                    # Use detection's bounding box as key
                    key = self._detection_key(vr.original_detection)
                    validation_map[key] = vr
        
        # Draw each detection
        for detection in detections:
            # Get color for this detection
            color = self._get_color_for_detection(detection)
            
            # Draw bounding box
            annotated = self._draw_bounding_box(
                annotated,
                detection.bounding_box,
                color
            )
            
            # Draw semi-transparent overlay
            annotated = self._draw_overlay(
                annotated,
                detection.bounding_box,
                color
            )
            
            # Build label text
            label = self._build_label(
                detection,
                validation_map.get(self._detection_key(detection)),
                show_confidence,
                show_validation
            )
            
            # Draw label
            annotated = self._draw_label(
                annotated,
                detection.bounding_box,
                label,
                color
            )
        
        return annotated

    def annotate_from_result(
        self,
        image: np.ndarray,
        result: AnalysisResult,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> np.ndarray:
        """
        Annotate image using an AnalysisResult.
        
        Convenience method that extracts detections and validations
        from an AnalysisResult object.
        
        Args:
            image: Input image as numpy array
            result: AnalysisResult containing detections
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            
        Returns:
            Annotated image as numpy array
        """
        return self.annotate(
            image,
            result.detections,
            result.validated_detections,
            show_confidence,
            show_validation
        )
    
    def _draw_bounding_box(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a bounding box rectangle on the image.
        
        Args:
            image: Image to draw on
            bbox: Bounding box coordinates
            color: BGR color tuple
            
        Returns:
            Image with bounding box drawn
            
        Requirements: 8.1
        """
        cv2.rectangle(
            image,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            self.box_thickness
        )
        return image
    
    def _draw_overlay(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a semi-transparent overlay on the bounding box region.
        
        Args:
            image: Image to draw on
            bbox: Bounding box coordinates
            color: BGR color tuple
            
        Returns:
            Image with overlay drawn
            
        Requirements: 8.4
        """
        if self.overlay_alpha <= 0:
            return image
        
        # Create overlay
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            -1  # Filled rectangle
        )
        
        # Blend with original
        cv2.addWeighted(
            overlay,
            self.overlay_alpha,
            image,
            1 - self.overlay_alpha,
            0,
            image
        )
        
        return image
    
    def _draw_label(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        label: str,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a label above the bounding box.
        
        Args:
            image: Image to draw on
            bbox: Bounding box coordinates
            label: Text to display
            color: BGR color tuple
            
        Returns:
            Image with label drawn
            
        Requirements: 8.3
        """
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            self.font,
            self.font_scale,
            self.font_thickness
        )
        
        # Position label above bounding box
        label_x = bbox.x1
        label_y = bbox.y1 - 5
        
        # If label would go off top of image, put it inside the box
        if label_y - text_height < 0:
            label_y = bbox.y1 + text_height + 5
        
        # Draw background rectangle for label
        bg_x1 = label_x
        bg_y1 = label_y - text_height - 2
        bg_x2 = label_x + text_width + 4
        bg_y2 = label_y + 2
        
        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1  # Filled
        )
        
        # Draw text in contrasting color (white or black)
        text_color = self._get_contrasting_color(color)
        cv2.putText(
            image,
            label,
            (label_x + 2, label_y),
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness
        )
        
        return image
    
    def _build_label(
        self,
        detection: PatternDetection,
        validation: Optional[ValidationResult],
        show_confidence: bool,
        show_validation: bool
    ) -> str:
        """
        Build label text for a detection.
        
        Args:
            detection: Pattern detection
            validation: Optional validation result
            show_confidence: Whether to include confidence
            show_validation: Whether to include validation status
            
        Returns:
            Label string
        """
        # Get pattern name (formatted)
        pattern_name = detection.pattern_type.value.replace("_", " ").title()
        
        parts = [pattern_name]
        
        if show_confidence:
            parts.append(f"{detection.confidence:.0%}")
        
        if show_validation and validation:
            if validation.is_confirmed:
                parts.append("✓")
            else:
                parts.append("?")
        
        return " ".join(parts)
    
    def _get_color_for_detection(
        self,
        detection: PatternDetection
    ) -> Tuple[int, int, int]:
        """
        Get the appropriate color for a detection based on its category and type.
        
        Args:
            detection: Pattern detection
            
        Returns:
            BGR color tuple
            
        Requirements: 8.2
        """
        category = detection.category
        pattern_type = detection.pattern_type
        
        # Handle bilateral patterns (always yellow)
        if category == PatternCategory.BILATERAL:
            color = self.colors.get(PatternCategory.BILATERAL, (0, 255, 255))
            if isinstance(color, tuple):
                return color
            return color.get("default", (0, 255, 255))
        
        # Determine if bullish or bearish
        if pattern_type in self.BULLISH_PATTERNS:
            direction = "bullish"
        elif pattern_type in self.BEARISH_PATTERNS:
            direction = "bearish"
        else:
            direction = "default"
        
        # Get color from category colors
        category_colors = self.colors.get(category, {})
        if isinstance(category_colors, tuple):
            return category_colors
        
        return category_colors.get(direction, (0, 200, 200))
    
    def _get_contrasting_color(
        self,
        color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Get a contrasting color (white or black) for text readability.
        
        Args:
            color: BGR color tuple
            
        Returns:
            White (255, 255, 255) or black (0, 0, 0)
        """
        # Calculate luminance (simplified)
        b, g, r = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Return white for dark backgrounds, black for light
        return (255, 255, 255) if luminance < 128 else (0, 0, 0)
    
    def _detection_key(self, detection: PatternDetection) -> int:
        """Generate a hash key for a detection based on its bounding box."""
        bbox = detection.bounding_box
        return hash((bbox.x1, bbox.y1, bbox.x2, bbox.y2, detection.pattern_type))
    
    def save(
        self,
        image: np.ndarray,
        output_path: str,
        compression_level: int = 9
    ) -> bool:
        """
        Save annotated image to PNG file with lossless compression.
        
        Args:
            image: Annotated image to save
            output_path: Path to save the image
            compression_level: PNG compression level 0-9 (default 9, max compression)
            
        Returns:
            True if save was successful, False otherwise
            
        Requirements: 8.5
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Set PNG compression parameters
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            
            # Save image
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
        """
        Annotate image and save to file in one operation.
        
        Args:
            image: Input image
            detections: Pattern detections to annotate
            output_path: Path to save the annotated image
            validated_detections: Optional validation results
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            
        Returns:
            True if successful, False otherwise
        """
        annotated = self.annotate(
            image,
            detections,
            validated_detections,
            show_confidence,
            show_validation
        )
        return self.save(annotated, output_path)
    
    def get_color_for_category(
        self,
        category: PatternCategory
    ) -> Tuple[int, int, int]:
        """
        Get the default color for a pattern category.
        
        Useful for legend generation or external color coordination.
        
        Args:
            category: Pattern category
            
        Returns:
            BGR color tuple
        """
        color = self.colors.get(category, (0, 200, 200))
        if isinstance(color, tuple):
            return color
        return color.get("default", (0, 200, 200))
