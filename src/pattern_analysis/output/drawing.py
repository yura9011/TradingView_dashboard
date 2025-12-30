"""
Drawing primitives for chart annotation.

Feature: chart-pattern-analysis-framework
Requirements: 8.1, 8.3, 8.4
"""

from typing import Tuple

import cv2
import numpy as np

from ..models.dataclasses import BoundingBox
from .colors import get_contrasting_color


class DrawingUtils:
    """Utility class for drawing annotation primitives."""
    
    def __init__(
        self,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        box_thickness: int = 2,
        overlay_alpha: float = 0.3
    ):
        """Initialize drawing utilities."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.overlay_alpha = max(0.0, min(1.0, overlay_alpha))
    
    def draw_bounding_box(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw a bounding box rectangle on the image."""
        cv2.rectangle(
            image,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            self.box_thickness
        )
        return image
    
    def draw_overlay(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw a semi-transparent overlay on the bounding box region."""
        if self.overlay_alpha <= 0:
            return image
        
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            -1
        )
        
        cv2.addWeighted(
            overlay,
            self.overlay_alpha,
            image,
            1 - self.overlay_alpha,
            0,
            image
        )
        
        return image
    
    def draw_label(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        label: str,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw a label above the bounding box."""
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            self.font,
            self.font_scale,
            self.font_thickness
        )
        
        label_x = bbox.x1
        label_y = bbox.y1 - 5
        
        if label_y - text_height < 0:
            label_y = bbox.y1 + text_height + 5
        
        bg_x1 = label_x
        bg_y1 = label_y - text_height - 2
        bg_x2 = label_x + text_width + 4
        bg_y2 = label_y + 2
        
        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1
        )
        
        text_color = get_contrasting_color(color)
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
