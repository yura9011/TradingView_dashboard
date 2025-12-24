"""
Chart Annotator - Visual annotation using PIL.
Draws zones, lines, labels on chart screenshots.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default colors (RGBA)
COLORS = {
    "support": (0, 200, 100, 100),      # Green transparent
    "resistance": (200, 50, 50, 100),    # Red transparent
    "fibonacci": (100, 100, 255, 80),    # Blue transparent
    "pattern": (255, 200, 0, 120),       # Yellow transparent
    "trend_up": (0, 255, 0, 255),        # Green solid
    "trend_down": (255, 0, 0, 255),      # Red solid
    "neutral": (128, 128, 128, 255),     # Gray solid
    "text": (255, 255, 255, 255),        # White
    "text_bg": (0, 0, 0, 180),           # Dark background
}


@dataclass
class AnnotationLayer:
    """A layer of annotations to draw on chart."""
    zones: List[Tuple[int, int, int, int, str]] = field(default_factory=list)  # (x1, y1, x2, y2, type)
    lines: List[Tuple[int, int, int, int, str]] = field(default_factory=list)  # (x1, y1, x2, y2, color)
    labels: List[Tuple[int, int, str, str]] = field(default_factory=list)       # (x, y, text, color)
    arrows: List[Tuple[int, int, int, int, str]] = field(default_factory=list)  # (x1, y1, x2, y2, color)


class ChartAnnotator:
    """Annotates chart images with analysis results."""
    
    def __init__(self, font_size: int = 16):
        """Initialize annotator.
        
        Args:
            font_size: Base font size for labels
        """
        self.font_size = font_size
        self._font = None
        self._font_small = None
    
    def _get_font(self, size: int = None):
        """Get PIL font, fallback to default if custom not available."""
        size = size or self.font_size
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
            except:
                return ImageFont.load_default()
    
    def annotate(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        fibonacci_levels: Optional[dict] = None,
        pattern_box: Optional[Tuple[int, int, int, int]] = None,
        pattern_name: Optional[str] = None,
        trend: Optional[str] = None,
        signal_type: Optional[str] = None,
        analysis_summary: Optional[str] = None,
    ) -> Path:
        """Annotate a chart image with analysis results.
        
        Args:
            image_path: Path to original chart image
            output_path: Where to save annotated image (default: adds _annotated suffix)
            support_level: Support price level (draws line at percentage of height)
            resistance_level: Resistance price level
            fibonacci_levels: Dict of fib levels to mark
            pattern_box: (x1, y1, x2, y2) to highlight pattern
            pattern_name: Name of detected pattern
            trend: Current trend (up/down/sideways)
            signal_type: Signal classification
            analysis_summary: Short summary text
            
        Returns:
            Path to annotated image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size
        
        # Create transparent overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw support/resistance zones
        if support_level is not None:
            # Map support to Y coordinate (assume it's a percentage 0-100)
            y = int(height * (1 - support_level / 100)) if support_level <= 100 else int(height * 0.7)
            self._draw_zone(draw, 0, y - 10, width, y + 10, "support")
            self._draw_label(draw, 10, y - 25, f"Support", "support")
        
        if resistance_level is not None:
            y = int(height * (1 - resistance_level / 100)) if resistance_level <= 100 else int(height * 0.3)
            self._draw_zone(draw, 0, y - 10, width, y + 10, "resistance")
            self._draw_label(draw, 10, y + 15, f"Resistance", "resistance")
        
        # Draw fibonacci levels
        if fibonacci_levels:
            for level_name, y_percent in fibonacci_levels.items():
                if isinstance(y_percent, (int, float)):
                    y = int(height * (1 - y_percent / 100))
                    color = COLORS["fibonacci"]
                    draw.line([(0, y), (width, y)], fill=color[:3] + (150,), width=1)
        
        # Draw pattern with detailed shape
        if pattern_box:
            self._draw_pattern(draw, pattern_box, pattern_name, width, height)
        
        # Draw trend indicator
        if trend:
            trend_color = "trend_up" if trend == "up" else ("trend_down" if trend == "down" else "neutral")
            trend_text = f"Trend: {trend.upper()}"
            self._draw_label(draw, width - 150, 10, trend_text, trend_color)
        
        # Draw signal badge
        if signal_type:
            badge_color = "trend_up" if signal_type == "candidate" else "trend_down"
            self._draw_badge(draw, width - 150, 50, signal_type.upper(), badge_color)
        
        # Draw summary at bottom
        if analysis_summary:
            summary_short = analysis_summary[:100] + "..." if len(analysis_summary) > 100 else analysis_summary
            self._draw_text_box(draw, 10, height - 60, summary_short, width - 20)
        
        # Composite overlay onto image
        img = Image.alpha_composite(img, overlay)
        
        # Convert to RGB for saving as JPEG/PNG
        img = img.convert("RGB")
        
        # Determine output path
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
        else:
            output_path = Path(output_path)
        
        img.save(output_path)
        logger.info(f"Saved annotated chart to {output_path}")
        
        return output_path
    
    def _draw_zone(
        self,
        draw: ImageDraw.Draw,
        x1: int, y1: int, x2: int, y2: int,
        zone_type: str,
    ):
        """Draw a semi-transparent zone."""
        color = COLORS.get(zone_type, COLORS["neutral"])
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    def _draw_label(
        self,
        draw: ImageDraw.Draw,
        x: int, y: int,
        text: str,
        color_key: str,
    ):
        """Draw a text label with background."""
        font = self._get_font()
        color = COLORS.get(color_key, COLORS["text"])
        
        # Get text size
        bbox = draw.textbbox((x, y), text, font=font)
        padding = 4
        
        # Draw background
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=COLORS["text_bg"]
        )
        
        # Draw text
        draw.text((x, y), text, fill=color[:3] + (255,), font=font)
    
    def _draw_badge(
        self,
        draw: ImageDraw.Draw,
        x: int, y: int,
        text: str,
        color_key: str,
    ):
        """Draw a colored badge."""
        font = self._get_font(14)
        color = COLORS.get(color_key, COLORS["neutral"])
        
        bbox = draw.textbbox((x, y), text, font=font)
        padding = 6
        
        # Draw badge background
        draw.rounded_rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            radius=5,
            fill=color[:3] + (200,)
        )
        
        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    def _draw_text_box(
        self,
        draw: ImageDraw.Draw,
        x: int, y: int,
        text: str,
        max_width: int,
    ):
        """Draw text in a box at the bottom."""
        font = self._get_font(12)
        padding = 8
        
        # Draw background box
        draw.rectangle(
            [x, y, x + max_width, y + 50],
            fill=COLORS["text_bg"]
        )
        
        # Draw text
        draw.text((x + padding, y + padding), text, fill=COLORS["text"], font=font)
    
    def _draw_pattern(
        self,
        draw: ImageDraw.Draw,
        pattern_box: Tuple[int, int, int, int],
        pattern_name: Optional[str],
        img_width: int,
        img_height: int,
    ):
        """Draw pattern shape with annotations based on pattern type.
        
        Args:
            draw: PIL ImageDraw object
            pattern_box: (x1, y1, x2, y2) as percentages 0-100
            pattern_name: Name of the detected pattern
            img_width: Image width in pixels
            img_height: Image height in pixels
        """
        # Convert percentage to pixels
        x1 = int(pattern_box[0] * img_width / 100)
        y1 = int(pattern_box[1] * img_height / 100)
        x2 = int(pattern_box[2] * img_width / 100)
        y2 = int(pattern_box[3] * img_height / 100)
        
        pattern_lower = (pattern_name or "").lower()
        color = COLORS["pattern"]
        line_color = color[:3] + (200,)
        line_width = 3
        
        # Draw bounding rectangle first (semi-transparent)
        self._draw_zone(draw, x1, y1, x2, y2, "pattern")
        
        # Calculate pattern dimensions
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        
        # Draw pattern-specific shapes
        if "double bottom" in pattern_lower or "w pattern" in pattern_lower:
            # Draw W shape for double bottom
            points = [
                (x1, y1),                 # Start top-left
                (x1 + w//4, y2),          # First bottom
                (cx, cy),                 # Middle peak
                (x1 + 3*w//4, y2),        # Second bottom
                (x2, y1),                 # End top-right
            ]
            draw.line(points, fill=line_color, width=line_width)
            # Draw horizontal support line
            draw.line([(x1, y2), (x2, y2)], fill=COLORS["support"][:3] + (200,), width=2)
            
        elif "double top" in pattern_lower or "m pattern" in pattern_lower:
            # Draw M shape for double top
            points = [
                (x1, y2),                 # Start bottom-left
                (x1 + w//4, y1),          # First top
                (cx, cy),                 # Middle dip
                (x1 + 3*w//4, y1),        # Second top
                (x2, y2),                 # End bottom-right
            ]
            draw.line(points, fill=line_color, width=line_width)
            # Draw horizontal resistance line
            draw.line([(x1, y1), (x2, y1)], fill=COLORS["resistance"][:3] + (200,), width=2)
            
        elif "head" in pattern_lower and "shoulder" in pattern_lower:
            # Draw head and shoulders shape
            points = [
                (x1, y2),                     # Left shoulder base
                (x1 + w//6, y1 + h//3),       # Left shoulder top
                (x1 + w//3, y2),              # Neckline left
                (cx, y1),                     # Head top
                (x1 + 2*w//3, y2),            # Neckline right
                (x1 + 5*w//6, y1 + h//3),     # Right shoulder top
                (x2, y2),                     # Right shoulder base
            ]
            draw.line(points, fill=line_color, width=line_width)
            # Draw neckline
            draw.line([(x1 + w//3, y2), (x1 + 2*w//3, y2)], fill=COLORS["resistance"][:3] + (200,), width=2)
            
        elif "triangle" in pattern_lower:
            # Draw triangle converging lines
            if "ascending" in pattern_lower:
                draw.line([(x1, y2), (x2, cy)], fill=line_color, width=line_width)  # Rising support
                draw.line([(x1, y1 + h//3), (x2, y1 + h//3)], fill=line_color, width=line_width)  # Flat resistance
            elif "descending" in pattern_lower:
                draw.line([(x1, y2 - h//3), (x2, y2 - h//3)], fill=line_color, width=line_width)  # Flat support
                draw.line([(x1, y1), (x2, cy)], fill=line_color, width=line_width)  # Falling resistance
            else:
                # Symmetrical triangle
                draw.line([(x1, y2), (x2, cy)], fill=line_color, width=line_width)  # Rising support
                draw.line([(x1, y1), (x2, cy)], fill=line_color, width=line_width)  # Falling resistance
        
        elif "engulfing" in pattern_lower:
            # Draw rectangle around the engulfing candles
            inner_margin = 5
            draw.rectangle(
                [x1 + inner_margin, y1 + inner_margin, x2 - inner_margin, y2 - inner_margin],
                outline=line_color,
                width=3
            )
            # Add arrow pointing up or down
            if "bullish" in pattern_lower:
                # Upward arrow
                arrow_x = x2 + 10
                draw.polygon([(arrow_x, y1), (arrow_x - 10, y1 + 20), (arrow_x + 10, y1 + 20)], 
                           fill=COLORS["trend_up"])
            else:
                # Downward arrow  
                arrow_x = x2 + 10
                draw.polygon([(arrow_x, y2), (arrow_x - 10, y2 - 20), (arrow_x + 10, y2 - 20)],
                           fill=COLORS["trend_down"])
        
        elif "wedge" in pattern_lower:
            # Draw wedge pattern
            if "rising" in pattern_lower or "falling" not in pattern_lower:
                draw.line([(x1, y2), (x2, y1 + h//3)], fill=line_color, width=line_width)
                draw.line([(x1, y1 + h//2), (x2, y1)], fill=line_color, width=line_width)
            else:
                draw.line([(x1, y1), (x2, y2 - h//3)], fill=line_color, width=line_width)
                draw.line([(x1, y1 + h//2), (x2, y2)], fill=line_color, width=line_width)
        
        else:
            # Default: just draw a highlighted rectangle with corners
            corner_len = min(w, h) // 4
            # Top-left corner
            draw.line([(x1, y1 + corner_len), (x1, y1), (x1 + corner_len, y1)], fill=line_color, width=3)
            # Top-right corner  
            draw.line([(x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len)], fill=line_color, width=3)
            # Bottom-left corner
            draw.line([(x1, y2 - corner_len), (x1, y2), (x1 + corner_len, y2)], fill=line_color, width=3)
            # Bottom-right corner
            draw.line([(x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len)], fill=line_color, width=3)
        
        # Draw pattern name label
        if pattern_name:
            label = pattern_name.replace("_", " ").title()
            self._draw_label(draw, x1, max(y1 - 30, 5), f"📊 {label}", "pattern")


def get_annotator() -> ChartAnnotator:
    """Get ChartAnnotator instance."""
    return ChartAnnotator()

