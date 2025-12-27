"""
Pattern Detector Agent using YOLOv8 - Replaces VLM-based detection.

Uses pretrained foduucom/stockmarket-pattern-detection-yolov8 model.
Detects: Head & Shoulders, M_Head (Double Top), W_Bottom (Double Bottom), Triangle
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Pattern name mapping from YOLO classes to our standard names
YOLO_PATTERN_MAPPING = {
    "Head and shoulders bottom": "inverse_head_shoulders",
    "Head and shoulders top": "head_shoulders",
    "M_Head": "double_top",
    "M_Head_Resistance": "double_top",
    "W_Bottom": "double_bottom",
    "W_Bottom_Support": "double_bottom",
    "Triangle": "triangle",
    "StockLine": "trendline",
}


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class YOLOPatternDetectorAgent:
    """
    Pattern detector using YOLOv8 pretrained on chart patterns.
    
    Replaces VLM-based PatternDetectorAgentLocal for more accurate
    pattern detection without hallucinations.
    """
    
    def __init__(self, confidence_threshold: float = 0.15):
        """
        Initialize YOLO pattern detector.
        
        Args:
            confidence_threshold: Minimum confidence for pattern detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model from HuggingFace."""
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
            
            logger.info("Loading YOLO stockmarket pattern detection model...")
            
            model_path = hf_hub_download(
                repo_id="foduucom/stockmarket-pattern-detection-yolov8",
                filename="model.pt"
            )
            
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded. Classes: {list(self.model.names.values())}")
            
        except ImportError as e:
            logger.error(f"YOLO dependencies not installed: {e}")
            logger.error("Install with: pip install ultralytics huggingface_hub")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def analyze(self, image_path: str, market_context: str = "") -> AgentResponse:
        """
        Analyze chart image for patterns using YOLO.
        
        Args:
            image_path: Path to chart image
            market_context: Additional context (ignored by YOLO but kept for compatibility)
            
        Returns:
            AgentResponse with detected patterns
        """
        if self.model is None:
            return AgentResponse(
                raw_text="YOLO model not loaded",
                parsed={
                    "pattern": "none",
                    "confidence": 0.0,
                    "pattern_box": None,
                    "description": "YOLO model not available",
                },
                success=False,
                error="YOLO model not loaded"
            )
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                image_path, 
                conf=self.confidence_threshold, 
                iou=0.45, 
                verbose=False
            )
            
            best_pattern = None
            best_confidence = 0.0
            best_box = None
            all_detections = []
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Map to standard pattern name
                    pattern_name = YOLO_PATTERN_MAPPING.get(cls_name, cls_name.lower())
                    
                    detection = {
                        "pattern": pattern_name,
                        "raw_class": cls_name,
                        "confidence": conf,
                        "box": (int(x1), int(y1), int(x2), int(y2))
                    }
                    all_detections.append(detection)
                    
                    if conf > best_confidence:
                        best_confidence = conf
                        best_pattern = pattern_name
                        best_box = (int(x1), int(y1), int(x2), int(y2))
            
            # Build response
            if best_pattern:
                raw_text = f"PATTERN: {best_pattern}\nCONFIDENCE: {best_confidence}\nPATTERN_BOX: {best_box}\nDESCRIPTION: Detected {best_pattern} pattern with {best_confidence:.1%} confidence using YOLO"
                
                parsed = {
                    "pattern": best_pattern,
                    "confidence": best_confidence,
                    "pattern_box": best_box,
                    "description": f"YOLO detected {best_pattern} with {best_confidence:.1%} confidence",
                    "all_detections": all_detections,
                }
                
                logger.info(f"YOLO detected: {best_pattern} ({best_confidence:.1%})")
                
            else:
                raw_text = "PATTERN: none\nCONFIDENCE: 0.0\nDESCRIPTION: No patterns detected by YOLO"
                
                parsed = {
                    "pattern": "none",
                    "confidence": 0.0,
                    "pattern_box": None,
                    "description": "No patterns detected by YOLO",
                    "all_detections": [],
                }
                
                logger.info("YOLO: No patterns detected")
            
            return AgentResponse(
                raw_text=raw_text,
                parsed=parsed,
                success=True
            )
            
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return AgentResponse(
                raw_text=f"Error: {str(e)}",
                parsed={
                    "pattern": "none",
                    "confidence": 0.0,
                    "pattern_box": None,
                    "description": f"YOLO error: {str(e)}",
                },
                success=False,
                error=str(e)
            )
    
    def get_annotated_chart(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Run detection and save annotated chart with bounding boxes.
        
        Args:
            image_path: Input chart image
            output_path: Output path for annotated image
            
        Returns:
            Path to annotated image or None if failed
        """
        if self.model is None:
            return None
        
        try:
            results = self.model.predict(
                image_path, 
                conf=self.confidence_threshold,
                verbose=False
            )
            
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_annotated{ext}"
            
            for result in results:
                result.save(filename=output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create annotated chart: {e}")
            return None


def get_pattern_detector_yolo(confidence_threshold: float = 0.15) -> YOLOPatternDetectorAgent:
    """Factory function for YOLO pattern detector."""
    return YOLOPatternDetectorAgent(confidence_threshold=confidence_threshold)


# For backwards compatibility
PatternDetectorAgentYOLO = YOLOPatternDetectorAgent
