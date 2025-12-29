"""
Hybrid pattern classifier implementation for chart pattern detection.

This module implements the PatternClassifier interface using a hybrid approach
that combines rule-based geometric detection with optional ML model integration.

Requirements:
- 3.1: Classify patterns into primary categories (reversal, continuation, bilateral)
- 3.2: Determine specific pattern type (e.g., head_shoulders, double_top, triangle)
- 3.3: Assign confidence score based on feature match quality
- 3.4: Rank detections by confidence and return all above threshold
- 3.5: Provide bounding box coordinates for each detection
- 3.6: Include pattern-specific metadata
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from ..models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ..models.enums import PatternCategory, PatternType
from .interfaces import PatternClassifier


logger = logging.getLogger(__name__)


class HybridPatternClassifier(PatternClassifier):
    """
    Pattern classifier combining rule-based and ML approaches.
    
    Uses geometric rules for initial detection, then optionally refines
    with ML model predictions. Detections are merged and filtered by
    confidence threshold.
    
    Requirements: 3.1-3.6
    """
    
    # Default configuration values
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5
    DEFAULT_TRIANGLE_CONVERGENCE_THRESHOLD = 0.3
    DEFAULT_DOUBLE_PATTERN_LEVEL_TOLERANCE = 20
    DEFAULT_HS_SHOULDER_TOLERANCE = 30
    DEFAULT_MIN_CANDLES_FOR_HS = 20
    
    def __init__(
        self,
        pattern_registry: Optional[Any] = None,
        ml_model_path: Optional[str] = None
    ):
        """
        Initialize the hybrid pattern classifier.
        
        Args:
            pattern_registry: Optional PatternRegistry for pattern definitions
            ml_model_path: Optional path to ML model (YOLO) for detection
        """
        self.registry = pattern_registry
        self.ml_model = self._load_ml_model(ml_model_path) if ml_model_path else None
        self._config: Dict[str, Any] = {}
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this classifier."""
        return "hybrid_classifier_v1"
    
    def process(
        self,
        input_data: Tuple[FeatureMap, np.ndarray],
        config: Dict[str, Any]
    ) -> List[PatternDetection]:
        """
        Process features and image to detect patterns.
        
        Args:
            input_data: Tuple of (FeatureMap, image array)
            config: Configuration dictionary
            
        Returns:
            List of PatternDetection objects, ordered by confidence descending
        """
        if not self.validate_input(input_data):
            return []
        
        features, image = input_data
        self._config = config
        return self.classify(features, image)
    
    def classify(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """
        Classify patterns using hybrid approach.
        
        Combines rule-based detection with optional ML detection,
        merges overlapping detections, and filters by confidence.
        
        Args:
            features: Extracted features from FeatureExtractor
            image: Preprocessed image for additional analysis
            
        Returns:
            List of PatternDetection objects, ordered by confidence descending
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
        """
        detections: List[PatternDetection] = []
        
        # 1. Rule-based detection
        rule_detections = self._rule_based_detection(features, image)
        detections.extend(rule_detections)
        
        # 2. ML-based detection (if model available)
        if self.ml_model is not None:
            ml_detections = self._ml_detection(image)
            detections.extend(ml_detections)
        
        # 3. Merge overlapping detections
        merged = self._merge_detections(detections)
        
        # 4. Filter by confidence threshold
        threshold = self._config.get(
            "confidence_threshold",
            self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        filtered = [d for d in merged if d.confidence >= threshold]
        
        # 5. Apply Recency Filter (Req: Analyze only last X%)
        # Default min_end_x_ratio = 0.0 (analyze everything)
        # If set to 0.8, only patterns ending in the last 20% of image width are kept.
        min_end_x_ratio = self._config.get("min_end_x_ratio", 0.0)
        if min_end_x_ratio > 0.0:
            h, w = image.shape[:2]
            min_x = w * (1.0 - min_end_x_ratio)
            # Keep patterns where the *end* (x2) is after the cutoff
            filtered = [d for d in filtered if d.bounding_box.x2 >= min_x]
        
        # 6. Sort by confidence descending (Requirements 3.4)
        sorted_detections = sorted(
            filtered,
            key=lambda d: d.confidence,
            reverse=True
        )
        
        return sorted_detections
    
    def get_supported_patterns(self) -> List[PatternType]:
        """Return list of patterns this classifier can detect."""
        return list(PatternType)
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate that input is a tuple of (FeatureMap, ndarray)."""
        if not isinstance(input_data, tuple):
            return False
        if len(input_data) != 2:
            return False
        features, image = input_data
        if not isinstance(features, FeatureMap):
            return False
        if not isinstance(image, np.ndarray):
            return False
        return True
    
    # =========================================================================
    # Rule-Based Detection Methods
    # =========================================================================
    
    def _rule_based_detection(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Apply geometric rules to detect patterns."""
        detections: List[PatternDetection] = []
        
        # Detect triangles from converging trendlines
        triangles = self._detect_triangles(features.trendlines, image)
        detections.extend(triangles)
        
        # Detect double tops/bottoms from support/resistance
        doubles = self._detect_double_patterns(
            features.support_zones,
            features.resistance_zones,
            features.candlestick_regions,
            image
        )
        detections.extend(doubles)
        
        # Detect head and shoulders from peak analysis
        hs_patterns = self._detect_head_shoulders(
            features.candlestick_regions,
            image
        )
        detections.extend(hs_patterns)
        
        return detections

    
    # =========================================================================
    # Task 9.1: Triangle Detection
    # =========================================================================
    
    def _detect_triangles(
        self,
        trendlines: List[Dict[str, Any]],
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect triangle patterns from converging trendlines."""
        detections: List[PatternDetection] = []
        
        if len(trendlines) < 2:
            return detections
        
        up_lines = [t for t in trendlines if t.get("direction") == "up"]
        down_lines = [t for t in trendlines if t.get("direction") == "down"]
        
        for up_line in up_lines:
            for down_line in down_lines:
                if self._lines_converge(up_line, down_line, image):
                    pattern_type, category = self._classify_triangle_type(up_line, down_line)
                    bbox = self._calculate_triangle_bbox(up_line, down_line)
                    
                    if not bbox.is_valid():
                        continue
                    
                    confidence = self._calculate_triangle_confidence(up_line, down_line, image)
                    
                    detection = PatternDetection(
                        pattern_type=pattern_type,
                        category=category,
                        confidence=confidence,
                        bounding_box=bbox,
                        metadata={
                            "upper_line": down_line,
                            "lower_line": up_line,
                            "apex": self._calculate_apex(up_line, down_line)
                        },
                        detector_id="rule_triangle"
                    )
                    detections.append(detection)
        
        return detections
    
    def _lines_converge(self, line1: Dict[str, Any], line2: Dict[str, Any], image: np.ndarray) -> bool:
        """Check if two lines converge (form a triangle pattern)."""
        if line1.get("direction") == line2.get("direction"):
            return False
        
        x1_1, y1_1 = line1["start"]
        x1_2, y1_2 = line1["end"]
        x2_1, y2_1 = line2["start"]
        x2_2, y2_2 = line2["end"]
        
        intersection = self._line_intersection(
            (x1_1, y1_1), (x1_2, y1_2),
            (x2_1, y2_1), (x2_2, y2_2)
        )
        
        if intersection is None:
            return False
        
        ix, iy = intersection
        h, w = image.shape[:2]
        margin = max(w, h) * 0.5
        
        if ix < -margin or ix > w + margin:
            return False
        if iy < -margin or iy > h + margin:
            return False
        
        min_x = min(x1_1, x1_2, x2_1, x2_2)
        if ix < min_x:
            return False
        
        return True
    
    def _line_intersection(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (ix, iy)
    
    def _classify_triangle_type(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any]
    ) -> Tuple[PatternType, PatternCategory]:
        """Classify triangle type based on trendline angles."""
        up_angle = abs(up_line.get("angle", 0))
        down_angle = abs(down_line.get("angle", 0))
        angle_threshold = 10
        
        if down_angle < angle_threshold and up_angle >= angle_threshold:
            return PatternType.ASCENDING_TRIANGLE, PatternCategory.CONTINUATION
        if up_angle < angle_threshold and down_angle >= angle_threshold:
            return PatternType.DESCENDING_TRIANGLE, PatternCategory.CONTINUATION
        return PatternType.SYMMETRICAL_TRIANGLE, PatternCategory.BILATERAL
    
    def _calculate_triangle_bbox(self, up_line: Dict[str, Any], down_line: Dict[str, Any]) -> BoundingBox:
        """Calculate bounding box for triangle pattern."""
        all_x = [up_line["start"][0], up_line["end"][0], down_line["start"][0], down_line["end"][0]]
        all_y = [up_line["start"][1], up_line["end"][1], down_line["start"][1], down_line["end"][1]]
        
        x1, y1 = int(min(all_x)), int(min(all_y))
        x2, y2 = int(max(all_x)), int(max(all_y))
        
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        
        return BoundingBox(x1, y1, x2, y2)
    
    def _calculate_triangle_confidence(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any],
        image: np.ndarray
    ) -> float:
        """Calculate confidence score for triangle detection."""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        up_length = up_line.get("length", 0)
        down_length = down_line.get("length", 0)
        avg_length = (up_length + down_length) / 2
        length_factor = min(avg_length / (max_dim * 0.3), 1.0)
        
        up_angle = abs(up_line.get("angle", 0))
        down_angle = abs(down_line.get("angle", 0))
        angle_factor = min((up_angle + down_angle) / 60, 1.0)
        
        confidence = 0.4 * length_factor + 0.4 * angle_factor + 0.2
        return min(max(confidence, 0.0), 0.9)
    
    def _calculate_apex(self, up_line: Dict[str, Any], down_line: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate the apex (convergence point) of the triangle."""
        intersection = self._line_intersection(
            up_line["start"], up_line["end"],
            down_line["start"], down_line["end"]
        )
        if intersection:
            return {"x": intersection[0], "y": intersection[1]}
        return None

    
    # =========================================================================
    # Task 9.2: Double Top/Bottom Detection
    # =========================================================================
    
    def _detect_double_patterns(
        self,
        support_zones: List[BoundingBox],
        resistance_zones: List[BoundingBox],
        candlesticks: List[BoundingBox],
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect double top and double bottom patterns."""
        detections: List[PatternDetection] = []
        
        level_tolerance = self._config.get(
            "double_pattern_level_tolerance",
            self.DEFAULT_DOUBLE_PATTERN_LEVEL_TOLERANCE
        )
        
        double_bottoms = self._find_double_touches(
            support_zones, level_tolerance, PatternType.DOUBLE_BOTTOM,
            PatternCategory.REVERSAL, image
        )
        detections.extend(double_bottoms)
        
        double_tops = self._find_double_touches(
            resistance_zones, level_tolerance, PatternType.DOUBLE_TOP,
            PatternCategory.REVERSAL, image
        )
        detections.extend(double_tops)
        
        return detections
    
    def _find_double_touches(
        self,
        zones: List[BoundingBox],
        level_tolerance: int,
        pattern_type: PatternType,
        category: PatternCategory,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Find pairs of zones at similar levels (double touches)."""
        detections: List[PatternDetection] = []
        
        if len(zones) < 2:
            return detections
        
        h, w = image.shape[:2]
        
        for i, zone1 in enumerate(zones):
            for zone2 in zones[i + 1:]:
                y1_center = (zone1.y1 + zone1.y2) // 2
                y2_center = (zone2.y1 + zone2.y2) // 2
                
                if abs(y1_center - y2_center) <= level_tolerance:
                    x_separation = abs(zone1.x1 - zone2.x1)
                    min_separation = w * 0.1
                    
                    if x_separation >= min_separation:
                        bbox = BoundingBox(
                            min(zone1.x1, zone2.x1),
                            min(zone1.y1, zone2.y1),
                            max(zone1.x2, zone2.x2),
                            max(zone1.y2, zone2.y2)
                        )
                        
                        if not bbox.is_valid():
                            continue
                        
                        level_diff = abs(y1_center - y2_center)
                        level_similarity = 1.0 - (level_diff / max(level_tolerance, 1))
                        confidence = 0.5 + 0.3 * level_similarity
                        
                        detection = PatternDetection(
                            pattern_type=pattern_type,
                            category=category,
                            confidence=min(confidence, 0.9),
                            bounding_box=bbox,
                            metadata={
                                "touch_1": zone1.to_dict(),
                                "touch_2": zone2.to_dict(),
                                "level_difference": level_diff,
                                "neckline_y": (y1_center + y2_center) // 2
                            },
                            detector_id="rule_double"
                        )
                        detections.append(detection)
        
        return detections

    
    # =========================================================================
    # Task 9.3: Head and Shoulders Detection
    # =========================================================================
    
    def _detect_head_shoulders(
        self,
        candlesticks: List[BoundingBox],
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect head and shoulders patterns from candlestick peaks."""
        detections: List[PatternDetection] = []
        
        min_candles = self._config.get("min_candles_for_hs", self.DEFAULT_MIN_CANDLES_FOR_HS)
        
        if len(candlesticks) < min_candles:
            return detections
        
        shoulder_tolerance = self._config.get("hs_shoulder_tolerance", self.DEFAULT_HS_SHOULDER_TOLERANCE)
        
        peaks = self._find_local_peaks(candlesticks, is_maxima=True)
        hs_top = self._find_hs_pattern(
            peaks, candlesticks, shoulder_tolerance,
            PatternType.HEAD_SHOULDERS, is_inverted=False, image=image
        )
        detections.extend(hs_top)
        
        troughs = self._find_local_peaks(candlesticks, is_maxima=False)
        hs_bottom = self._find_hs_pattern(
            troughs, candlesticks, shoulder_tolerance,
            PatternType.INVERSE_HEAD_SHOULDERS, is_inverted=True, image=image
        )
        detections.extend(hs_bottom)
        
        return detections
    
    def _find_local_peaks(
        self,
        candlesticks: List[BoundingBox],
        is_maxima: bool,
        window: int = 2
    ) -> List[Tuple[int, BoundingBox]]:
        """Find local maxima or minima in candlestick positions."""
        peaks: List[Tuple[int, BoundingBox]] = []
        n = len(candlesticks)
        
        for i in range(window, n - window):
            current = candlesticks[i]
            
            if is_maxima:
                current_y = current.y1
                is_peak = all(
                    current_y < candlesticks[j].y1
                    for j in range(i - window, i + window + 1)
                    if j != i
                )
            else:
                current_y = current.y2
                is_peak = all(
                    current_y > candlesticks[j].y2
                    for j in range(i - window, i + window + 1)
                    if j != i
                )
            
            if is_peak:
                peaks.append((i, current))
        
        return peaks
    
    def _find_hs_pattern(
        self,
        peaks: List[Tuple[int, BoundingBox]],
        candlesticks: List[BoundingBox],
        shoulder_tolerance: int,
        pattern_type: PatternType,
        is_inverted: bool,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Find head and shoulders pattern from peaks."""
        detections: List[PatternDetection] = []
        
        if len(peaks) < 3:
            return detections
        
        for i in range(len(peaks) - 2):
            left_idx, left_shoulder = peaks[i]
            head_idx, head = peaks[i + 1]
            right_idx, right_shoulder = peaks[i + 2]
            
            if is_inverted:
                left_y, head_y, right_y = left_shoulder.y2, head.y2, right_shoulder.y2
                head_is_extreme = head_y > left_y and head_y > right_y
            else:
                left_y, head_y, right_y = left_shoulder.y1, head.y1, right_shoulder.y1
                head_is_extreme = head_y < left_y and head_y < right_y
            
            if not head_is_extreme:
                continue
            
            shoulder_diff = abs(left_y - right_y)
            if shoulder_diff > shoulder_tolerance:
                continue
            
            bbox = BoundingBox(
                left_shoulder.x1,
                min(left_shoulder.y1, head.y1, right_shoulder.y1),
                right_shoulder.x2,
                max(left_shoulder.y2, head.y2, right_shoulder.y2)
            )
            
            if not bbox.is_valid():
                continue
            
            confidence = self._calculate_hs_confidence(
                left_shoulder, head, right_shoulder,
                shoulder_diff, shoulder_tolerance, is_inverted
            )
            
            neckline_y = (left_y + right_y) // 2
            
            detection = PatternDetection(
                pattern_type=pattern_type,
                category=PatternCategory.REVERSAL,
                confidence=confidence,
                bounding_box=bbox,
                metadata={
                    "left_shoulder": left_shoulder.to_dict(),
                    "head": head.to_dict(),
                    "right_shoulder": right_shoulder.to_dict(),
                    "neckline_y": neckline_y,
                    "shoulder_difference": shoulder_diff
                },
                detector_id="rule_hs"
            )
            detections.append(detection)
        
        return detections
    
    def _calculate_hs_confidence(
        self,
        left_shoulder: BoundingBox,
        head: BoundingBox,
        right_shoulder: BoundingBox,
        shoulder_diff: int,
        shoulder_tolerance: int,
        is_inverted: bool
    ) -> float:
        """Calculate confidence score for head and shoulders pattern."""
        symmetry_factor = 1.0 - (shoulder_diff / max(shoulder_tolerance, 1))
        
        if is_inverted:
            left_y, head_y, right_y = left_shoulder.y2, head.y2, right_shoulder.y2
            avg_shoulder_y = (left_y + right_y) / 2
            head_prominence = head_y - avg_shoulder_y
        else:
            left_y, head_y, right_y = left_shoulder.y1, head.y1, right_shoulder.y1
            avg_shoulder_y = (left_y + right_y) / 2
            head_prominence = avg_shoulder_y - head_y
        
        prominence_factor = min(head_prominence / 50, 1.0) if head_prominence > 0 else 0.0
        confidence = 0.5 + 0.25 * symmetry_factor + 0.15 * prominence_factor
        
        return min(max(confidence, 0.0), 0.9)

    
    # =========================================================================
    # Task 9.4: ML Model Integration (YOLO)
    # =========================================================================
    
    def _load_ml_model(self, path: str) -> Optional[Any]:
        """Load ML model from path."""
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            logger.info(f"Loaded ML model from {path}")
            return model
        except ImportError:
            logger.warning("ultralytics not installed, ML detection disabled")
            return None
        except Exception as e:
            logger.warning(f"Failed to load ML model from {path}: {e}")
            return None
    
    def _ml_detection(self, image: np.ndarray) -> List[PatternDetection]:
        """Run ML model for pattern detection."""
        if self.ml_model is None:
            return []
        
        try:
            results = self.ml_model.predict(
                image,
                verbose=False,
                conf=self._config.get("ml_confidence_threshold", 0.25)
            )
            return self._convert_ml_results(results)
        except Exception as e:
            logger.warning(f"ML detection failed: {e}")
            return []
    
    def _convert_ml_results(self, results: Any) -> List[PatternDetection]:
        """Convert ML model results to PatternDetection objects."""
        detections: List[PatternDetection] = []
        
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue
            
            for box in result.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    pattern_type = self._map_ml_class(cls_id, result.names)
                    if pattern_type is None:
                        continue
                    
                    category = self._get_category_for_pattern(pattern_type)
                    
                    detection = PatternDetection(
                        pattern_type=pattern_type,
                        category=category,
                        confidence=conf,
                        bounding_box=BoundingBox(x1, y1, x2, y2),
                        metadata={
                            "ml_class": result.names.get(cls_id, "unknown"),
                            "ml_class_id": cls_id
                        },
                        detector_id="ml_yolo"
                    )
                    detections.append(detection)
                except Exception as e:
                    logger.debug(f"Failed to convert ML result: {e}")
                    continue
        
        return detections
    
    def _map_ml_class(self, cls_id: int, names: Dict[int, str]) -> Optional[PatternType]:
        """Map ML class ID to PatternType enum."""
        class_name = names.get(cls_id, "").lower()
        
        mapping = {
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
        
        return mapping.get(class_name)
    
    def _get_category_for_pattern(self, pattern_type: PatternType) -> PatternCategory:
        """Get the category for a given pattern type."""
        reversal_patterns = {
            PatternType.HEAD_SHOULDERS,
            PatternType.INVERSE_HEAD_SHOULDERS,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM,
            PatternType.TRIPLE_TOP,
            PatternType.TRIPLE_BOTTOM,
        }
        
        bilateral_patterns = {
            PatternType.SYMMETRICAL_TRIANGLE,
        }
        
        if pattern_type in reversal_patterns:
            return PatternCategory.REVERSAL
        elif pattern_type in bilateral_patterns:
            return PatternCategory.BILATERAL
        else:
            return PatternCategory.CONTINUATION

    
    # =========================================================================
    # Task 9.5: Merge Detections
    # =========================================================================
    
    def _merge_detections(self, detections: List[PatternDetection]) -> List[PatternDetection]:
        """
        Merge overlapping detections, keeping highest confidence.
        
        Uses IoU (Intersection over Union) to identify overlapping
        detections and keeps only the one with highest confidence.
        
        Requirements: 3.4
        """
        if not detections:
            return []
        
        iou_threshold = self._config.get("iou_threshold", self.DEFAULT_IOU_THRESHOLD)
        
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        merged: List[PatternDetection] = []
        
        for det in sorted_dets:
            overlaps = False
            for existing in merged:
                iou = self._calculate_iou(det.bounding_box, existing.bounding_box)
                if iou > iou_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(det)
        
        return merged
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area()
        area2 = box2.area()
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
