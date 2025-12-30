"""
Metrics Tracker - Handles TP/FP/FN tracking and IoU calculations.

Feature: chart-pattern-analysis-framework
Requirements: 9.1
"""

from typing import Dict, List, Tuple

from ..models import PatternType, PatternDetection, BoundingBox
from .models import MetricsEntry


class MetricsTracker:
    """
    Tracks detection metrics (TP, FP, FN) for pattern detection.
    
    Uses IoU (Intersection over Union) for matching predictions to ground truth.
    """
    
    DEFAULT_IOU_THRESHOLD = 0.5
    
    def __init__(self, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        """Initialize tracker with IoU threshold."""
        self.iou_threshold = iou_threshold
        self._metrics: Dict[PatternType, MetricsEntry] = {
            pt: MetricsEntry() for pt in PatternType
        }
    
    def record_evaluation(
        self,
        predictions: List[PatternDetection],
        ground_truth: List[PatternDetection]
    ) -> Dict[PatternType, MetricsEntry]:
        """
        Record evaluation results by comparing predictions to ground truth.
        
        Args:
            predictions: List of predicted pattern detections.
            ground_truth: List of ground truth pattern detections.
            
        Returns:
            Dictionary of metrics entries per pattern type for this evaluation.
        """
        # Group by pattern type
        pred_by_type: Dict[PatternType, List[PatternDetection]] = {}
        gt_by_type: Dict[PatternType, List[PatternDetection]] = {}
        
        for pred in predictions:
            if pred.pattern_type not in pred_by_type:
                pred_by_type[pred.pattern_type] = []
            pred_by_type[pred.pattern_type].append(pred)
        
        for gt in ground_truth:
            if gt.pattern_type not in gt_by_type:
                gt_by_type[gt.pattern_type] = []
            gt_by_type[gt.pattern_type].append(gt)
        
        # Calculate metrics per pattern type
        evaluation_metrics: Dict[PatternType, MetricsEntry] = {}
        all_pattern_types = set(pred_by_type.keys()) | set(gt_by_type.keys())
        
        for pattern_type in all_pattern_types:
            preds = pred_by_type.get(pattern_type, [])
            gts = gt_by_type.get(pattern_type, [])
            
            tp, fp, fn = self._calculate_tp_fp_fn(preds, gts)
            
            entry = MetricsEntry(
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn
            )
            evaluation_metrics[pattern_type] = entry
            
            # Update cumulative metrics
            self._metrics[pattern_type].true_positives += tp
            self._metrics[pattern_type].false_positives += fp
            self._metrics[pattern_type].false_negatives += fn
        
        return evaluation_metrics
    
    def _calculate_tp_fp_fn(
        self,
        predictions: List[PatternDetection],
        ground_truth: List[PatternDetection]
    ) -> Tuple[int, int, int]:
        """Calculate TP, FP, FN using IoU matching with greedy assignment."""
        if not predictions and not ground_truth:
            return 0, 0, 0
        
        if not predictions:
            return 0, 0, len(ground_truth)
        
        if not ground_truth:
            return 0, len(predictions), 0
        
        matched_gt = set()
        true_positives = 0
        
        # Sort predictions by confidence (highest first)
        sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
        
        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred.bounding_box, gt.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - len(matched_gt)
        
        return true_positives, false_positives, false_negatives
    
    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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
    
    def add_true_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add true positive count."""
        self._metrics[pattern_type].true_positives += count
    
    def add_false_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false positive count."""
        self._metrics[pattern_type].false_positives += count
    
    def add_false_negative(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false negative count."""
        self._metrics[pattern_type].false_negatives += count
    
    def get_metrics(self, pattern_type: PatternType) -> MetricsEntry:
        """Get metrics for a specific pattern type."""
        return self._metrics[pattern_type]
    
    def get_all_metrics(self) -> Dict[PatternType, MetricsEntry]:
        """Get metrics for all pattern types."""
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._metrics = {pt: MetricsEntry() for pt in PatternType}
    
    def set_metrics(self, metrics: Dict[PatternType, MetricsEntry]) -> None:
        """Set metrics from external source (for deserialization)."""
        self._metrics = metrics
