"""
Detection merger - merges overlapping detections.
"""

from typing import List

from ...models.dataclasses import BoundingBox, PatternDetection


class DetectionMerger:
    """Merges overlapping pattern detections."""
    
    DEFAULT_IOU_THRESHOLD = 0.5
    
    def __init__(self, iou_threshold: float = None):
        self.iou_threshold = iou_threshold or self.DEFAULT_IOU_THRESHOLD
    
    def merge(self, detections: List[PatternDetection]) -> List[PatternDetection]:
        """
        Merge overlapping detections, keeping highest confidence.
        
        Uses IoU (Intersection over Union) to identify overlaps.
        """
        if not detections:
            return []
        
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        merged: List[PatternDetection] = []
        
        for det in sorted_dets:
            overlaps = False
            for existing in merged:
                iou = self._calculate_iou(det.bounding_box, existing.bounding_box)
                if iou > self.iou_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(det)
        
        return merged
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area() + box2.area() - intersection
        
        return intersection / union if union > 0 else 0.0
