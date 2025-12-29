"""
MetricsCollector for tracking and evaluating pattern detection accuracy.

Provides:
- Tracking of True Positives (TP), False Positives (FP), False Negatives (FN)
- Calculation of precision, recall, and F1 scores
- Historical metrics storage and reporting
- Comparison between model versions

Feature: chart-pattern-analysis-framework
Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import os

from ..models import PatternType, PatternDetection, BoundingBox


@dataclass
class MetricsEntry:
    """Single metrics entry for a pattern type."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def precision(self) -> float:
        """
        Calculate precision: TP / (TP + FP).
        
        Returns 0.0 if there are no positive predictions (TP + FP = 0).
        """
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def recall(self) -> float:
        """
        Calculate recall: TP / (TP + FN).
        
        Returns 0.0 if there are no actual positives (TP + FN = 0).
        """
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def f1_score(self) -> float:
        """
        Calculate F1 score: 2 * (precision * recall) / (precision + recall).
        
        Returns 0.0 if both precision and recall are 0.
        """
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def total_predictions(self) -> int:
        """Total number of predictions made (TP + FP)."""
        return self.true_positives + self.false_positives
    
    def total_actual(self) -> int:
        """Total number of actual positives (TP + FN)."""
        return self.true_positives + self.false_negatives
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsEntry":
        """Create MetricsEntry from dictionary."""
        return cls(
            true_positives=data.get("true_positives", 0),
            false_positives=data.get("false_positives", 0),
            false_negatives=data.get("false_negatives", 0)
        )
    
    def __add__(self, other: "MetricsEntry") -> "MetricsEntry":
        """Add two MetricsEntry objects together."""
        return MetricsEntry(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives
        )


@dataclass
class HistoricalRecord:
    """Historical record of metrics at a point in time."""
    timestamp: str
    version: str
    metrics_by_pattern: Dict[str, Dict[str, Any]]
    aggregate_metrics: Dict[str, Any]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "version": self.version,
            "metrics_by_pattern": self.metrics_by_pattern,
            "aggregate_metrics": self.aggregate_metrics,
            "config_snapshot": self.config_snapshot
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalRecord":
        """Create HistoricalRecord from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            version=data["version"],
            metrics_by_pattern=data["metrics_by_pattern"],
            aggregate_metrics=data["aggregate_metrics"],
            config_snapshot=data.get("config_snapshot", {})
        )


class ReportFormat(Enum):
    """Report output formats."""
    SUMMARY = "summary"
    DETAILED = "detailed"



class MetricsCollector:
    """
    Collector for tracking and evaluating pattern detection accuracy.
    
    Tracks TP/FP/FN per pattern type, calculates precision/recall/F1,
    maintains historical metrics, and generates reports.
    
    Requirements:
    - 9.1: Track true positives, false positives, and false negatives
    - 9.2: Calculate precision, recall, and F1 score per pattern
    - 9.3: Maintain historical metrics for trend analysis
    - 9.4: Generate reports in summary and detailed formats
    - 9.5: Support comparison between different model versions
    """
    
    DEFAULT_IOU_THRESHOLD = 0.5
    
    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        version: str = "v1.0.0",
        storage_path: Optional[str] = None
    ):
        """
        Initialize MetricsCollector.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth.
            version: Current model/configuration version identifier.
            storage_path: Optional path for persisting historical metrics.
        """
        self.iou_threshold = iou_threshold
        self.version = version
        self.storage_path = storage_path
        
        # Metrics by pattern type
        self._metrics: Dict[PatternType, MetricsEntry] = {
            pt: MetricsEntry() for pt in PatternType
        }
        
        # Historical records
        self._history: List[HistoricalRecord] = []
        
        # Load existing history if storage path provided
        if storage_path and os.path.exists(storage_path):
            self._load_history()
    
    # =========================================================================
    # Tracking Methods (Requirement 9.1, 9.3)
    # =========================================================================
    
    def record_evaluation(
        self,
        predictions: List[PatternDetection],
        ground_truth: List[PatternDetection]
    ) -> Dict[PatternType, MetricsEntry]:
        """
        Record evaluation results by comparing predictions to ground truth.
        
        Uses IoU (Intersection over Union) to match predictions to ground truth.
        
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
        """
        Calculate TP, FP, FN for a set of predictions and ground truth.
        
        Uses IoU matching with greedy assignment.
        
        Args:
            predictions: Predicted detections for a single pattern type.
            ground_truth: Ground truth detections for a single pattern type.
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives).
        """
        if not predictions and not ground_truth:
            return 0, 0, 0
        
        if not predictions:
            return 0, 0, len(ground_truth)
        
        if not ground_truth:
            return 0, len(predictions), 0
        
        # Track which ground truths have been matched
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
                
                iou = self._calculate_iou(pred.bounding_box, gt.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - len(matched_gt)
        
        return true_positives, false_positives, false_negatives
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box.
            box2: Second bounding box.
            
        Returns:
            IoU value between 0.0 and 1.0.
        """
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = box1.area()
        area2 = box2.area()
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def add_true_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add true positive count for a pattern type."""
        self._metrics[pattern_type].true_positives += count
    
    def add_false_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false positive count for a pattern type."""
        self._metrics[pattern_type].false_positives += count
    
    def add_false_negative(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false negative count for a pattern type."""
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

    # =========================================================================
    # Calculation Methods (Requirement 9.2)
    # =========================================================================
    
    def precision(self, pattern_type: PatternType) -> float:
        """
        Calculate precision for a specific pattern type.
        
        Precision = TP / (TP + FP)
        
        Args:
            pattern_type: The pattern type to calculate precision for.
            
        Returns:
            Precision value between 0.0 and 1.0.
        """
        return self._metrics[pattern_type].precision()
    
    def recall(self, pattern_type: PatternType) -> float:
        """
        Calculate recall for a specific pattern type.
        
        Recall = TP / (TP + FN)
        
        Args:
            pattern_type: The pattern type to calculate recall for.
            
        Returns:
            Recall value between 0.0 and 1.0.
        """
        return self._metrics[pattern_type].recall()
    
    def f1_score(self, pattern_type: PatternType) -> float:
        """
        Calculate F1 score for a specific pattern type.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Args:
            pattern_type: The pattern type to calculate F1 for.
            
        Returns:
            F1 score between 0.0 and 1.0.
        """
        return self._metrics[pattern_type].f1_score()
    
    def aggregate_precision(self) -> float:
        """
        Calculate aggregate precision across all pattern types.
        
        Uses micro-averaging: sum(TP) / (sum(TP) + sum(FP))
        
        Returns:
            Aggregate precision value between 0.0 and 1.0.
        """
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fp = sum(m.false_positives for m in self._metrics.values())
        
        denominator = total_tp + total_fp
        if denominator == 0:
            return 0.0
        return total_tp / denominator
    
    def aggregate_recall(self) -> float:
        """
        Calculate aggregate recall across all pattern types.
        
        Uses micro-averaging: sum(TP) / (sum(TP) + sum(FN))
        
        Returns:
            Aggregate recall value between 0.0 and 1.0.
        """
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fn = sum(m.false_negatives for m in self._metrics.values())
        
        denominator = total_tp + total_fn
        if denominator == 0:
            return 0.0
        return total_tp / denominator
    
    def aggregate_f1_score(self) -> float:
        """
        Calculate aggregate F1 score across all pattern types.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Returns:
            Aggregate F1 score between 0.0 and 1.0.
        """
        p = self.aggregate_precision()
        r = self.aggregate_recall()
        
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all pattern types.
        
        Returns:
            Dictionary with aggregate TP, FP, FN, precision, recall, F1.
        """
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fp = sum(m.false_positives for m in self._metrics.values())
        total_fn = sum(m.false_negatives for m in self._metrics.values())
        
        return {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": self.aggregate_precision(),
            "recall": self.aggregate_recall(),
            "f1_score": self.aggregate_f1_score()
        }
    
    # =========================================================================
    # Historical Storage Methods (Requirement 9.3)
    # =========================================================================
    
    def save_snapshot(self, config_snapshot: Optional[Dict[str, Any]] = None) -> HistoricalRecord:
        """
        Save current metrics as a historical snapshot.
        
        Args:
            config_snapshot: Optional configuration to associate with this snapshot.
            
        Returns:
            The created HistoricalRecord.
        """
        metrics_by_pattern = {
            pt.value: entry.to_dict()
            for pt, entry in self._metrics.items()
            if entry.total_predictions() > 0 or entry.total_actual() > 0
        }
        
        record = HistoricalRecord(
            timestamp=datetime.now().isoformat(),
            version=self.version,
            metrics_by_pattern=metrics_by_pattern,
            aggregate_metrics=self.get_aggregate_metrics(),
            config_snapshot=config_snapshot or {}
        )
        
        self._history.append(record)
        
        # Persist if storage path is set
        if self.storage_path:
            self._save_history()
        
        return record
    
    def get_history(self) -> List[HistoricalRecord]:
        """Get all historical records."""
        return self._history.copy()
    
    def get_history_for_version(self, version: str) -> List[HistoricalRecord]:
        """Get historical records for a specific version."""
        return [r for r in self._history if r.version == version]
    
    def _save_history(self) -> None:
        """Save history to storage path."""
        if not self.storage_path:
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        
        data = {
            "version": self.version,
            "history": [r.to_dict() for r in self._history]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_history(self) -> None:
        """Load history from storage path."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self._history = [
                HistoricalRecord.from_dict(r)
                for r in data.get("history", [])
            ]
        except (json.JSONDecodeError, KeyError):
            self._history = []

    # =========================================================================
    # Report Generation Methods (Requirement 9.4, 9.5)
    # =========================================================================
    
    def generate_report(
        self,
        format: ReportFormat = ReportFormat.SUMMARY,
        include_history: bool = False
    ) -> str:
        """
        Generate a metrics report.
        
        Args:
            format: Report format (SUMMARY or DETAILED).
            include_history: Whether to include historical data.
            
        Returns:
            Formatted report string.
        """
        if format == ReportFormat.SUMMARY:
            return self._generate_summary_report(include_history)
        else:
            return self._generate_detailed_report(include_history)
    
    def _generate_summary_report(self, include_history: bool = False) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 60,
            "PATTERN DETECTION METRICS SUMMARY",
            f"Version: {self.version}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "AGGREGATE METRICS:",
            "-" * 40,
        ]
        
        agg = self.get_aggregate_metrics()
        lines.extend([
            f"  True Positives:  {agg['true_positives']}",
            f"  False Positives: {agg['false_positives']}",
            f"  False Negatives: {agg['false_negatives']}",
            f"  Precision:       {agg['precision']:.4f}",
            f"  Recall:          {agg['recall']:.4f}",
            f"  F1 Score:        {agg['f1_score']:.4f}",
            "",
        ])
        
        # Pattern types with data
        active_patterns = [
            (pt, m) for pt, m in self._metrics.items()
            if m.total_predictions() > 0 or m.total_actual() > 0
        ]
        
        if active_patterns:
            lines.extend([
                "PATTERN BREAKDOWN:",
                "-" * 40,
            ])
            for pt, m in sorted(active_patterns, key=lambda x: x[0].value):
                lines.append(
                    f"  {pt.value}: P={m.precision():.3f} R={m.recall():.3f} F1={m.f1_score():.3f}"
                )
            lines.append("")
        
        if include_history and self._history:
            lines.extend([
                "HISTORICAL SNAPSHOTS:",
                "-" * 40,
                f"  Total snapshots: {len(self._history)}",
            ])
            for i, record in enumerate(self._history[-5:], 1):  # Last 5
                lines.append(
                    f"  [{i}] {record.timestamp} (v{record.version}) "
                    f"F1={record.aggregate_metrics.get('f1_score', 0):.3f}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _generate_detailed_report(self, include_history: bool = False) -> str:
        """Generate a detailed report."""
        lines = [
            "=" * 80,
            "PATTERN DETECTION METRICS - DETAILED REPORT",
            f"Version: {self.version}",
            f"Generated: {datetime.now().isoformat()}",
            f"IoU Threshold: {self.iou_threshold}",
            "=" * 80,
            "",
        ]
        
        # Aggregate metrics
        agg = self.get_aggregate_metrics()
        lines.extend([
            "AGGREGATE METRICS",
            "-" * 80,
            f"  True Positives (TP):  {agg['true_positives']:>8}",
            f"  False Positives (FP): {agg['false_positives']:>8}",
            f"  False Negatives (FN): {agg['false_negatives']:>8}",
            "",
            f"  Precision = TP / (TP + FP) = {agg['precision']:.6f}",
            f"  Recall    = TP / (TP + FN) = {agg['recall']:.6f}",
            f"  F1 Score  = 2 * P * R / (P + R) = {agg['f1_score']:.6f}",
            "",
        ])
        
        # Per-pattern metrics
        lines.extend([
            "PER-PATTERN METRICS",
            "-" * 80,
            f"{'Pattern Type':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}",
            "-" * 80,
        ])
        
        for pt in sorted(PatternType, key=lambda x: x.value):
            m = self._metrics[pt]
            if m.total_predictions() > 0 or m.total_actual() > 0:
                lines.append(
                    f"{pt.value:<30} {m.true_positives:>6} {m.false_positives:>6} "
                    f"{m.false_negatives:>6} {m.precision():>8.4f} {m.recall():>8.4f} "
                    f"{m.f1_score():>8.4f}"
                )
        
        lines.append("")
        
        # Historical data
        if include_history and self._history:
            lines.extend([
                "HISTORICAL DATA",
                "-" * 80,
            ])
            
            for record in self._history:
                lines.extend([
                    f"\nSnapshot: {record.timestamp}",
                    f"Version: {record.version}",
                    f"Aggregate F1: {record.aggregate_metrics.get('f1_score', 0):.4f}",
                ])
                
                if record.config_snapshot:
                    lines.append(f"Config: {json.dumps(record.config_snapshot, indent=2)}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare metrics between two versions.
        
        Args:
            version1: First version to compare.
            version2: Second version to compare.
            
        Returns:
            Dictionary with comparison results.
        """
        records_v1 = self.get_history_for_version(version1)
        records_v2 = self.get_history_for_version(version2)
        
        if not records_v1 or not records_v2:
            return {
                "error": "One or both versions have no historical data",
                "version1_records": len(records_v1),
                "version2_records": len(records_v2)
            }
        
        # Use latest record for each version
        latest_v1 = records_v1[-1]
        latest_v2 = records_v2[-1]
        
        agg1 = latest_v1.aggregate_metrics
        agg2 = latest_v2.aggregate_metrics
        
        return {
            "version1": {
                "version": version1,
                "timestamp": latest_v1.timestamp,
                "metrics": agg1
            },
            "version2": {
                "version": version2,
                "timestamp": latest_v2.timestamp,
                "metrics": agg2
            },
            "comparison": {
                "precision_diff": agg2.get("precision", 0) - agg1.get("precision", 0),
                "recall_diff": agg2.get("recall", 0) - agg1.get("recall", 0),
                "f1_diff": agg2.get("f1_score", 0) - agg1.get("f1_score", 0),
                "improved": agg2.get("f1_score", 0) > agg1.get("f1_score", 0)
            }
        }
    
    def generate_comparison_report(
        self,
        version1: str,
        version2: str
    ) -> str:
        """
        Generate a comparison report between two versions.
        
        Args:
            version1: First version to compare.
            version2: Second version to compare.
            
        Returns:
            Formatted comparison report string.
        """
        comparison = self.compare_versions(version1, version2)
        
        if "error" in comparison:
            return f"Error: {comparison['error']}"
        
        v1 = comparison["version1"]
        v2 = comparison["version2"]
        diff = comparison["comparison"]
        
        lines = [
            "=" * 60,
            "VERSION COMPARISON REPORT",
            "=" * 60,
            "",
            f"Version 1: {v1['version']} ({v1['timestamp']})",
            f"  Precision: {v1['metrics'].get('precision', 0):.4f}",
            f"  Recall:    {v1['metrics'].get('recall', 0):.4f}",
            f"  F1 Score:  {v1['metrics'].get('f1_score', 0):.4f}",
            "",
            f"Version 2: {v2['version']} ({v2['timestamp']})",
            f"  Precision: {v2['metrics'].get('precision', 0):.4f}",
            f"  Recall:    {v2['metrics'].get('recall', 0):.4f}",
            f"  F1 Score:  {v2['metrics'].get('f1_score', 0):.4f}",
            "",
            "DIFFERENCES (v2 - v1):",
            "-" * 40,
            f"  Precision: {diff['precision_diff']:+.4f}",
            f"  Recall:    {diff['recall_diff']:+.4f}",
            f"  F1 Score:  {diff['f1_diff']:+.4f}",
            "",
            f"Overall: {'IMPROVED' if diff['improved'] else 'DEGRADED'}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collector state to dictionary for serialization."""
        return {
            "version": self.version,
            "iou_threshold": self.iou_threshold,
            "metrics": {
                pt.value: entry.to_dict()
                for pt, entry in self._metrics.items()
            },
            "history": [r.to_dict() for r in self._history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsCollector":
        """Create MetricsCollector from dictionary."""
        collector = cls(
            iou_threshold=data.get("iou_threshold", cls.DEFAULT_IOU_THRESHOLD),
            version=data.get("version", "v1.0.0")
        )
        
        # Restore metrics
        for pt_value, entry_data in data.get("metrics", {}).items():
            try:
                pt = PatternType(pt_value)
                collector._metrics[pt] = MetricsEntry.from_dict(entry_data)
            except ValueError:
                pass  # Skip unknown pattern types
        
        # Restore history
        collector._history = [
            HistoricalRecord.from_dict(r)
            for r in data.get("history", [])
        ]
        
        return collector
