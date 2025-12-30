"""
MetricsCollector - Orchestrates metrics tracking, calculation, and reporting.

This is the main entry point for the metrics module, composing:
- MetricsTracker: TP/FP/FN tracking
- MetricsCalculator: Precision/recall/F1 calculation
- MetricsReporter: Report generation
- MetricsStorage: Historical persistence

Feature: chart-pattern-analysis-framework
Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from typing import Dict, List, Optional, Any

from ..models import PatternType, PatternDetection
from .models import MetricsEntry, HistoricalRecord, ReportFormat
from .tracker import MetricsTracker
from .calculator import MetricsCalculator
from .reporter import MetricsReporter
from .storage import MetricsStorage


class MetricsCollector:
    """
    Collector for tracking and evaluating pattern detection accuracy.
    
    Composes tracker, calculator, reporter, and storage components.
    
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
        """Initialize MetricsCollector with components."""
        self.iou_threshold = iou_threshold
        self.version = version
        self.storage_path = storage_path
        
        # Initialize components
        self._tracker = MetricsTracker(iou_threshold)
        self._calculator = MetricsCalculator(self._tracker.get_all_metrics())
        self._reporter = MetricsReporter(
            self._tracker.get_all_metrics(),
            version,
            iou_threshold
        )
        self._storage = MetricsStorage(storage_path)
    
    # =========================================================================
    # Tracking Methods (Requirement 9.1)
    # =========================================================================
    
    def record_evaluation(
        self,
        predictions: List[PatternDetection],
        ground_truth: List[PatternDetection]
    ) -> Dict[PatternType, MetricsEntry]:
        """Record evaluation results by comparing predictions to ground truth."""
        result = self._tracker.record_evaluation(predictions, ground_truth)
        self._sync_components()
        return result
    
    def add_true_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add true positive count."""
        self._tracker.add_true_positive(pattern_type, count)
        self._sync_components()
    
    def add_false_positive(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false positive count."""
        self._tracker.add_false_positive(pattern_type, count)
        self._sync_components()
    
    def add_false_negative(self, pattern_type: PatternType, count: int = 1) -> None:
        """Manually add false negative count."""
        self._tracker.add_false_negative(pattern_type, count)
        self._sync_components()
    
    def get_metrics(self, pattern_type: PatternType) -> MetricsEntry:
        """Get metrics for a specific pattern type."""
        return self._tracker.get_metrics(pattern_type)
    
    def get_all_metrics(self) -> Dict[PatternType, MetricsEntry]:
        """Get metrics for all pattern types."""
        return self._tracker.get_all_metrics()
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._tracker.reset_metrics()
        self._sync_components()
    
    # =========================================================================
    # Calculation Methods (Requirement 9.2)
    # =========================================================================
    
    def precision(self, pattern_type: PatternType) -> float:
        """Calculate precision for a specific pattern type."""
        return self._calculator.precision(pattern_type)
    
    def recall(self, pattern_type: PatternType) -> float:
        """Calculate recall for a specific pattern type."""
        return self._calculator.recall(pattern_type)
    
    def f1_score(self, pattern_type: PatternType) -> float:
        """Calculate F1 score for a specific pattern type."""
        return self._calculator.f1_score(pattern_type)
    
    def aggregate_precision(self) -> float:
        """Calculate aggregate precision across all pattern types."""
        return self._calculator.aggregate_precision()
    
    def aggregate_recall(self) -> float:
        """Calculate aggregate recall across all pattern types."""
        return self._calculator.aggregate_recall()
    
    def aggregate_f1_score(self) -> float:
        """Calculate aggregate F1 score across all pattern types."""
        return self._calculator.aggregate_f1_score()
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all pattern types."""
        return self._calculator.get_aggregate_metrics()
    
    # =========================================================================
    # Historical Storage Methods (Requirement 9.3)
    # =========================================================================
    
    def save_snapshot(
        self,
        config_snapshot: Optional[Dict[str, Any]] = None
    ) -> HistoricalRecord:
        """Save current metrics as a historical snapshot."""
        return self._storage.save_snapshot(
            self._tracker.get_all_metrics(),
            self.version,
            config_snapshot
        )
    
    def get_history(self) -> List[HistoricalRecord]:
        """Get all historical records."""
        return self._storage.get_history()
    
    def get_history_for_version(self, version: str) -> List[HistoricalRecord]:
        """Get historical records for a specific version."""
        return self._storage.get_history_for_version(version)
    
    # =========================================================================
    # Report Generation Methods (Requirement 9.4, 9.5)
    # =========================================================================
    
    def generate_report(
        self,
        format: ReportFormat = ReportFormat.SUMMARY,
        include_history: bool = False
    ) -> str:
        """Generate a metrics report."""
        return self._reporter.generate_report(
            format,
            include_history,
            self._storage.get_history()
        )
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare metrics between two versions."""
        return MetricsReporter.compare_versions(
            self._storage.get_history(),
            version1,
            version2
        )
    
    def generate_comparison_report(self, version1: str, version2: str) -> str:
        """Generate a comparison report between two versions."""
        return MetricsReporter.generate_comparison_report(
            self._storage.get_history(),
            version1,
            version2
        )
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collector state to dictionary for serialization."""
        return {
            "version": self.version,
            "iou_threshold": self.iou_threshold,
            "metrics": {
                pt.value: entry.to_dict()
                for pt, entry in self._tracker.get_all_metrics().items()
            },
            "history": [r.to_dict() for r in self._storage.get_history()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsCollector":
        """Create MetricsCollector from dictionary."""
        collector = cls(
            iou_threshold=data.get("iou_threshold", cls.DEFAULT_IOU_THRESHOLD),
            version=data.get("version", "v1.0.0")
        )
        
        # Restore metrics
        metrics = {}
        for pt_value, entry_data in data.get("metrics", {}).items():
            try:
                pt = PatternType(pt_value)
                metrics[pt] = MetricsEntry.from_dict(entry_data)
            except ValueError:
                pass
        
        if metrics:
            collector._tracker.set_metrics(metrics)
            collector._sync_components()
        
        # Restore history
        history = [
            HistoricalRecord.from_dict(r)
            for r in data.get("history", [])
        ]
        collector._storage.set_history(history)
        
        return collector
    
    # =========================================================================
    # Internal
    # =========================================================================
    
    def _sync_components(self) -> None:
        """Sync metrics to calculator and reporter after changes."""
        metrics = self._tracker.get_all_metrics()
        self._calculator.update_metrics(metrics)
        self._reporter.update_metrics(metrics)
