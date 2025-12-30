"""
Metrics Calculator - Calculates precision, recall, F1 scores.

Feature: chart-pattern-analysis-framework
Requirements: 9.2
"""

from typing import Dict, Any

from ..models import PatternType
from .models import MetricsEntry


class MetricsCalculator:
    """
    Calculates precision, recall, and F1 scores from metrics data.
    
    Supports both per-pattern and aggregate calculations.
    """
    
    def __init__(self, metrics: Dict[PatternType, MetricsEntry]):
        """Initialize with metrics dictionary."""
        self._metrics = metrics
    
    def update_metrics(self, metrics: Dict[PatternType, MetricsEntry]) -> None:
        """Update the metrics reference."""
        self._metrics = metrics
    
    # Per-pattern calculations
    
    def precision(self, pattern_type: PatternType) -> float:
        """Calculate precision for a specific pattern type."""
        return self._metrics[pattern_type].precision()
    
    def recall(self, pattern_type: PatternType) -> float:
        """Calculate recall for a specific pattern type."""
        return self._metrics[pattern_type].recall()
    
    def f1_score(self, pattern_type: PatternType) -> float:
        """Calculate F1 score for a specific pattern type."""
        return self._metrics[pattern_type].f1_score()
    
    # Aggregate calculations (micro-averaging)
    
    def aggregate_precision(self) -> float:
        """Calculate aggregate precision: sum(TP) / (sum(TP) + sum(FP))."""
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fp = sum(m.false_positives for m in self._metrics.values())
        
        denominator = total_tp + total_fp
        if denominator == 0:
            return 0.0
        return total_tp / denominator
    
    def aggregate_recall(self) -> float:
        """Calculate aggregate recall: sum(TP) / (sum(TP) + sum(FN))."""
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fn = sum(m.false_negatives for m in self._metrics.values())
        
        denominator = total_tp + total_fn
        if denominator == 0:
            return 0.0
        return total_tp / denominator
    
    def aggregate_f1_score(self) -> float:
        """Calculate aggregate F1 score."""
        p = self.aggregate_precision()
        r = self.aggregate_recall()
        
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all pattern types."""
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
