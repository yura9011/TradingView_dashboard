"""
Metrics data models.

Data classes for metrics tracking and historical records.

Feature: chart-pattern-analysis-framework
Requirements: 9.1, 9.3
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum


@dataclass
class MetricsEntry:
    """Single metrics entry for a pattern type."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)."""
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
