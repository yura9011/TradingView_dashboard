"""
Metrics and evaluation module.

Provides tracking and calculation of:
- Detection accuracy (TP, FP, FN)
- Precision, recall, F1 scores
- Historical performance metrics
- Report generation and version comparison

Feature: chart-pattern-analysis-framework
Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from .collector import (
    MetricsCollector,
    MetricsEntry,
    HistoricalRecord,
    ReportFormat,
)

__all__ = [
    "MetricsCollector",
    "MetricsEntry",
    "HistoricalRecord",
    "ReportFormat",
]
