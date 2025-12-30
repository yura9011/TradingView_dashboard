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

# Import from new modular structure
from .models import MetricsEntry, HistoricalRecord, ReportFormat
from .tracker import MetricsTracker
from .calculator import MetricsCalculator
from .reporter import MetricsReporter
from .storage import MetricsStorage
from .collector_new import MetricsCollector

__all__ = [
    # Main collector (orchestrator)
    "MetricsCollector",
    # Data models
    "MetricsEntry",
    "HistoricalRecord",
    "ReportFormat",
    # Component classes
    "MetricsTracker",
    "MetricsCalculator",
    "MetricsReporter",
    "MetricsStorage",
]
