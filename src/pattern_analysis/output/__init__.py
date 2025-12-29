"""
Output generation module.

Contains components for:
- ChartAnnotator: Visual annotation of detected patterns
- ReportGenerator: Markdown and JSON report generation

Requirements:
- 8.1-8.5: Visual annotation of detected patterns
"""

from .annotator import ChartAnnotator

__all__ = [
    "ChartAnnotator",
]
