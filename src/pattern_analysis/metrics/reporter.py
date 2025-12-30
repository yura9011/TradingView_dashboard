"""
Metrics Reporter - Generates reports and version comparisons.

Feature: chart-pattern-analysis-framework
Requirements: 9.4, 9.5
"""

import json
from datetime import datetime
from typing import Dict, List, Any

from ..models import PatternType
from .models import MetricsEntry, HistoricalRecord, ReportFormat


class MetricsReporter:
    """
    Generates metrics reports in various formats.
    
    Supports summary and detailed reports, plus version comparisons.
    """
    
    def __init__(
        self,
        metrics: Dict[PatternType, MetricsEntry],
        version: str,
        iou_threshold: float = 0.5
    ):
        """Initialize reporter with metrics data."""
        self._metrics = metrics
        self.version = version
        self.iou_threshold = iou_threshold
    
    def update_metrics(self, metrics: Dict[PatternType, MetricsEntry]) -> None:
        """Update the metrics reference."""
        self._metrics = metrics
    
    def generate_report(
        self,
        format: ReportFormat = ReportFormat.SUMMARY,
        include_history: bool = False,
        history: List[HistoricalRecord] = None
    ) -> str:
        """Generate a metrics report."""
        if format == ReportFormat.SUMMARY:
            return self._generate_summary_report(include_history, history or [])
        else:
            return self._generate_detailed_report(include_history, history or [])
    
    def _get_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics."""
        total_tp = sum(m.true_positives for m in self._metrics.values())
        total_fp = sum(m.false_positives for m in self._metrics.values())
        total_fn = sum(m.false_negatives for m in self._metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def _generate_summary_report(
        self,
        include_history: bool,
        history: List[HistoricalRecord]
    ) -> str:
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
        
        agg = self._get_aggregate_metrics()
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
        
        if include_history and history:
            lines.extend([
                "HISTORICAL SNAPSHOTS:",
                "-" * 40,
                f"  Total snapshots: {len(history)}",
            ])
            for i, record in enumerate(history[-5:], 1):
                lines.append(
                    f"  [{i}] {record.timestamp} (v{record.version}) "
                    f"F1={record.aggregate_metrics.get('f1_score', 0):.3f}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _generate_detailed_report(
        self,
        include_history: bool,
        history: List[HistoricalRecord]
    ) -> str:
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
        agg = self._get_aggregate_metrics()
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
        if include_history and history:
            lines.extend([
                "HISTORICAL DATA",
                "-" * 80,
            ])
            
            for record in history:
                lines.extend([
                    f"\nSnapshot: {record.timestamp}",
                    f"Version: {record.version}",
                    f"Aggregate F1: {record.aggregate_metrics.get('f1_score', 0):.4f}",
                ])
                
                if record.config_snapshot:
                    lines.append(f"Config: {json.dumps(record.config_snapshot, indent=2)}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    @staticmethod
    def compare_versions(
        history: List[HistoricalRecord],
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare metrics between two versions."""
        records_v1 = [r for r in history if r.version == version1]
        records_v2 = [r for r in history if r.version == version2]
        
        if not records_v1 or not records_v2:
            return {
                "error": "One or both versions have no historical data",
                "version1_records": len(records_v1),
                "version2_records": len(records_v2)
            }
        
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
    
    @staticmethod
    def generate_comparison_report(
        history: List[HistoricalRecord],
        version1: str,
        version2: str
    ) -> str:
        """Generate a comparison report between two versions."""
        comparison = MetricsReporter.compare_versions(history, version1, version2)
        
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
