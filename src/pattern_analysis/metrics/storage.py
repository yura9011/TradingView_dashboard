"""
Metrics Storage - Handles persistence of historical metrics.

Feature: chart-pattern-analysis-framework
Requirements: 9.3
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..models import PatternType
from .models import MetricsEntry, HistoricalRecord


class MetricsStorage:
    """
    Handles persistence of metrics history to disk.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize storage with optional path."""
        self.storage_path = storage_path
        self._history: List[HistoricalRecord] = []
        
        if storage_path and os.path.exists(storage_path):
            self._load_history()
    
    def save_snapshot(
        self,
        metrics: Dict[PatternType, MetricsEntry],
        version: str,
        config_snapshot: Optional[Dict[str, Any]] = None
    ) -> HistoricalRecord:
        """Save current metrics as a historical snapshot."""
        # Calculate aggregate metrics
        total_tp = sum(m.true_positives for m in metrics.values())
        total_fp = sum(m.false_positives for m in metrics.values())
        total_fn = sum(m.false_negatives for m in metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        aggregate_metrics = {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        metrics_by_pattern = {
            pt.value: entry.to_dict()
            for pt, entry in metrics.items()
            if entry.total_predictions() > 0 or entry.total_actual() > 0
        }
        
        record = HistoricalRecord(
            timestamp=datetime.now().isoformat(),
            version=version,
            metrics_by_pattern=metrics_by_pattern,
            aggregate_metrics=aggregate_metrics,
            config_snapshot=config_snapshot or {}
        )
        
        self._history.append(record)
        
        if self.storage_path:
            self._save_history(version)
        
        return record
    
    def get_history(self) -> List[HistoricalRecord]:
        """Get all historical records."""
        return self._history.copy()
    
    def get_history_for_version(self, version: str) -> List[HistoricalRecord]:
        """Get historical records for a specific version."""
        return [r for r in self._history if r.version == version]
    
    def set_history(self, history: List[HistoricalRecord]) -> None:
        """Set history from external source (for deserialization)."""
        self._history = history
    
    def _save_history(self, version: str) -> None:
        """Save history to storage path."""
        if not self.storage_path:
            return
        
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        
        data = {
            "version": version,
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
