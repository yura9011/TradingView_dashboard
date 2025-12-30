"""
Property-based tests for MetricsCollector.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
Property 15: Metrics Calculation Correctness
Validates: Requirements 9.1, 9.2
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import math

from src.pattern_analysis.metrics import MetricsCollector, MetricsEntry
from src.pattern_analysis.models import PatternType, PatternCategory, BoundingBox, PatternDetection


# =============================================================================
# Custom Strategies for Domain Objects
# =============================================================================

@st.composite
def metrics_entry_strategy(draw):
    """Generate valid MetricsEntry objects with non-negative counts."""
    tp = draw(st.integers(min_value=0, max_value=1000))
    fp = draw(st.integers(min_value=0, max_value=1000))
    fn = draw(st.integers(min_value=0, max_value=1000))
    return MetricsEntry(true_positives=tp, false_positives=fp, false_negatives=fn)


@st.composite
def bounding_box_strategy(draw, max_dim=500):
    """Generate valid bounding boxes where x1 < x2 and y1 < y2."""
    x1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    y1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    x2 = draw(st.integers(min_value=x1 + 1, max_value=max_dim))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=max_dim))
    return BoundingBox(x1, y1, x2, y2)


@st.composite
def pattern_detection_strategy(draw, pattern_type=None):
    """Generate valid pattern detections."""
    bbox = draw(bounding_box_strategy())
    pt = pattern_type or draw(st.sampled_from(list(PatternType)))
    return PatternDetection(
        pattern_type=pt,
        category=draw(st.sampled_from(list(PatternCategory))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bounding_box=bbox,
        metadata={},
        detector_id="test_detector"
    )


# =============================================================================
# Property Tests for Metrics Calculation Correctness (Property 15)
# =============================================================================

class TestMetricsCalculationCorrectness:
    """
    Property tests for metrics calculation correctness.
    
    Feature: chart-pattern-analysis-framework
    Property 15: Metrics Calculation Correctness
    Validates: Requirements 9.1, 9.2
    
    For any set of predictions P and ground truth G:
    - precision SHALL equal TP / (TP + FP)
    - recall SHALL equal TP / (TP + FN)
    - F1 SHALL equal 2 * (precision * recall) / (precision + recall)
    """
    
    @given(
        tp=st.integers(min_value=0, max_value=1000),
        fp=st.integers(min_value=0, max_value=1000),
        fn=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_precision_formula(self, tp: int, fp: int, fn: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        Precision SHALL equal TP / (TP + FP).
        Returns 0.0 when TP + FP = 0.
        """
        entry = MetricsEntry(true_positives=tp, false_positives=fp, false_negatives=fn)
        
        denominator = tp + fp
        if denominator == 0:
            expected_precision = 0.0
        else:
            expected_precision = tp / denominator
        
        actual_precision = entry.precision()
        
        assert abs(actual_precision - expected_precision) < 1e-10, (
            f"Precision mismatch: expected {expected_precision}, got {actual_precision} "
            f"for TP={tp}, FP={fp}"
        )
    
    @given(
        tp=st.integers(min_value=0, max_value=1000),
        fp=st.integers(min_value=0, max_value=1000),
        fn=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_recall_formula(self, tp: int, fp: int, fn: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        Recall SHALL equal TP / (TP + FN).
        Returns 0.0 when TP + FN = 0.
        """
        entry = MetricsEntry(true_positives=tp, false_positives=fp, false_negatives=fn)
        
        denominator = tp + fn
        if denominator == 0:
            expected_recall = 0.0
        else:
            expected_recall = tp / denominator
        
        actual_recall = entry.recall()
        
        assert abs(actual_recall - expected_recall) < 1e-10, (
            f"Recall mismatch: expected {expected_recall}, got {actual_recall} "
            f"for TP={tp}, FN={fn}"
        )
    
    @given(
        tp=st.integers(min_value=0, max_value=1000),
        fp=st.integers(min_value=0, max_value=1000),
        fn=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_f1_formula(self, tp: int, fp: int, fn: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        F1 SHALL equal 2 * (precision * recall) / (precision + recall).
        Returns 0.0 when precision + recall = 0.
        """
        entry = MetricsEntry(true_positives=tp, false_positives=fp, false_negatives=fn)
        
        precision = entry.precision()
        recall = entry.recall()
        
        if precision + recall == 0:
            expected_f1 = 0.0
        else:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        actual_f1 = entry.f1_score()
        
        assert abs(actual_f1 - expected_f1) < 1e-10, (
            f"F1 mismatch: expected {expected_f1}, got {actual_f1} "
            f"for precision={precision}, recall={recall}"
        )
    
    @given(metrics_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_metrics_in_valid_range(self, entry: MetricsEntry):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        All metrics (precision, recall, F1) SHALL be in range [0.0, 1.0].
        """
        precision = entry.precision()
        recall = entry.recall()
        f1 = entry.f1_score()
        
        assert 0.0 <= precision <= 1.0, f"Precision {precision} out of range [0, 1]"
        assert 0.0 <= recall <= 1.0, f"Recall {recall} out of range [0, 1]"
        assert 0.0 <= f1 <= 1.0, f"F1 {f1} out of range [0, 1]"
    
    @given(
        tp=st.integers(min_value=1, max_value=1000),
        fp=st.integers(min_value=0, max_value=1000),
        fn=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_f1_bounded_by_arithmetic_mean(self, tp: int, fp: int, fn: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.2
        
        F1 score is the harmonic mean of precision and recall,
        so it SHALL be <= arithmetic mean of precision and recall.
        """
        entry = MetricsEntry(true_positives=tp, false_positives=fp, false_negatives=fn)
        
        precision = entry.precision()
        recall = entry.recall()
        f1 = entry.f1_score()
        
        # F1 is harmonic mean, so F1 <= arithmetic mean when both > 0
        if precision > 0 and recall > 0:
            arithmetic_mean = (precision + recall) / 2
            assert f1 <= arithmetic_mean + 1e-10, (
                f"F1 ({f1}) should be <= arithmetic mean = {arithmetic_mean}"
            )


class TestAggregateMetricsCalculation:
    """
    Property tests for aggregate metrics calculation across pattern types.
    
    Feature: chart-pattern-analysis-framework
    Property 15: Metrics Calculation Correctness
    Validates: Requirements 9.1, 9.2
    """
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(list(PatternType)),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100)
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_precision_is_micro_average(self, pattern_metrics):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        Aggregate precision SHALL equal sum(TP) / (sum(TP) + sum(FP)).
        """
        collector = MetricsCollector()
        
        total_tp = 0
        total_fp = 0
        
        for pattern_type, tp, fp, fn in pattern_metrics:
            collector.add_true_positive(pattern_type, tp)
            collector.add_false_positive(pattern_type, fp)
            collector.add_false_negative(pattern_type, fn)
            total_tp += tp
            total_fp += fp
        
        expected = 0.0 if (total_tp + total_fp) == 0 else total_tp / (total_tp + total_fp)
        actual = collector.aggregate_precision()
        
        assert abs(actual - expected) < 1e-10, (
            f"Aggregate precision mismatch: expected {expected}, got {actual}"
        )
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(list(PatternType)),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100)
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_recall_is_micro_average(self, pattern_metrics):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        Aggregate recall SHALL equal sum(TP) / (sum(TP) + sum(FN)).
        """
        collector = MetricsCollector()
        
        total_tp = 0
        total_fn = 0
        
        for pattern_type, tp, fp, fn in pattern_metrics:
            collector.add_true_positive(pattern_type, tp)
            collector.add_false_positive(pattern_type, fp)
            collector.add_false_negative(pattern_type, fn)
            total_tp += tp
            total_fn += fn
        
        expected = 0.0 if (total_tp + total_fn) == 0 else total_tp / (total_tp + total_fn)
        actual = collector.aggregate_recall()
        
        assert abs(actual - expected) < 1e-10, (
            f"Aggregate recall mismatch: expected {expected}, got {actual}"
        )
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(list(PatternType)),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100)
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_f1_formula(self, pattern_metrics):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1, 9.2
        
        Aggregate F1 SHALL equal 2 * (precision * recall) / (precision + recall).
        """
        collector = MetricsCollector()
        
        for pattern_type, tp, fp, fn in pattern_metrics:
            collector.add_true_positive(pattern_type, tp)
            collector.add_false_positive(pattern_type, fp)
            collector.add_false_negative(pattern_type, fn)
        
        precision = collector.aggregate_precision()
        recall = collector.aggregate_recall()
        
        if precision + recall == 0:
            expected = 0.0
        else:
            expected = 2 * (precision * recall) / (precision + recall)
        
        actual = collector.aggregate_f1_score()
        
        assert abs(actual - expected) < 1e-10, (
            f"Aggregate F1 mismatch: expected {expected}, got {actual}"
        )


class TestMetricsEntryAddition:
    """
    Property tests for MetricsEntry addition operation.
    
    Feature: chart-pattern-analysis-framework
    Property 15: Metrics Calculation Correctness
    Validates: Requirements 9.1
    """
    
    @given(metrics_entry_strategy(), metrics_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_metrics_entry_addition(self, entry1: MetricsEntry, entry2: MetricsEntry):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1
        
        Adding two MetricsEntry objects SHALL sum their TP, FP, FN counts.
        """
        combined = entry1 + entry2
        
        assert combined.true_positives == entry1.true_positives + entry2.true_positives
        assert combined.false_positives == entry1.false_positives + entry2.false_positives
        assert combined.false_negatives == entry1.false_negatives + entry2.false_negatives
    
    @given(metrics_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_metrics_entry_addition_identity(self, entry: MetricsEntry):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1
        
        Adding a zero MetricsEntry SHALL not change the original.
        """
        zero = MetricsEntry(0, 0, 0)
        combined = entry + zero
        
        assert combined.true_positives == entry.true_positives
        assert combined.false_positives == entry.false_positives
        assert combined.false_negatives == entry.false_negatives


class TestMetricsCollectorTracking:
    """
    Property tests for MetricsCollector tracking functionality.
    
    Feature: chart-pattern-analysis-framework
    Property 15: Metrics Calculation Correctness
    Validates: Requirements 9.1
    """
    
    @given(
        pattern_type=st.sampled_from(list(PatternType)),
        tp_count=st.integers(min_value=0, max_value=100),
        fp_count=st.integers(min_value=0, max_value=100),
        fn_count=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_manual_tracking_accumulates(
        self,
        pattern_type: PatternType,
        tp_count: int,
        fp_count: int,
        fn_count: int
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1
        
        Manual tracking methods SHALL accumulate counts correctly.
        """
        collector = MetricsCollector()
        
        collector.add_true_positive(pattern_type, tp_count)
        collector.add_false_positive(pattern_type, fp_count)
        collector.add_false_negative(pattern_type, fn_count)
        
        metrics = collector.get_metrics(pattern_type)
        
        assert metrics.true_positives == tp_count
        assert metrics.false_positives == fp_count
        assert metrics.false_negatives == fn_count
    
    @given(
        pattern_type=st.sampled_from(list(PatternType)),
        counts=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_additions_accumulate(
        self,
        pattern_type: PatternType,
        counts: list
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 15: Metrics Calculation Correctness
        Validates: Requirements 9.1
        
        Multiple additions SHALL accumulate to the sum of all counts.
        """
        collector = MetricsCollector()
        
        for count in counts:
            collector.add_true_positive(pattern_type, count)
        
        metrics = collector.get_metrics(pattern_type)
        assert metrics.true_positives == sum(counts)
