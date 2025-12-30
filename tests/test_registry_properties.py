"""
Property-based tests for PatternRegistry.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import string

from src.pattern_analysis.registry import (
    PatternRegistry,
    PatternDefinition,
    RegistryValidationError,
)


# =============================================================================
# Custom Strategies for Domain Objects
# =============================================================================

# Valid values for pattern definitions
VALID_CATEGORIES = ["reversal", "continuation", "bilateral"]
VALID_DIRECTIONS = ["bullish", "bearish", "neutral"]


@st.composite
def valid_pattern_id_strategy(draw):
    """Generate valid pattern identifiers (snake_case)."""
    # Generate a valid identifier: lowercase letters and underscores
    first_char = draw(st.sampled_from(string.ascii_lowercase))
    rest = draw(st.text(
        alphabet=string.ascii_lowercase + "_",
        min_size=2,
        max_size=20
    ))
    return first_char + rest


@st.composite
def valid_alias_strategy(draw):
    """Generate valid alias strings."""
    return draw(st.text(
        alphabet=string.ascii_letters + string.digits + "_&",
        min_size=1,
        max_size=20
    ))


@st.composite
def valid_component_strategy(draw):
    """Generate valid component dictionaries."""
    return {
        "name": draw(st.text(
            alphabet=string.ascii_lowercase + "_",
            min_size=1,
            max_size=20
        )),
        "type": draw(st.sampled_from(["peak", "trough", "trendline", "support", "resistance", "point"])),
        "required": draw(st.booleans())
    }


@st.composite
def valid_pattern_data_strategy(draw):
    """Generate valid pattern definition data."""
    name = draw(st.text(
        alphabet=string.ascii_letters + " ",
        min_size=3,
        max_size=50
    ))
    # Ensure name is not just whitespace
    assume(name.strip())
    
    return {
        "name": name.strip(),
        "category": draw(st.sampled_from(VALID_CATEGORIES)),
        "direction": draw(st.sampled_from(VALID_DIRECTIONS)),
        "description": draw(st.text(min_size=0, max_size=100)),
        "aliases": draw(st.lists(valid_alias_strategy(), min_size=0, max_size=5, unique=True)),
        "components": draw(st.lists(valid_component_strategy(), min_size=0, max_size=5)),
        "validation_rules": draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=3)),
        "min_confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    }


@st.composite
def invalid_pattern_data_strategy(draw):
    """Generate invalid pattern definition data (missing required fields)."""
    # Randomly omit required fields
    data = {}
    
    # Randomly include or exclude required fields
    if draw(st.booleans()):
        data["name"] = draw(st.text(min_size=1, max_size=20))
    if draw(st.booleans()):
        data["category"] = draw(st.sampled_from(VALID_CATEGORIES))
    if draw(st.booleans()):
        data["direction"] = draw(st.sampled_from(VALID_DIRECTIONS))
    
    # Ensure at least one required field is missing
    required = {"name", "category", "direction"}
    assume(not required.issubset(data.keys()))
    
    return data


@st.composite
def invalid_category_data_strategy(draw):
    """Generate pattern data with invalid category."""
    invalid_category = draw(st.text(min_size=1, max_size=20))
    assume(invalid_category.lower() not in VALID_CATEGORIES)
    
    return {
        "name": draw(st.text(min_size=1, max_size=20)),
        "category": invalid_category,
        "direction": draw(st.sampled_from(VALID_DIRECTIONS))
    }


@st.composite
def invalid_direction_data_strategy(draw):
    """Generate pattern data with invalid direction."""
    invalid_direction = draw(st.text(min_size=1, max_size=20))
    assume(invalid_direction.lower() not in VALID_DIRECTIONS)
    
    return {
        "name": draw(st.text(min_size=1, max_size=20)),
        "category": draw(st.sampled_from(VALID_CATEGORIES)),
        "direction": invalid_direction
    }


@st.composite
def invalid_confidence_data_strategy(draw):
    """Generate pattern data with invalid min_confidence."""
    # Generate confidence outside valid range
    invalid_conf = draw(st.one_of(
        st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.01, allow_nan=False, allow_infinity=False)
    ))
    
    return {
        "name": draw(st.text(min_size=1, max_size=20)),
        "category": draw(st.sampled_from(VALID_CATEGORIES)),
        "direction": draw(st.sampled_from(VALID_DIRECTIONS)),
        "min_confidence": invalid_conf
    }


# =============================================================================
# Property Tests for Registry Schema Validation (Property 9)
# =============================================================================

class TestRegistrySchemaValidation:
    """
    Property tests for registry schema validation.
    
    Feature: chart-pattern-analysis-framework
    Property 9: Registry Schema Validation
    Validates: Requirements 6.2
    """
    
    @given(invalid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_missing_required_fields_rejected(self, invalid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 9: Registry Schema Validation
        Validates: Requirements 6.2
        
        For any pattern definition missing required fields,
        the registry SHALL reject it and raise a validation error.
        """
        registry = PatternRegistry()
        
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.add_pattern("test_pattern", invalid_data)
        
        # Verify error contains information about missing fields
        assert exc_info.value.errors, "Validation error should contain error details"
        assert any("Missing required field" in err for err in exc_info.value.errors), \
            f"Error should mention missing required field, got: {exc_info.value.errors}"
    
    @given(invalid_category_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_invalid_category_rejected(self, invalid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 9: Registry Schema Validation
        Validates: Requirements 6.2
        
        For any pattern definition with invalid category,
        the registry SHALL reject it and raise a validation error.
        """
        registry = PatternRegistry()
        
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.add_pattern("test_pattern", invalid_data)
        
        assert exc_info.value.errors, "Validation error should contain error details"
        assert any("Invalid category" in err for err in exc_info.value.errors), \
            f"Error should mention invalid category, got: {exc_info.value.errors}"
    
    @given(invalid_direction_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_invalid_direction_rejected(self, invalid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 9: Registry Schema Validation
        Validates: Requirements 6.2
        
        For any pattern definition with invalid direction,
        the registry SHALL reject it and raise a validation error.
        """
        registry = PatternRegistry()
        
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.add_pattern("test_pattern", invalid_data)
        
        assert exc_info.value.errors, "Validation error should contain error details"
        assert any("Invalid direction" in err for err in exc_info.value.errors), \
            f"Error should mention invalid direction, got: {exc_info.value.errors}"
    
    @given(invalid_confidence_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_invalid_confidence_rejected(self, invalid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 9: Registry Schema Validation
        Validates: Requirements 6.2
        
        For any pattern definition with min_confidence outside [0, 1],
        the registry SHALL reject it and raise a validation error.
        """
        registry = PatternRegistry()
        
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.add_pattern("test_pattern", invalid_data)
        
        assert exc_info.value.errors, "Validation error should contain error details"
        assert any("min_confidence" in err for err in exc_info.value.errors), \
            f"Error should mention min_confidence, got: {exc_info.value.errors}"
    
    @given(valid_pattern_id_strategy(), valid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_valid_pattern_accepted(self, pattern_id: str, valid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 9: Registry Schema Validation
        Validates: Requirements 6.2
        
        For any valid pattern definition conforming to schema,
        the registry SHALL accept it without raising an error.
        """
        registry = PatternRegistry()
        
        # Should not raise
        pattern = registry.add_pattern(pattern_id, valid_data)
        
        assert pattern is not None, "Valid pattern should be added"
        assert pattern.id == pattern_id, "Pattern ID should match"
        assert pattern.name == valid_data["name"], "Pattern name should match"
        assert pattern.category == valid_data["category"], "Pattern category should match"
        assert pattern.direction == valid_data["direction"], "Pattern direction should match"


# =============================================================================
# Property Tests for Registry Lookup Consistency (Property 10)
# =============================================================================

class TestRegistryLookupConsistency:
    """
    Property tests for registry lookup consistency.
    
    Feature: chart-pattern-analysis-framework
    Property 10: Registry Lookup Consistency
    Validates: Requirements 6.3, 6.5
    """
    
    @given(valid_pattern_id_strategy(), valid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_lookup_by_name_returns_same_as_alias(self, pattern_id: str, valid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 10: Registry Lookup Consistency
        Validates: Requirements 6.3, 6.5
        
        For any pattern registered with name N and aliases [A1, A2, ...],
        looking up by N or any alias SHALL return the same pattern definition.
        """
        registry = PatternRegistry()
        registry.add_pattern(pattern_id, valid_data)
        
        # Lookup by pattern_id
        by_id = registry.get(pattern_id)
        assert by_id is not None, f"Pattern should be found by ID: {pattern_id}"
        
        # Lookup by each alias
        for alias in valid_data.get("aliases", []):
            by_alias = registry.get(alias)
            assert by_alias is not None, f"Pattern should be found by alias: {alias}"
            assert by_alias.id == by_id.id, \
                f"Lookup by alias '{alias}' should return same pattern as lookup by ID"
            assert by_alias.name == by_id.name, \
                f"Pattern name should match for alias '{alias}'"
    
    @given(valid_pattern_id_strategy(), valid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_case_insensitive_alias_lookup(self, pattern_id: str, valid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 10: Registry Lookup Consistency
        Validates: Requirements 6.5
        
        For any pattern with aliases, lookup SHALL be case-insensitive.
        """
        registry = PatternRegistry()
        registry.add_pattern(pattern_id, valid_data)
        
        # Test case-insensitive lookup for each alias
        for alias in valid_data.get("aliases", []):
            # Try different case variations
            lower_result = registry.get(alias.lower())
            upper_result = registry.get(alias.upper())
            
            assert lower_result is not None, f"Lowercase alias lookup should work: {alias.lower()}"
            assert upper_result is not None, f"Uppercase alias lookup should work: {alias.upper()}"
            assert lower_result.id == upper_result.id, \
                "Case variations should return same pattern"
    
    @given(valid_pattern_id_strategy(), valid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_category_lookup_includes_pattern(self, pattern_id: str, valid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 10: Registry Lookup Consistency
        Validates: Requirements 6.3
        
        For any pattern with category C, get_by_category(C) SHALL include that pattern.
        """
        registry = PatternRegistry()
        registry.add_pattern(pattern_id, valid_data)
        
        category = valid_data["category"]
        patterns_in_category = registry.get_by_category(category)
        
        pattern_ids = [p.id for p in patterns_in_category]
        assert pattern_id in pattern_ids, \
            f"Pattern '{pattern_id}' should be in category '{category}' results"
    
    @given(valid_pattern_id_strategy(), valid_pattern_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_search_by_name_finds_pattern(self, pattern_id: str, valid_data: dict):
        """
        Feature: chart-pattern-analysis-framework
        Property 10: Registry Lookup Consistency
        Validates: Requirements 6.3
        
        For any pattern with name N, search(name=partial_N) SHALL include that pattern.
        """
        registry = PatternRegistry()
        registry.add_pattern(pattern_id, valid_data)
        
        # Search by partial name (first 3 characters)
        name = valid_data["name"]
        if len(name) >= 3:
            partial_name = name[:3]
            results = registry.search(name=partial_name)
            
            pattern_ids = [p.id for p in results]
            assert pattern_id in pattern_ids, \
                f"Pattern '{pattern_id}' should be found when searching for '{partial_name}'"
    
    @given(
        st.lists(
            st.tuples(valid_pattern_id_strategy(), valid_pattern_data_strategy()),
            min_size=2,
            max_size=5,
            unique_by=lambda x: x[0]  # Unique pattern IDs
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_multiple_patterns_all_findable(self, patterns: list):
        """
        Feature: chart-pattern-analysis-framework
        Property 10: Registry Lookup Consistency
        Validates: Requirements 6.3, 6.5
        
        For any set of registered patterns, each pattern SHALL be findable
        by its ID. Note: aliases may be overwritten if duplicated across patterns.
        """
        registry = PatternRegistry()
        
        # Collect all aliases to track which pattern owns each alias (last one wins)
        alias_to_pattern = {}
        for pattern_id, data in patterns:
            for alias in data.get("aliases", []):
                alias_to_pattern[alias.lower()] = pattern_id
        
        # Add all patterns
        for pattern_id, data in patterns:
            registry.add_pattern(pattern_id, data)
        
        # Verify each pattern is findable by ID
        for pattern_id, data in patterns:
            by_id = registry.get(pattern_id)
            assert by_id is not None, f"Pattern '{pattern_id}' should be findable by ID"
        
        # Verify aliases map to the correct pattern (last one added wins)
        for alias, expected_pattern_id in alias_to_pattern.items():
            by_alias = registry.get(alias)
            assert by_alias is not None, f"Pattern should be findable by alias '{alias}'"
            assert by_alias.id == expected_pattern_id, \
                f"Alias '{alias}' should map to pattern '{expected_pattern_id}'"
