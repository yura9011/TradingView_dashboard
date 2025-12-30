"""
Property-based tests for ConfigurationManager.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
Property 16: Configuration Override Precedence
Validates: Requirements 10.1
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import string
import yaml

from src.pattern_analysis.config import (
    ConfigurationManager,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigParseError,
)


# =============================================================================
# Custom Strategies for Configuration Testing
# =============================================================================

# Valid configuration keys (simple lowercase letters only, no underscores)
@st.composite
def valid_config_key_strategy(draw):
    """Generate valid configuration key names (simple, no underscores)."""
    return draw(st.text(
        alphabet=string.ascii_lowercase,
        min_size=2,
        max_size=15
    ))


@st.composite
def simple_config_value_strategy(draw):
    """Generate simple configuration values (not nested)."""
    return draw(st.one_of(
        st.integers(min_value=0, max_value=10000),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20),
    ))


@st.composite
def env_override_value_strategy(draw):
    """Generate values suitable for environment variable overrides."""
    return draw(st.one_of(
        st.integers(min_value=0, max_value=10000),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20),
    ))


@st.composite
def config_section_strategy(draw):
    """Generate a configuration section with key-value pairs."""
    num_keys = draw(st.integers(min_value=1, max_value=5))
    section = {}
    
    for _ in range(num_keys):
        key = draw(valid_config_key_strategy())
        value = draw(simple_config_value_strategy())
        section[key] = value
    
    return section


@st.composite
def full_config_strategy(draw):
    """Generate a full configuration dictionary."""
    # Use predefined section names to match schema
    sections = ["preprocessing", "classification", "logging", "metrics"]
    config = {}
    
    for section in sections:
        if draw(st.booleans()):  # Randomly include sections
            config[section] = draw(config_section_strategy())
    
    # Ensure at least one section
    if not config:
        config["preprocessing"] = draw(config_section_strategy())
    
    return config


# =============================================================================
# Property Tests for Configuration Override Precedence (Property 16)
# =============================================================================

class TestConfigOverridePrecedence:
    """
    Property tests for configuration override precedence.
    
    Feature: chart-pattern-analysis-framework
    Property 16: Configuration Override Precedence
    Validates: Requirements 10.1
    """
    
    @given(
        section=st.sampled_from(["preprocessing", "classification", "logging", "metrics"]),
        key=valid_config_key_strategy(),
        yaml_value=st.integers(min_value=0, max_value=100),
        env_value=st.integers(min_value=101, max_value=200)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_env_overrides_yaml_value(self, section: str, key: str, yaml_value: int, env_value: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 16: Configuration Override Precedence
        Validates: Requirements 10.1
        
        For any configuration key K with value V1 in YAML file and value V2 
        in environment variable, the loaded configuration SHALL use V2 
        (environment overrides file).
        """
        # Ensure values are different
        assume(yaml_value != env_value)
        
        # Create temporary YAML config file
        config = {section: {key: yaml_value}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Set environment variable override
            env_key = f"PATTERN_ANALYSIS_{section.upper()}_{key.upper()}"
            original_env = os.environ.get(env_key)
            os.environ[env_key] = str(env_value)
            
            try:
                # Load configuration with env overrides
                cm = ConfigurationManager(
                    config_path=temp_path,
                    validate=False,  # Skip validation for arbitrary keys
                    apply_env_overrides=True
                )
                
                # Verify environment value takes precedence
                loaded_value = cm.get(f"{section}.{key}")
                
                assert loaded_value == env_value, \
                    f"Environment value ({env_value}) should override YAML value ({yaml_value}), " \
                    f"but got {loaded_value}"
                
            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original_env
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    @given(
        section=st.sampled_from(["preprocessing", "classification", "logging", "metrics"]),
        key=valid_config_key_strategy(),
        yaml_value=st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=10),
        env_value=st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=10)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_env_overrides_yaml_string_value(self, section: str, key: str, yaml_value: str, env_value: str):
        """
        Feature: chart-pattern-analysis-framework
        Property 16: Configuration Override Precedence
        Validates: Requirements 10.1
        
        For any string configuration key, environment variable SHALL override YAML value.
        """
        # Ensure values are different and not boolean-like or null-like strings
        assume(yaml_value != env_value)
        assume(env_value.lower() not in ["true", "false", "yes", "no", "on", "off", "none", "null"])
        
        # Create temporary YAML config file
        config = {section: {key: yaml_value}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Set environment variable override
            env_key = f"PATTERN_ANALYSIS_{section.upper()}_{key.upper()}"
            original_env = os.environ.get(env_key)
            os.environ[env_key] = env_value
            
            try:
                # Load configuration with env overrides
                cm = ConfigurationManager(
                    config_path=temp_path,
                    validate=False,
                    apply_env_overrides=True
                )
                
                # Verify environment value takes precedence
                loaded_value = cm.get(f"{section}.{key}")
                
                assert loaded_value == env_value, \
                    f"Environment value ({env_value}) should override YAML value ({yaml_value}), " \
                    f"but got {loaded_value}"
                
            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original_env
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    @given(
        section=st.sampled_from(["preprocessing", "classification", "logging", "metrics"]),
        key=valid_config_key_strategy(),
        yaml_value=st.booleans(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_env_overrides_yaml_boolean_value(self, section: str, key: str, yaml_value: bool):
        """
        Feature: chart-pattern-analysis-framework
        Property 16: Configuration Override Precedence
        Validates: Requirements 10.1
        
        For any boolean configuration key, environment variable SHALL override YAML value.
        """
        env_value = not yaml_value  # Opposite boolean
        
        # Create temporary YAML config file
        config = {section: {key: yaml_value}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Set environment variable override
            env_key = f"PATTERN_ANALYSIS_{section.upper()}_{key.upper()}"
            original_env = os.environ.get(env_key)
            os.environ[env_key] = str(env_value).lower()  # "true" or "false"
            
            try:
                # Load configuration with env overrides
                cm = ConfigurationManager(
                    config_path=temp_path,
                    validate=False,
                    apply_env_overrides=True
                )
                
                # Verify environment value takes precedence
                loaded_value = cm.get(f"{section}.{key}")
                
                assert loaded_value == env_value, \
                    f"Environment value ({env_value}) should override YAML value ({yaml_value}), " \
                    f"but got {loaded_value}"
                
            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original_env
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    @given(
        section=st.sampled_from(["preprocessing", "classification", "logging", "metrics"]),
        key=valid_config_key_strategy(),
        yaml_value=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_yaml_value_preserved_without_env_override(self, section: str, key: str, yaml_value: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 16: Configuration Override Precedence
        Validates: Requirements 10.1
        
        For any configuration key without environment override, 
        the YAML value SHALL be preserved.
        """
        # Create temporary YAML config file
        config = {section: {key: yaml_value}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Ensure no environment variable is set for this key
            env_key = f"PATTERN_ANALYSIS_{section.upper()}_{key.upper()}"
            original_env = os.environ.get(env_key)
            os.environ.pop(env_key, None)
            
            try:
                # Load configuration
                cm = ConfigurationManager(
                    config_path=temp_path,
                    validate=False,
                    apply_env_overrides=True
                )
                
                # Verify YAML value is preserved
                loaded_value = cm.get(f"{section}.{key}")
                
                assert loaded_value == yaml_value, \
                    f"YAML value ({yaml_value}) should be preserved when no env override, " \
                    f"but got {loaded_value}"
                
            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ[env_key] = original_env
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    @given(
        section=st.sampled_from(["preprocessing", "classification"]),
        nested_key=valid_config_key_strategy(),
        inner_key=valid_config_key_strategy(),
        yaml_value=st.integers(min_value=0, max_value=100),
        env_value=st.integers(min_value=101, max_value=200)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_env_overrides_nested_yaml_value(self, section: str, nested_key: str, inner_key: str, yaml_value: int, env_value: int):
        """
        Feature: chart-pattern-analysis-framework
        Property 16: Configuration Override Precedence
        Validates: Requirements 10.1
        
        For any nested configuration key, environment variable with double underscore
        SHALL override the nested YAML value.
        """
        # Ensure values are different
        assume(yaml_value != env_value)
        assume(nested_key != inner_key)  # Avoid key collision
        
        # Create temporary YAML config file with nested structure
        config = {section: {nested_key: {inner_key: yaml_value}}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Set environment variable override with double underscore for nesting
            env_key = f"PATTERN_ANALYSIS_{section.upper()}_{nested_key.upper()}__{inner_key.upper()}"
            original_env = os.environ.get(env_key)
            os.environ[env_key] = str(env_value)
            
            try:
                # Load configuration with env overrides
                cm = ConfigurationManager(
                    config_path=temp_path,
                    validate=False,
                    apply_env_overrides=True
                )
                
                # Verify environment value takes precedence for nested key
                loaded_value = cm.get(f"{section}.{nested_key}.{inner_key}")
                
                assert loaded_value == env_value, \
                    f"Environment value ({env_value}) should override nested YAML value ({yaml_value}), " \
                    f"but got {loaded_value}"
                
            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original_env
        finally:
            # Clean up temp file
            os.unlink(temp_path)


# =============================================================================
# Additional Property Tests for Configuration Manager
# =============================================================================

class TestConfigValidation:
    """
    Property tests for configuration validation.
    
    Feature: chart-pattern-analysis-framework
    Validates: Requirements 10.5
    """
    
    @given(
        invalid_level=st.text(alphabet=string.ascii_letters, min_size=3, max_size=10)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_invalid_log_level_rejected(self, invalid_level: str):
        """
        For any invalid log level, validation SHALL reject the configuration.
        """
        # Ensure it's not a valid level
        assume(invalid_level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        # Create temporary YAML config file
        config = {"logging": {"level": invalid_level}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigurationManager(
                    config_path=temp_path,
                    validate=True,
                    apply_env_overrides=False
                )
            
            assert any("level" in err.lower() for err in exc_info.value.errors), \
                f"Error should mention 'level', got: {exc_info.value.errors}"
        finally:
            os.unlink(temp_path)
    
    @given(
        invalid_threshold=st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_invalid_confidence_threshold_rejected(self, invalid_threshold: float):
        """
        For any confidence threshold > 1.0, validation SHALL reject the configuration.
        """
        # Create temporary YAML config file
        config = {"classification": {"confidence_threshold": invalid_threshold}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigurationManager(
                    config_path=temp_path,
                    validate=True,
                    apply_env_overrides=False
                )
            
            assert any("confidence_threshold" in err for err in exc_info.value.errors), \
                f"Error should mention 'confidence_threshold', got: {exc_info.value.errors}"
        finally:
            os.unlink(temp_path)


class TestConfigFileHandling:
    """
    Property tests for configuration file handling.
    
    Feature: chart-pattern-analysis-framework
    Validates: Requirements 10.1
    """
    
    @given(
        nonexistent_path=st.text(alphabet=string.ascii_letters + string.digits, min_size=5, max_size=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_nonexistent_file_raises_error(self, nonexistent_path: str):
        """
        For any nonexistent configuration file path, loading SHALL raise ConfigFileNotFoundError.
        """
        # Ensure path doesn't exist
        full_path = f"/tmp/nonexistent_{nonexistent_path}.yaml"
        assume(not os.path.exists(full_path))
        
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            ConfigurationManager(
                config_path=full_path,
                validate=False,
                apply_env_overrides=False
            )
        
        assert full_path in str(exc_info.value), \
            f"Error should mention the path, got: {exc_info.value}"
