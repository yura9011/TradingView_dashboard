"""
Environment variable parsing for configuration overrides.

Feature: chart-pattern-analysis-framework
Requirements: 10.5
"""

import os
import json
from typing import Any, Dict, List
from copy import deepcopy


class EnvConfigParser:
    """Parses environment variables for configuration overrides."""
    
    ENV_PREFIX = "PATTERN_ANALYSIS"
    
    # Known section names (to handle multi-word sections)
    KNOWN_SECTIONS = [
        "preprocessing", "feature_extraction", "classification",
        "cross_validation", "output", "registry", "metrics", "logging"
    ]
    
    def apply_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables follow the pattern:
        PATTERN_ANALYSIS_<SECTION>_<KEY>=value
        
        For nested keys, use double underscore:
        PATTERN_ANALYSIS_PREPROCESSING_DENOISE__ENABLED=true
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        result = deepcopy(config)
        prefix = f"{self.ENV_PREFIX}_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Remove prefix and split into path components
            key_path = env_key[len(prefix):].lower()
            
            # Handle nested keys (double underscore for nesting)
            path_parts = self._parse_key_path(key_path)
            
            if not path_parts:
                continue
            
            # Convert value to appropriate type
            typed_value = self._convert_value(env_value)
            
            # Set the value in config
            self._set_nested_value(result, path_parts, typed_value)
        
        return result
    
    def _parse_key_path(self, key_path: str) -> List[str]:
        """
        Parse environment variable key path into nested path components.
        
        Examples:
            "preprocessing_target_width" -> ["preprocessing", "target_width"]
            "preprocessing_denoise__enabled" -> ["preprocessing", "denoise", "enabled"]
        
        Args:
            key_path: Lowercase key path from environment variable
            
        Returns:
            List of path components
        """
        # First, try to match a known section
        section = None
        remaining = key_path
        
        for known in self.KNOWN_SECTIONS:
            if key_path.startswith(known + "_"):
                section = known
                remaining = key_path[len(known) + 1:]
                break
        
        if section is None:
            # No known section found, use first part as section
            first_underscore = key_path.find("_")
            if first_underscore == -1:
                return [key_path]
            section = key_path[:first_underscore]
            remaining = key_path[first_underscore + 1:]
        
        # Parse remaining part for nested keys (double underscore)
        result = [section]
        
        if remaining:
            nested_parts = remaining.split("__")
            result.extend(nested_parts)
        
        return result
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert environment variable string value to appropriate Python type.
        
        Supports:
        - Booleans: "true", "false", "yes", "no", "1", "0"
        - Integers: "123"
        - Floats: "1.23"
        - Lists: "[1, 2, 3]" or "1,2,3"
        - Strings: everything else
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Handle None/null
        if value.lower() in ("none", "null", ""):
            return None
        
        # Handle booleans
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Handle integers
        try:
            return int(value)
        except ValueError:
            pass
        
        # Handle floats
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle JSON-like lists
        if value.startswith("[") and value.endswith("]"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Handle comma-separated lists
        if "," in value and not value.startswith('"'):
            parts = [p.strip() for p in value.split(",")]
            return [self._convert_value(p) for p in parts]
        
        # Return as string
        return value
    
    @staticmethod
    def _set_nested_value(config: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        Set a value in a nested dictionary using a path.
        
        Args:
            config: Configuration dictionary to modify
            path: List of keys forming the path
            value: Value to set
        """
        current = config
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        if path:
            current[path[-1]] = value
