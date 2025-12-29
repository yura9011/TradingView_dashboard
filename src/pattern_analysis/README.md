# Chart Pattern Analysis Framework

A modular framework for detecting, classifying, and validating chart patterns in financial price charts using computer vision and machine learning techniques.

## Overview

This framework provides a complete pipeline for analyzing financial chart images to identify technical patterns such as Head and Shoulders, Double Top/Bottom, Triangles, Wedges, and more. It combines classical computer vision techniques with optional machine learning models for high-accuracy pattern detection.

### Key Features

- **Modular Pipeline Architecture**: Each processing stage is independent and can be upgraded or replaced
- **Hybrid Detection**: Combines rule-based geometric analysis with ML-based detection (YOLO)
- **Cross-Validation**: Multiple detection methods verify results to minimize false positives
- **Extensible Pattern Registry**: Add new patterns via YAML configuration without code changes
- **Multiple Output Formats**: JSON, Markdown, and annotated images
- **Comprehensive Metrics**: Track precision, recall, and F1 scores per pattern type

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input                                     │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ Chart Image  │    │ Configuration│                           │
│  └──────┬───────┘    └──────┬───────┘                           │
└─────────┼───────────────────┼───────────────────────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Preprocessor │─▶│  Feature     │─▶│   Pattern    │           │
│  │              │  │  Extractor   │  │  Classifier  │           │
│  └──────────────┘  └──────────────┘  └──────┬───────┘           │
│                                             │                    │
│                                             ▼                    │
│                                      ┌──────────────┐           │
│                                      │    Cross     │           │
│                                      │  Validator   │           │
│                                      └──────┬───────┘           │
└─────────────────────────────────────────────┼───────────────────┘
                                              │
          ┌───────────────────────────────────┼───────────────────┐
          │                                   ▼                   │
          │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
          │  │ JSON Result  │  │   Markdown   │  │  Annotated  │ │
          │  │              │  │    Report    │  │    Image    │ │
          │  └──────────────┘  └──────────────┘  └─────────────┘ │
          │                        Output                         │
          └───────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Preprocessor**: Normalizes image dimensions, converts color space, applies denoising, and masks UI elements
2. **Feature Extractor**: Extracts candlesticks, trendlines, support/resistance zones, and volume profiles
3. **Pattern Classifier**: Classifies patterns using rule-based and ML-based detection
4. **Cross Validator**: Validates detections using alternative methods for consensus

## Installation

The framework is part of the main project. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Required dependencies:
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pydantic` - Data validation
- `PyYAML` - Configuration parsing
- `scipy` - Signal processing (for peak detection)
- `hypothesis` - Property-based testing (dev)

Optional:
- `ultralytics` - YOLO model support for ML-based detection

## Quick Start

### Python API

```python
from src.pattern_analysis import create_analyzer

# Create analyzer with default configuration
analyzer = create_analyzer()

# Analyze a chart image
result = analyzer.analyze("path/to/chart.png")

# Print detected patterns
for detection in result.detections:
    print(f"{detection.pattern_type.value}: {detection.confidence:.1%}")

# Generate annotated image
analyzer.save_annotated("chart.png", result, "annotated_chart.png")

# Export to JSON
json_output = analyzer.to_json(result)

# Export to Markdown
markdown_report = analyzer.to_markdown(result)
```

### Command Line Interface

```bash
# Analyze and output JSON
python -m src.pattern_analysis analyze chart.png --format json

# Generate annotated image
python -m src.pattern_analysis analyze chart.png --format annotated -o result.png

# Generate all output formats
python -m src.pattern_analysis analyze chart.png --format all -o ./output/

# Use custom configuration
python -m src.pattern_analysis analyze chart.png --config config/custom.yaml

# Disable cross-validation for faster processing
python -m src.pattern_analysis analyze chart.png --no-validation
```

## Configuration

### Main Configuration (`config/pattern_analysis.yaml`)

```yaml
pipeline:
  stages:
    - id: preprocessor
      class: StandardPreprocessor
      config:
        target_size: [1280, 720]
        denoise: true
        
    - id: feature_extractor
      class: EdgeBasedFeatureExtractor
      config:
        extract_volume: true
        min_trendline_length: 100
        
    - id: classifier
      class: HybridPatternClassifier
      config:
        confidence_threshold: 0.3
        ml_model_path: null  # Optional YOLO model
        
    - id: validator
      class: MultiMethodCrossValidator
      config:
        consensus_threshold: 0.5

output:
  formats: ["json", "markdown"]
  annotate_image: true

logging:
  level: INFO
  file: "logs/pattern_analysis.log"
```

### Pattern Definitions (`config/pattern_definitions.yaml`)

Patterns are defined in YAML format with the following structure:

```yaml
patterns:
  head_shoulders:
    name: "Head and Shoulders"
    category: reversal
    direction: bearish
    aliases: ["H&S", "head_and_shoulders"]
    components:
      - name: left_shoulder
        type: peak
        required: true
      - name: head
        type: peak
        required: true
      - name: right_shoulder
        type: peak
        required: true
    validation_rules:
      - "head.high > left_shoulder.high"
      - "head.high > right_shoulder.high"
    min_confidence: 0.6
```

## Supported Patterns

### Reversal Patterns
- Head and Shoulders / Inverse Head and Shoulders
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Rising Wedge / Falling Wedge

### Continuation Patterns
- Bull Flag / Bear Flag
- Cup and Handle
- Ascending Channel / Descending Channel

### Triangle Patterns
- Ascending Triangle
- Descending Triangle
- Symmetrical Triangle

## Module Structure

```
src/pattern_analysis/
├── __init__.py          # Public API exports
├── __main__.py          # Entry point for python -m
├── cli.py               # Command-line interface
├── factory.py           # Factory functions for component creation
├── integration.py       # Integration with legacy systems
├── config/
│   ├── __init__.py
│   └── manager.py       # Configuration management
├── metrics/
│   ├── __init__.py
│   └── collector.py     # Metrics tracking and reporting
├── models/
│   ├── __init__.py
│   ├── dataclasses.py   # Core data structures
│   ├── enums.py         # Pattern types and categories
│   └── schemas.py       # JSON schema validation
├── output/
│   ├── __init__.py
│   └── annotator.py     # Visual annotation
├── pipeline/
│   ├── __init__.py
│   ├── interfaces.py    # Abstract base classes
│   ├── preprocessor.py  # Image preprocessing
│   ├── feature_extractor.py  # Feature extraction
│   ├── classifier.py    # Pattern classification
│   ├── cross_validator.py    # Cross-validation
│   └── executor.py      # Pipeline orchestration
└── registry/
    ├── __init__.py
    └── pattern_registry.py  # Pattern definitions management
```

## API Reference

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_analyzer()` | Create a fully configured analyzer (recommended) |
| `create_pipeline()` | Create a pipeline executor |
| `create_preprocessor()` | Create a preprocessor instance |
| `create_feature_extractor()` | Create a feature extractor |
| `create_classifier()` | Create a pattern classifier |
| `create_cross_validator()` | Create a cross-validator |
| `create_annotator()` | Create a chart annotator |
| `create_pattern_registry()` | Create a pattern registry |

### Core Classes

| Class | Description |
|-------|-------------|
| `ChartPatternAnalyzer` | High-level analyzer for pattern detection |
| `PipelineExecutor` | Orchestrates pipeline stage execution |
| `PatternRegistry` | Manages pattern definitions |
| `ChartAnnotator` | Draws annotations on chart images |

### Data Models

| Model | Description |
|-------|-------------|
| `AnalysisResult` | Complete analysis output with all detections |
| `PatternDetection` | Single pattern detection result |
| `ValidationResult` | Cross-validation result for a detection |
| `FeatureMap` | Extracted features from chart image |
| `BoundingBox` | Coordinates delimiting a detected region |

### Enums

| Enum | Values |
|------|--------|
| `PatternCategory` | `REVERSAL`, `CONTINUATION`, `BILATERAL` |
| `PatternType` | `HEAD_SHOULDERS`, `DOUBLE_TOP`, `ASCENDING_TRIANGLE`, etc. |

## Output Formats

### JSON Output

```json
{
  "image_path": "chart.png",
  "timestamp": "2024-12-29T10:30:00",
  "total_time_ms": 245.5,
  "detections": [
    {
      "pattern_type": "head_shoulders",
      "category": "reversal",
      "confidence": 0.85,
      "bounding_box": {"x1": 100, "y1": 50, "x2": 400, "y2": 300},
      "is_validated": true,
      "validation_score": 0.75
    }
  ]
}
```

### Annotated Image

Patterns are drawn with color-coded bounding boxes:
- **Green**: Bullish patterns (continuation/reversal up)
- **Red**: Bearish patterns (reversal down)
- **Yellow**: Neutral/bilateral patterns

Labels show pattern name and confidence percentage.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run property-based tests only
pytest tests/test_*_properties.py -v

# Run with coverage
pytest tests/ --cov=src/pattern_analysis --cov-report=html
```

### Property-Based Tests

The framework uses Hypothesis for property-based testing to verify:
- Preprocessor output validity and aspect ratio preservation
- Feature extraction completeness
- Pattern detection validity and ordering
- Serialization round-trip consistency
- Cross-validation consistency
- Metrics calculation correctness

## Error Handling

The framework implements graceful degradation:

| Error | Behavior |
|-------|----------|
| Image not found | Returns error with path info |
| Image corrupted | Returns error with details |
| Stage failure | Logs error, continues with partial results |
| Configuration error | Fails fast at startup |

## Performance

Typical processing times on a standard machine:
- Preprocessing: 20-50ms
- Feature extraction: 50-100ms
- Classification: 30-80ms
- Cross-validation: 20-50ms
- **Total**: 120-280ms per image

## Contributing

1. Follow the existing code style and patterns
2. Add property-based tests for new functionality
3. Update pattern definitions in YAML for new patterns
4. Document new features in this README

## License

Part of the Chart Analysis System. See main project LICENSE.
