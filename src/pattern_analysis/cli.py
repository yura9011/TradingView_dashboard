"""
Command-line interface for Chart Pattern Analysis Framework.

Provides CLI commands for analyzing chart images and generating
various output formats (JSON, Markdown, annotated images).

Requirements: 7.5 - Generate both machine-readable (JSON) and human-readable (Markdown) formats

Usage:
    python -m src.pattern_analysis.cli analyze <image_path> [options]
    python -m src.pattern_analysis.cli --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .factory import create_analyzer, ChartPatternAnalyzer


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def analyze_command(args: argparse.Namespace) -> int:
    """
    Execute the analyze command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    image_path = args.image
    
    # Validate input file exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1
    
    # Create analyzer with configuration
    try:
        analyzer = create_analyzer(
            config_path=args.config if args.config else None,
            pattern_definitions_path=args.patterns if args.patterns else None,
            ml_model_path=args.model if args.model else None,
            enable_cross_validation=not args.no_validation,
            consensus_threshold=args.threshold
        )
    except Exception as e:
        print(f"Error creating analyzer: {e}", file=sys.stderr)
        return 1
    
    # Run analysis
    try:
        print(f"Analyzing: {image_path}")
        result = analyzer.analyze(image_path)
        print(f"Found {len(result.detections)} pattern(s) in {result.total_time_ms:.2f}ms")
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1
    
    # Generate outputs based on format
    output_format = args.format.lower()
    output_path = args.output
    
    if output_format == "json":
        output = analyzer.to_json(result, validate=True)
        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
            print(f"JSON output saved to: {output_path}")
        else:
            print(output)
    
    elif output_format == "markdown" or output_format == "md":
        output = analyzer.to_markdown(result)
        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
            print(f"Markdown output saved to: {output_path}")
        else:
            print(output)
    
    elif output_format == "annotated" or output_format == "image":
        if not output_path:
            # Generate default output path
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_annotated.png")
        
        success = analyzer.save_annotated(
            image_path,
            result,
            output_path,
            show_confidence=not args.no_confidence,
            show_validation=not args.no_validation
        )
        
        if success:
            print(f"Annotated image saved to: {output_path}")
        else:
            print(f"Error: Failed to save annotated image", file=sys.stderr)
            return 1
    
    elif output_format == "all":
        # Generate all output formats
        input_path = Path(image_path)
        base_name = input_path.stem
        output_dir = Path(output_path) if output_path else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON
        json_path = output_dir / f"{base_name}_analysis.json"
        json_path.write_text(analyzer.to_json(result), encoding="utf-8")
        print(f"JSON saved to: {json_path}")
        
        # Markdown
        md_path = output_dir / f"{base_name}_analysis.md"
        md_path.write_text(analyzer.to_markdown(result), encoding="utf-8")
        print(f"Markdown saved to: {md_path}")
        
        # Annotated image
        img_path = output_dir / f"{base_name}_annotated.png"
        analyzer.save_annotated(
            image_path,
            result,
            str(img_path),
            show_confidence=not args.no_confidence,
            show_validation=not args.no_validation
        )
        print(f"Annotated image saved to: {img_path}")
    
    else:
        print(f"Error: Unknown output format: {output_format}", file=sys.stderr)
        return 1
    
    # Print summary if not quiet
    if not args.quiet and result.detections:
        print("\nDetected Patterns:")
        for i, det in enumerate(result.detections, 1):
            name = det.pattern_type.value.replace("_", " ").title()
            conf = det.confidence * 100
            print(f"  {i}. {name} ({conf:.1f}% confidence)")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pattern-analysis",
        description="Chart Pattern Analysis Framework - Detect and classify chart patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a chart and print JSON to stdout
  python -m src.pattern_analysis.cli analyze chart.png --format json

  # Analyze and save annotated image
  python -m src.pattern_analysis.cli analyze chart.png --format annotated -o result.png

  # Generate all output formats
  python -m src.pattern_analysis.cli analyze chart.png --format all -o ./output/

  # Use custom configuration
  python -m src.pattern_analysis.cli analyze chart.png --config config/custom.yaml
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (debug logging)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a chart image for patterns",
        description="Analyze a chart image and detect patterns"
    )
    
    analyze_parser.add_argument(
        "image",
        help="Path to the chart image file"
    )
    
    analyze_parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "md", "annotated", "image", "all"],
        default="json",
        help="Output format (default: json)"
    )
    
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout for text formats)"
    )
    
    analyze_parser.add_argument(
        "-c", "--config",
        help="Path to configuration YAML file"
    )
    
    analyze_parser.add_argument(
        "-p", "--patterns",
        help="Path to pattern definitions YAML file"
    )
    
    analyze_parser.add_argument(
        "-m", "--model",
        help="Path to ML model file (YOLO) for detection"
    )
    
    analyze_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Consensus threshold for cross-validation (default: 0.5)"
    )
    
    analyze_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable cross-validation"
    )
    
    analyze_parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Hide confidence scores in annotated output"
    )
    
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (debug logging)"
    )
    
    analyze_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False)
    )
    
    # Execute command
    if args.command == "analyze":
        return analyze_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
