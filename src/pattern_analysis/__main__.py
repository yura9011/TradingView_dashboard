"""
Entry point for running the pattern analysis module as a script.

Usage:
    python -m src.pattern_analysis analyze <image_path> [options]
    python -m src.pattern_analysis --help
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
