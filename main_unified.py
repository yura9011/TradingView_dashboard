"""
Trading Analysis - Unified Entry Point
Run: python main_unified.py analyze --symbol AAPL
     python main_unified.py dashboard
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except AttributeError:
            pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability for local model."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
        else:
            print("  ⚠️  No GPU - CPU mode (slow)")
        return cuda
    except ImportError:
        print("  ⚠️  PyTorch not installed")
        return False


async def cmd_analyze(args):
    """Run analysis command."""
    from src.analysis import run_analysis
    
    print("\n" + "=" * 60)
    print("🔍 SYSTEM CHECK")
    print("=" * 60)
    
    if args.local:
        check_gpu()
    
    print("=" * 60 + "\n")
    
    # Ensure directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    
    await run_analysis(
        symbol=args.symbol,
        exchange=args.exchange,
        use_local=args.local,
        model_name=args.model,
    )


def cmd_dashboard(args):
    """Run dashboard command."""
    from dashboard.app import app
    
    print("🚀 Starting Dashboard on http://localhost:8080")
    app.run(debug=args.debug, port=args.port, threaded=True)


def main():
    parser = argparse.ArgumentParser(
        description="Trading Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_unified.py analyze --symbol AAPL
  python main_unified.py analyze --symbol AAL --model Qwen/Qwen2-VL-7B-Instruct
  python main_unified.py dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a stock symbol")
    analyze_parser.add_argument("--symbol", "-s", required=True, help="Stock symbol")
    analyze_parser.add_argument("--exchange", "-e", default="", help="Exchange (empty for auto-detect)")
    analyze_parser.add_argument("--local", "-l", action="store_true", default=True, help="Use local model (default)")
    analyze_parser.add_argument("--cloud", "-c", action="store_true", help="Use Gemini cloud API")
    analyze_parser.add_argument("--model", "-m", default="Qwen/Qwen2-VL-2B-Instruct", help="Local model name")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start web dashboard")
    dash_parser.add_argument("--port", "-p", type=int, default=8080, help="Port number")
    dash_parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        if args.cloud:
            args.local = False
        asyncio.run(cmd_analyze(args))
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
