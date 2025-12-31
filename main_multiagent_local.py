"""
Multi-Agent Analysis with Local Model
Run: python main_multiagent_local.py --symbol MELI

DEPRECATED: Use main_unified.py instead
This file is kept for backward compatibility.

Updated to use:
- 1 Year (1Y) timeframe with Daily (1D) candles
- EnhancedPreprocessor for region detection and auto-cropping
- Timeframe-aware configuration
"""

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

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default timeframe configuration (1 Year with Daily candles)
DEFAULT_TIMEFRAME = "1Y"
DEFAULT_CANDLE_INTERVAL = "1D"
DEFAULT_RANGE_MONTHS = 12  # 1 year of data


async def analyze_with_local_model(
    symbol: str,
    exchange: str = "",
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    range_months: int = DEFAULT_RANGE_MONTHS,
    timeframe: str = DEFAULT_TIMEFRAME,
):
    """Run analysis using local model with enhanced preprocessing.
    
    This function is kept for backward compatibility.
    Dashboard and other scripts may import this directly.
    
    Args:
        symbol: Stock symbol to analyze
        exchange: Exchange (empty for auto-detect)
        model_name: HuggingFace model name
        range_months: Months of data to show (default: 12 for 1Y)
        timeframe: Timeframe configuration (default: "1Y")
    """
    from src.analysis import run_analysis
    
    return await run_analysis(
        symbol=symbol,
        exchange=exchange,
        use_local=True,
        model_name=model_name,
        range_months=range_months,
        timeframe=timeframe,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading Analysis with Local Model (1Y/1D timeframe)"
    )
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol")
    parser.add_argument("--exchange", "-e", type=str, default="", help="Exchange (empty for auto-detect)")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--range-months", "-r",
        type=int,
        default=DEFAULT_RANGE_MONTHS,
        help=f"Months of data to show (default: {DEFAULT_RANGE_MONTHS} for 1Y)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        default=DEFAULT_TIMEFRAME,
        choices=["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})"
    )
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📊 Timeframe: {args.timeframe} with {DEFAULT_CANDLE_INTERVAL} candles")
    logger.info(f"📅 Range: {args.range_months} months of data")
    
    asyncio.run(analyze_with_local_model(
        args.symbol,
        args.exchange,
        args.model,
        args.range_months,
        args.timeframe,
    ))


if __name__ == "__main__":
    main()
