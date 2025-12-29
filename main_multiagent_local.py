"""
Multi-Agent Analysis with Local Model
Run: python main_multiagent_local.py --symbol MELI

DEPRECATED: Use main_unified.py instead
This file is kept for backward compatibility.
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


async def analyze_with_local_model(
    symbol: str,
    exchange: str = "",
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
):
    """Run analysis using local model.
    
    This function is kept for backward compatibility.
    Dashboard and other scripts may import this directly.
    """
    from src.analysis import run_analysis
    
    return await run_analysis(
        symbol=symbol,
        exchange=exchange,
        use_local=True,
        model_name=model_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading Analysis with Local Model"
    )
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol")
    parser.add_argument("--exchange", "-e", type=str, default="", help="Exchange (empty for auto-detect)")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name"
    )
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    
    asyncio.run(analyze_with_local_model(
        args.symbol,
        args.exchange,
        args.model,
    ))


if __name__ == "__main__":
    main()
