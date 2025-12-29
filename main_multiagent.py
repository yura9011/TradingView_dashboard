"""
Multi-Agent Analysis with Gemini API
Run: python main_multiagent.py --symbol MELI

DEPRECATED: Use main_unified.py instead
This file is kept for backward compatibility.
"""

import os
import sys
import yaml
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


def load_api_key():
    """Load API key from config or environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        api_key = config.get("gemini", {}).get("api_key")
        if api_key and api_key != "YOUR_GEMINI_API_KEY":
            os.environ["GEMINI_API_KEY"] = api_key
            return api_key
    
    raise ValueError("GEMINI_API_KEY not found")


async def analyze_with_multiagent(symbol: str, exchange: str = ""):
    """Run analysis using Gemini API.
    
    This function is kept for backward compatibility.
    """
    from src.analysis import run_analysis
    
    return await run_analysis(
        symbol=symbol,
        exchange=exchange,
        use_local=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Trading Analysis")
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol")
    parser.add_argument("--exchange", "-e", type=str, default="", help="Exchange (empty for auto-detect)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    load_api_key()
    asyncio.run(analyze_with_multiagent(args.symbol, args.exchange))


if __name__ == "__main__":
    main()
