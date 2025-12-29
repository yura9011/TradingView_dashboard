"""
Main Orchestrator - Full pipeline for AI Trading Analysis.
Run: python main.py --symbol MELI
"""

import os
import sys
import yaml
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.screener import get_screener
from src.screener.chart_capture import get_chart_capture
from src.agents import get_chart_analyzer
from src.visual import get_report_generator
from src.database import get_signal_repository
from src.models import Market, Signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log", mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


class TradingAnalysisAgent:
    """Main orchestrator for the trading analysis pipeline."""
    
    def __init__(self, config_path: Path = None):
        """Initialize the agent.
        
        Args:
            config_path: Path to config YAML file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.chart_capture = get_chart_capture()
        self.report_generator = get_report_generator()
        self.signal_repo = get_signal_repository()
        
        # Gemini client (requires API key)
        self.analyzer = None
        gemini_key = os.getenv("GEMINI_API_KEY") or self.config.get("gemini", {}).get("api_key")
        
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY":
            self.analyzer = get_chart_analyzer(api_key=gemini_key)
            logger.info("Gemini analyzer initialized")
        else:
            logger.warning("GEMINI_API_KEY not set - analysis will be skipped")
        
        # Telegram notifier (optional)
        self.notifier = None
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN") or self.config.get("telegram", {}).get("bot_token")
        telegram_chat = os.getenv("TELEGRAM_CHAT_ID") or self.config.get("telegram", {}).get("chat_id")
        
        if telegram_token and telegram_chat and telegram_token != "YOUR_TELEGRAM_BOT_TOKEN":
            try:
                from src.notifier import get_telegram_notifier
                os.environ["TELEGRAM_BOT_TOKEN"] = telegram_token
                os.environ["TELEGRAM_CHAT_ID"] = str(telegram_chat)
                self.notifier = get_telegram_notifier()
                logger.info("Telegram notifier initialized")
            except Exception as e:
                logger.warning(f"Failed to init Telegram: {e}")
    
    def _load_config(self, config_path: Path = None) -> dict:
        """Load configuration from YAML."""
        if config_path is None:
            config_path = Path("config/config.yaml")
            if not config_path.exists():
                config_path = Path("config/config.example.yaml")
        
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    async def analyze_symbol(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        notify: bool = True,
    ) -> Optional[Signal]:
        """Run full analysis pipeline for a single symbol.
        
        Args:
            symbol: Ticker symbol (e.g., MELI)
            exchange: Exchange (e.g., NASDAQ)
            notify: Send Telegram notification
            
        Returns:
            Signal with analysis results
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting analysis for {exchange}:{symbol}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Capture chart screenshot (daily, 1 month)
            logger.info("📸 Step 1: Capturing chart (daily, 1 month)...")
            chart_path, price_range = await asyncio.to_thread(
                self.chart_capture.capture_sync,
                symbol=symbol,
                exchange=exchange,
                interval="D",
                range_months=1,
            )
            logger.info(f"   Chart saved: {chart_path}")
            
            # Step 2: AI Analysis (if available)
            signal = None
            if self.analyzer:
                logger.info("🧠 Step 2: Running AI analysis...")
                signal = self.analyzer.analyze_chart_image(
                    image_path=str(chart_path),
                    symbol=symbol,
                    timeframe="1D",
                    save_signal=True,
                )
                logger.info(f"   Pattern: {signal.pattern_detected}")
                logger.info(f"   Confidence: {signal.pattern_confidence:.0%}")
                logger.info(f"   Trend: {signal.trend}")
            else:
                logger.warning("   Skipping AI analysis (no API key)")
                # Create placeholder signal
                from src.models import SignalType, PatternType
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.PENDING,
                    pattern_detected=PatternType.NONE,
                    pattern_confidence=0.0,
                    trend="unknown",
                    analysis_summary="AI analysis skipped - API key not configured",
                    chart_image_path=str(chart_path),
                )
                self.signal_repo.create(signal)
            
            # Step 3: Generate report
            logger.info("📄 Step 3: Generating report...")
            report_path = self.report_generator.generate(
                signal=signal,
                chart_image_path=str(chart_path),
                annotate=True,
            )
            logger.info(f"   Report: {report_path}")
            
            # Step 4: Send notification
            if notify and self.notifier and signal:
                logger.info("📱 Step 4: Sending Telegram notification...")
                try:
                    success = await self.notifier.send_signal_alert(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type,
                        pattern=signal.pattern_detected,
                        confidence=signal.pattern_confidence,
                        trend=signal.trend or "unknown",
                        summary=signal.analysis_summary or "No summary",
                        chart_path=signal.chart_image_path,
                    )
                    if success:
                        logger.info("   ✅ Notification sent!")
                    else:
                        logger.warning("   ⚠️ Notification failed")
                except Exception as e:
                    logger.error(f"   ❌ Notification error: {e}")
            
            logger.info(f"✅ Analysis complete for {symbol}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run(self, symbols: List[str], exchange: str = "NASDAQ"):
        """Run analysis for multiple symbols sequentially.
        
        Args:
            symbols: List of ticker symbols
            exchange: Exchange for all symbols
        """
        logger.info(f"Starting analysis for {len(symbols)} symbols")
        
        results = []
        for symbol in symbols:
            result = await self.analyze_symbol(symbol, exchange)
            results.append((symbol, result))
            
            # Small delay between symbols
            await asyncio.sleep(2)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("="*60)
        
        for symbol, signal in results:
            if signal:
                status = "✅" if signal.signal_type == "candidate" else "⏳"
                logger.info(f"{status} {symbol}: {signal.signal_type} ({signal.pattern_confidence:.0%})")
            else:
                logger.info(f"❌ {symbol}: Failed")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Trading Analysis Agent")
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol to analyze")
    parser.add_argument("--exchange", "-e", type=str, default="NASDAQ", help="Exchange")
    parser.add_argument("--no-notify", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Create and run agent
    config_path = Path(args.config) if args.config else None
    agent = TradingAnalysisAgent(config_path=config_path)
    
    # Run analysis
    asyncio.run(
        agent.analyze_symbol(
            symbol=args.symbol,
            exchange=args.exchange,
            notify=not args.no_notify,
        )
    )


if __name__ == "__main__":
    main()
