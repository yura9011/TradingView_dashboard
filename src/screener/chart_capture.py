"""
Chart Capture - Screenshot TradingView charts using Playwright.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from playwright.async_api import async_playwright, Browser, Page

logger = logging.getLogger(__name__)

# TradingView chart URL templates
# interval: D=daily, W=weekly, M=monthly (use W for ~1 year view)
TV_CHART_URL = "https://www.tradingview.com/chart/?symbol={exchange}%3A{symbol}&interval=W"
TV_SYMBOL_URL = "https://www.tradingview.com/symbols/{exchange}-{symbol}/"


class ChartCapture:
    """Captures chart screenshots from TradingView using Playwright."""
    
    def __init__(
        self,
        output_dir: Path = None,
        headless: bool = True,
        timeout: int = 30000,
    ):
        """Initialize chart capture.
        
        Args:
            output_dir: Directory to save screenshots
            headless: Run browser in headless mode
            timeout: Page load timeout in ms
        """
        self.output_dir = output_dir or Path("data/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.timeout = timeout
        self._browser: Optional[Browser] = None
    
    async def capture(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        timeframe: str = "D",
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Capture a chart screenshot.
        
        Args:
            symbol: Ticker symbol (e.g., MELI)
            exchange: Exchange (e.g., NASDAQ)
            timeframe: Chart timeframe
            width: Screenshot width
            height: Screenshot height
            
        Returns:
            Path to saved screenshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{symbol}_{timestamp}.png"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            try:
                # Use chart page for better control over timeframe
                url = TV_CHART_URL.format(exchange=exchange, symbol=symbol)
                logger.info(f"Navigating to {url}")
                logger.info(f"Using weekly timeframe for ~1 year view")
                
                await page.goto(url, timeout=self.timeout)
                
                # Wait longer for chart to fully load (including data)
                logger.info("Waiting for chart to load...")
                await page.wait_for_timeout(8000)
                
                # Try to find and screenshot the chart area
                # TradingView uses various selectors for charts
                chart_selectors = [
                    ".chart-markup-table",
                    ".chart-container",
                    "[data-name='legend-source-item']",
                    ".tv-symbol-price-quote",
                ]
                
                chart_element = None
                for selector in chart_selectors:
                    try:
                        chart_element = await page.query_selector(selector)
                        if chart_element:
                            logger.info(f"Found chart element with selector: {selector}")
                            break
                    except:
                        continue
                
                # Take full page screenshot if no specific element found
                if chart_element:
                    await chart_element.screenshot(path=str(output_path))
                else:
                    logger.warning("Chart element not found, taking full page screenshot")
                    await page.screenshot(path=str(output_path), full_page=False)
                
                logger.info(f"Screenshot saved: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to capture chart: {e}")
                # Take error state screenshot
                await page.screenshot(path=str(output_path))
                
            finally:
                await browser.close()
        
        return output_path
    
    def capture_sync(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        timeframe: str = "D",
    ) -> Path:
        """Synchronous wrapper for capture."""
        return asyncio.run(self.capture(symbol, exchange, timeframe))


def get_chart_capture(output_dir: Path = None) -> ChartCapture:
    """Get ChartCapture instance."""
    return ChartCapture(output_dir=output_dir)
