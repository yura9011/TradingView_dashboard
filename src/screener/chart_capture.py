"""
Chart Capture - Screenshot TradingView charts using Playwright.
Includes OCR for extracting price levels from chart axis.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from playwright.async_api import async_playwright, Browser, Page
from PIL import Image

logger = logging.getLogger(__name__)

# TradingView chart URL templates
# interval: D=daily, W=weekly, M=monthly
# range: 1M, 3M, 6M, 12M, 60M, ALL
TV_CHART_URL = "https://www.tradingview.com/chart/?symbol={exchange}%3A{symbol}&interval={interval}"
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
        self._last_price_range: Optional[Dict[str, float]] = None
    
    def extract_price_range_ocr(self, image_path: Path) -> Dict[str, Any]:
        """Extract price range from chart image using OCR.
        
        Focuses on the right side Y-axis where prices are displayed.
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dict with min_price, max_price, current_price (if found)
        """
        result = {
            "min_price": None,
            "max_price": None,
            "current_price": None,
            "price_range_text": None,
        }
        
        try:
            # Try to import pytesseract for OCR
            import pytesseract
            
            img = Image.open(image_path)
            width, height = img.size
            
            # Crop right side of image (Y-axis area) - last 10% of width
            right_margin = int(width * 0.90)
            y_axis_region = img.crop((right_margin, 0, width, height))
            
            # Run OCR on the Y-axis region
            ocr_text = pytesseract.image_to_string(y_axis_region)
            
            # Extract numbers that look like prices
            # Match patterns like: 150.00, 12.50, 1,234.56, etc.
            price_pattern = r'[\d,]+\.?\d*'
            matches = re.findall(price_pattern, ocr_text)
            
            # Clean and convert to floats
            prices = []
            for match in matches:
                try:
                    # Remove commas and convert
                    clean = match.replace(',', '')
                    if clean and clean != '.':
                        price = float(clean)
                        # Filter out unreasonable values (likely noise)
                        if 0.01 < price < 100000:
                            prices.append(price)
                except ValueError:
                    continue
            
            if prices:
                result["min_price"] = min(prices)
                result["max_price"] = max(prices)
                # Current price is often near the middle-right of chart
                # Use median as approximation
                prices_sorted = sorted(prices)
                mid_idx = len(prices_sorted) // 2
                result["current_price"] = prices_sorted[mid_idx]
                result["price_range_text"] = f"${result['min_price']:.2f} - ${result['max_price']:.2f}"
                
                logger.info(f"OCR extracted price range: {result['price_range_text']}")
            else:
                logger.warning("OCR could not extract prices from chart")
                
        except ImportError:
            logger.warning("pytesseract not installed - OCR disabled. Install with: pip install pytesseract")
            logger.warning("Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        
        self._last_price_range = result
        return result
    
    def get_last_price_range(self) -> Optional[Dict[str, Any]]:
        """Get the price range from the last captured chart."""
        return self._last_price_range
    
    async def capture(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        interval: str = "D",
        range_months: int = 3,
        width: int = 1920,
        height: int = 1080,
        extract_prices: bool = True,
    ) -> Path:
        """Capture a chart screenshot.
        
        Args:
            symbol: Ticker symbol (e.g., MELI)
            exchange: Exchange (e.g., NASDAQ)
            interval: Chart interval (D=daily, W=weekly)
            range_months: How many months of data to show
            width: Screenshot width
            height: Screenshot height
            extract_prices: Whether to run OCR to extract price range
            
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
                # Build URL with interval
                url = TV_CHART_URL.format(exchange=exchange, symbol=symbol, interval=interval)
                logger.info(f"Navigating to {url}")
                logger.info(f"Using {interval} interval ({range_months} months view)")
                
                await page.goto(url, timeout=self.timeout)
                
                # Wait for chart to load
                logger.info("Waiting for chart to load...")
                await page.wait_for_timeout(5000)
                
                # Try to set the date range to show last N months
                # TradingView keyboard shortcut: Alt+G opens "Go to" dialog
                # Or we can use the range selector
                try:
                    # Try clicking on date range selector and set to 3M
                    range_selector = await page.query_selector('[data-name="date-ranges-menu"]')
                    if range_selector:
                        await range_selector.click()
                        await page.wait_for_timeout(500)
                        # Click on 3M option
                        range_option = await page.query_selector(f'text="{range_months}M"')
                        if range_option:
                            await range_option.click()
                            await page.wait_for_timeout(2000)
                            logger.info(f"Set range to {range_months}M")
                except Exception as e:
                    logger.debug(f"Could not set date range via UI: {e}")
                
                # Wait for chart data to fully render
                await page.wait_for_timeout(3000)
                
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
                
                # Extract price range using OCR if requested
                if extract_prices:
                    self.extract_price_range_ocr(output_path)
                
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
        interval: str = "D",
        range_months: int = 3,
    ) -> Tuple[Path, Optional[Dict[str, Any]]]:
        """Synchronous wrapper for capture.
        
        Returns:
            Tuple of (image_path, price_range_dict)
        """
        path = asyncio.run(self.capture(symbol, exchange, interval, range_months))
        return path, self.get_last_price_range()


def get_chart_capture(output_dir: Path = None) -> ChartCapture:
    """Get ChartCapture instance."""
    return ChartCapture(output_dir=output_dir)
