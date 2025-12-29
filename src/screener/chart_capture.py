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
# interval: D=daily, W=weekly, M=monthly, 30=30min
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
            "ocr_success": False,
        }
        
        try:
            # Try to import pytesseract for OCR
            import pytesseract
            
            # Check if Tesseract is actually installed
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                logger.warning("=" * 50)
                logger.warning("⚠️  Tesseract OCR not installed!")
                logger.warning("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
                logger.warning("   Or run: winget install UB-Mannheim.TesseractOCR")
                logger.warning("   OCR disabled - model will estimate prices from chart")
                logger.warning("=" * 50)
                return result
            
            img = Image.open(image_path)
            width, height = img.size
            
            # Crop right side of image (Y-axis area) - last 8% of width
            right_margin = int(width * 0.92)
            y_axis_region = img.crop((right_margin, 0, width, height))
            
            # Run OCR on the Y-axis region with config for numbers
            ocr_config = '--psm 6 -c tessedit_char_whitelist=0123456789.,'
            ocr_text = pytesseract.image_to_string(y_axis_region, config=ocr_config)
            
            logger.debug(f"OCR raw text: {ocr_text}")
            
            # Extract numbers that look like prices
            # Match patterns like: 150.00, 12.50, 1,234.56, etc.
            price_pattern = r'[\d,]+\.?\d*'
            matches = re.findall(price_pattern, ocr_text)
            
            # Clean and convert to floats
            prices = []
            for match in matches:
                try:
                    # Remove commas and convert
                    clean = match.replace(',', '').strip()
                    if clean and clean != '.' and len(clean) > 0:
                        price = float(clean)
                        # Filter out unreasonable values (likely noise)
                        if 0.01 < price < 100000:
                            prices.append(price)
                except ValueError:
                    continue
            
            if len(prices) >= 2:
                # Validate: min should be significantly less than max
                min_p = min(prices)
                max_p = max(prices)
                
                # Sanity check: max should be at least 5% more than min
                if max_p > min_p * 1.05:
                    result["min_price"] = min_p
                    result["max_price"] = max_p
                    
                    # Current price is often the most frequent or near middle
                    prices_sorted = sorted(prices)
                    mid_idx = len(prices_sorted) // 2
                    result["current_price"] = prices_sorted[mid_idx]
                    result["price_range_text"] = f"${min_p:.2f} - ${max_p:.2f}"
                    result["ocr_success"] = True
                    
                    logger.info(f"✅ OCR extracted price range: {result['price_range_text']}")
                else:
                    logger.warning(f"OCR prices too close together: {min_p} - {max_p}, ignoring")
            else:
                logger.warning(f"OCR found insufficient prices: {prices}")
                
        except ImportError:
            logger.warning("pytesseract not installed - OCR disabled")
            logger.warning("Install with: pip install pytesseract")
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
                
                # Try to interact with the bottom toolbar for time range (1D, 1M, 3M, etc.)
                ranges_map = {
                    1: "1M",
                    3: "3M",
                    6: "6M",
                    12: "1Y",
                    60: "5Y"
                }
                
                # Logic for 30m interval -> prefer 1M range
                if interval == "30":
                     target_range = "1M"
                else:
                     target_range = ranges_map.get(range_months, "3M") # Default to 3M
                
                logger.info(f"Targeting time range: {target_range}")
                
                # Try clicking the specific range button in the bottom toolbar
                # Selectors common in TV: button with text "1M", "3M" etc. inside the date range container
                range_btn_selector = f'button:has-text("{target_range}")'
                
                try:
                    # Look for the button in the bottom toolbar specifically if possible, or generally on page
                    range_btn = await page.query_selector(range_btn_selector)
                    if range_btn:
                        logger.info(f"Found range button {target_range}, clicking...")
                        await range_btn.click()
                        await page.wait_for_timeout(2000) # Wait for redraw
                        range_set = True
                    else:
                        logger.warning(f"Range button {target_range} not found")
                except Exception as e:
                    logger.warning(f"Failed to click range button: {e}")

                # If direct button failed, try the "Go to..." or date range menu
                if not range_set:
                     # ... keep fallback ...
                     pass
                
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
        exchange: str = "", # Default empty to let TV find it
        interval: str = "30", # Default 30 min as requested
        range_months: int = 1, # Default 1 month for 30m
    ) -> Tuple[str, Dict[str, float]]:
        """Synchronous wrapper for capture."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            path = loop.run_until_complete(
                self.capture(
                    symbol=symbol, 
                    exchange=exchange, 
                    interval=interval, 
                    range_months=range_months
                )
            )
            # The capture method returns Path, and extract_price_range_ocr updates _last_price_range
            # We need to return the path as a string and the last price range
            return str(path), self.get_last_price_range() or {}
        finally:
            loop.close()


def get_chart_capture(output_dir: Path = None) -> ChartCapture:
    """Get ChartCapture instance."""
    return ChartCapture(output_dir=output_dir)
