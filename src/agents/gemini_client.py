"""
Gemini AI Client for Chart Analysis
Uses gemini-flash-latest for multimodal analysis.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "gemini-flash-latest"


@dataclass
class AnalysisResult:
    """Result from chart analysis."""
    pattern_detected: str
    pattern_confidence: float
    trend: str
    support_level: Optional[float]
    resistance_level: Optional[float]
    fibonacci_level: Optional[str]
    pattern_box: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) as percentages
    analysis_summary: str
    raw_response: str


class GeminiClient:
    """Client for Gemini AI multimodal analysis."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model to use (default: gemini-flash-latest)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required. Set env var or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"Gemini client initialized with model: {model}")
    
    def analyze_chart(
        self,
        image_path: Union[str, Path],
        symbol: str,
        timeframe: str = "1D",
        additional_context: str = "",
    ) -> AnalysisResult:
        """Analyze a chart image for trading patterns.
        
        Args:
            image_path: Path to chart screenshot
            symbol: Ticker symbol being analyzed
            timeframe: Chart timeframe (e.g., "1D", "4H")
            additional_context: Extra context for analysis
            
        Returns:
            AnalysisResult with detected patterns and analysis
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {image_path}")
        
        # Load image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Create the prompt
        prompt = self._build_analysis_prompt(symbol, timeframe, additional_context)
        
        # Send to Gemini
        logger.info(f"Analyzing chart for {symbol} ({timeframe})")
        
        response = self.model.generate_content(
            [
                prompt,
                {"mime_type": self._get_mime_type(image_path), "data": image_data}
            ],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        raw_text = response.text
        logger.debug(f"Raw response: {raw_text}")
        
        # Parse the response
        return self._parse_analysis_response(raw_text)
    
    def analyze_chart_from_bytes(
        self,
        image_bytes: bytes,
        symbol: str,
        timeframe: str = "1D",
        mime_type: str = "image/png",
        additional_context: str = "",
    ) -> AnalysisResult:
        """Analyze chart from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
            symbol: Ticker symbol
            timeframe: Chart timeframe
            mime_type: Image MIME type
            additional_context: Extra context
            
        Returns:
            AnalysisResult
        """
        prompt = self._build_analysis_prompt(symbol, timeframe, additional_context)
        
        response = self.model.generate_content(
            [
                prompt,
                {"mime_type": mime_type, "data": image_bytes}
            ]
        )
        
        return self._parse_analysis_response(response.text)
    
    def _build_analysis_prompt(
        self,
        symbol: str,
        timeframe: str,
        additional_context: str = "",
    ) -> str:
        """Build the analysis prompt for chart analysis."""
        
        return f"""You are an expert technical analyst. Analyze this chart for {symbol} on the {timeframe} timeframe.

ANALYSIS REQUIREMENTS:
1. Identify any candlestick patterns (e.g., Bullish Engulfing, Doji, Hammer, etc.)
2. Identify chart patterns (e.g., Double Bottom, Head & Shoulders, Triangle, etc.)
3. Determine the current trend (uptrend, downtrend, sideways)
4. Identify key support and resistance levels
5. Note any Fibonacci retracement levels if visible
6. IMPORTANT: If you detect a pattern, estimate its location as percentage coordinates

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

RESPOND IN THIS EXACT FORMAT:
PATTERN: [pattern name or "none"]
CONFIDENCE: [0.0 to 1.0]
TREND: [up/down/sideways]
SUPPORT: [price level or "unknown"]
RESISTANCE: [price level or "unknown"]
FIBONACCI: [relevant fib level or "none"]
PATTERN_BOX: [x1,y1,x2,y2 as percentages 0-100 of image dimensions, or "none"]
SUMMARY: [2-3 sentence analysis of the chart and recommendation]

PATTERN_BOX should contain the approximate bounding box of the pattern you detected.
Example: If a Double Bottom pattern is in the lower-right area, respond with: PATTERN_BOX: 60,50,95,90

Be precise and analytical. Base your analysis only on what you can see in the chart."""


    def _parse_analysis_response(self, raw_text: str) -> AnalysisResult:
        """Parse structured response from Gemini."""
        
        lines = raw_text.strip().split("\n")
        result = {
            "pattern_detected": "none",
            "pattern_confidence": 0.0,
            "trend": "unknown",
            "support_level": None,
            "resistance_level": None,
            "fibonacci_level": None,
            "pattern_box": None,
            "analysis_summary": "",
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("PATTERN:"):
                result["pattern_detected"] = line.replace("PATTERN:", "").strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                    result["pattern_confidence"] = min(max(conf, 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("TREND:"):
                trend = line.replace("TREND:", "").strip().lower()
                if trend in ["up", "uptrend"]:
                    result["trend"] = "up"
                elif trend in ["down", "downtrend"]:
                    result["trend"] = "down"
                else:
                    result["trend"] = "sideways"
            elif line.startswith("SUPPORT:"):
                val = line.replace("SUPPORT:", "").strip()
                result["support_level"] = self._parse_price(val)
            elif line.startswith("RESISTANCE:"):
                val = line.replace("RESISTANCE:", "").strip()
                result["resistance_level"] = self._parse_price(val)
            elif line.startswith("FIBONACCI:"):
                fib = line.replace("FIBONACCI:", "").strip()
                if fib.lower() != "none":
                    result["fibonacci_level"] = fib
            elif line.startswith("PATTERN_BOX:"):
                box_str = line.replace("PATTERN_BOX:", "").strip()
                result["pattern_box"] = self._parse_box(box_str)
            elif line.startswith("SUMMARY:"):
                result["analysis_summary"] = line.replace("SUMMARY:", "").strip()
        
        # If summary spans multiple lines, capture the rest
        if "SUMMARY:" in raw_text:
            summary_start = raw_text.find("SUMMARY:") + 8
            result["analysis_summary"] = raw_text[summary_start:].strip()
        
        return AnalysisResult(
            pattern_detected=result["pattern_detected"],
            pattern_confidence=result["pattern_confidence"],
            trend=result["trend"],
            support_level=result["support_level"],
            resistance_level=result["resistance_level"],
            fibonacci_level=result["fibonacci_level"],
            pattern_box=result["pattern_box"],
            analysis_summary=result["analysis_summary"],
            raw_response=raw_text,
        )
    
    def _parse_box(self, val: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse PATTERN_BOX coordinates.
        
        Format: x1,y1,x2,y2 as percentages (0-100)
        """
        if not val or val.lower() in ["none", "n/a", "-"]:
            return None
        
        try:
            parts = [int(float(p.strip())) for p in val.replace(" ", "").split(",")]
            if len(parts) == 4:
                return tuple(parts)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _parse_price(self, val: str) -> Optional[float]:
        """Parse a price value from various formats.
        
        Handles: $1,976.00, 1976.00, 1,976, etc.
        """
        if not val or val.lower() in ["unknown", "n/a", "none", "-"]:
            return None
        
        try:
            # Remove currency symbols and commas
            cleaned = val.replace("$", "").replace(",", "").strip()
            # Extract first number found
            import re
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type from file extension."""
        ext = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")
    
    def test_connection(self) -> bool:
        """Test API connection.
        
        Returns:
            True if connection successful
        """
        try:
            response = self.model.generate_content("Say 'OK' if you can read this.")
            return "OK" in response.text.upper()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def get_gemini_client(
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> GeminiClient:
    """Factory function for GeminiClient.
    
    Args:
        api_key: API key (or use GEMINI_API_KEY env var)
        model: Model name
        
    Returns:
        Configured GeminiClient
    """
    return GeminiClient(api_key=api_key, model=model)
