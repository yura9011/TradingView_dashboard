"""
Qwen2-VL Local Client for Chart Analysis
Uses Qwen/Qwen2-VL-7B-Instruct for multimodal analysis.

NOTE: This is a standalone client for direct model usage.
For the multi-agent architecture, use the agents in src/agents/specialists/*_local.py
which share a single model instance via LocalModelManager.

This client can be used for:
- Testing the model directly
- Single-shot analysis without the multi-agent pipeline
- Custom integrations
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


@dataclass
class AnalysisResult:
    """Result from chart analysis."""
    pattern_detected: str
    pattern_confidence: float
    trend: str
    support_level: Optional[float]
    resistance_level: Optional[float]
    fibonacci_level: Optional[str]
    pattern_box: Optional[Tuple[int, int, int, int]]
    analysis_summary: str
    raw_response: str


class QwenVisionClient:
    """Client for Qwen2-VL local multimodal analysis."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        flash_attention: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        """Initialize Qwen Vision client.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16")
            flash_attention: Use flash attention 2 if available
            min_pixels: Minimum number of pixels for image resizing
            max_pixels: Maximum number of pixels for image resizing
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

        # Determine dtype
        if torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.flash_attention = flash_attention
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        logger.info(f"QwenVisionClient initialized (model will be loaded on first use)")

    def _load_model(self):
        """Load model and processor (lazy loading)."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.torch_dtype}")

        # Model kwargs
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device,
        }

        # Add flash attention if available and requested
        if self.flash_attention and torch.cuda.is_available():
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.info("Flash Attention not installed, using default attention")

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Load processor
        processor_kwargs = {}
        if self.min_pixels:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels:
            processor_kwargs["max_pixels"] = self.max_pixels

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs
        )

        logger.info(f"Model loaded successfully on {next(self.model.parameters()).device}")

    def analyze_chart(
        self,
        image_path: Union[str, Path],
        symbol: str,
        timeframe: str = "1D",
        additional_context: str = "",
        max_new_tokens: int = 1024,
    ) -> AnalysisResult:
        """Analyze a chart image for trading patterns.

        Args:
            image_path: Path to chart screenshot
            symbol: Ticker symbol being analyzed
            timeframe: Chart timeframe (e.g., "1D", "4H")
            additional_context: Extra context for analysis
            max_new_tokens: Maximum tokens to generate

        Returns:
            AnalysisResult with detected patterns and analysis
        """
        # Lazy load model
        self._load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {image_path}")

        # Create the prompt
        prompt = self._build_analysis_prompt(symbol, timeframe, additional_context)

        logger.info(f"Analyzing chart for {symbol} ({timeframe})")

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        raw_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        logger.debug(f"Raw response: {raw_text}")

        # Parse the response
        return self._parse_analysis_response(raw_text)

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
        """Parse structured response from model."""
        import re

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
        """Parse PATTERN_BOX coordinates."""
        import re
        if not val or val.lower() in ["none", "n/a", "-"]:
            return None

        try:
            cleaned = re.sub(r'[^\d,.]', '', val)
            parts = [int(float(p.strip())) for p in cleaned.split(",") if p.strip()]
            if len(parts) == 4:
                return tuple(parts)
        except (ValueError, TypeError):
            pass

        return None

    def _parse_price(self, val: str) -> Optional[float]:
        """Parse a price value from various formats."""
        import re
        if not val or val.lower() in ["unknown", "n/a", "none", "-"]:
            return None

        try:
            cleaned = val.replace("$", "").replace(",", "").strip()
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass

        return None

    def test_connection(self) -> bool:
        """Test model loading.

        Returns:
            True if model loads successfully
        """
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    def get_device_info(self) -> dict:
        """Get information about the device being used."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "model_loaded": self.model is not None,
        }

        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            if self.model is not None:
                info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.1f} GB"

        return info


def get_qwen_client(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> QwenVisionClient:
    """Factory function for QwenVisionClient.

    Args:
        model_name: HuggingFace model name
        device: Device to use

    Returns:
        Configured QwenVisionClient
    """
    return QwenVisionClient(model_name=model_name, device=device)
