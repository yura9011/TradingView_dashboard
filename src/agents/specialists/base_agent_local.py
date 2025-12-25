"""
Base Agent Class for Local Models - Uses Qwen2-VL-7B-Instruct.
"""

import os
import yaml
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Default prompts directory
PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"

# Default model
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class LocalModelManager:
    """Thread-safe singleton manager for local model to avoid loading multiple times."""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _processor = None
    _model_name = None
    _model_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def load_model(self, model_name: str = DEFAULT_MODEL):
        """Load model if not already loaded (thread-safe)."""
        with self._model_lock:
            if self._model is not None and self._model_name == model_name:
                return self._model, self._processor
            
            logger.info(f"Loading local model: {model_name}")
            
            # Import Qwen2VL specific classes
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.bfloat16
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                logger.info("Using CPU (this will be slow)")
            
            # Load processor
            logger.info("Loading processor...")
            self._processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model
            logger.info("Loading model weights (this may take a while)...")
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map,
            }
            
            # Check if flash_attn is installed
            if torch.cuda.is_available():
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                except ImportError:
                    logger.info("Flash Attention not installed, using default attention")
            
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs,
            )
            
            self._model_name = model_name
            logger.info("Model loaded successfully")
            
            return self._model, self._processor
    
    def get_model(self):
        return self._model
    
    def get_processor(self):
        return self._processor


class BaseAgentLocal(ABC):
    """Base class for all specialist agents using local model."""
    
    def __init__(
        self,
        prompt_file: str,
        model_name: str = DEFAULT_MODEL,
    ):
        """Initialize base agent."""
        self.model_name = model_name
        self.system_prompt = self._load_prompt(prompt_file)
        self.model_manager = LocalModelManager.get_instance()
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load system prompt from YAML file."""
        prompt_path = PROMPTS_DIR / prompt_file
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config.get("system_prompt", "")
    
    def analyze(self, image_path: str, additional_context: str = "") -> AgentResponse:
        """Analyze chart image using Qwen2-VL model."""
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Chart image not found: {image_path}")
            
            model, processor = self.model_manager.load_model(self.model_name)
            
            logger.info(f"{self.__class__.__name__}: Loading and processing image...")
            image = Image.open(image_path).convert("RGB")
            
            # Log image info for debugging
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            
            full_prompt = f"{self.system_prompt}\n\n{additional_context}\n\nAnalyze this chart."
            
            # Qwen2-VL message format (without qwen_vl_utils)
            # When using PIL Image directly, only specify type in message
            # and pass the actual image to processor separately
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Image placeholder - actual image passed to processor
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ]
            
            logger.info(f"{self.__class__.__name__}: Applying chat template...")
            
            # Apply chat template to get the text prompt
            text_prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            
            logger.info(f"{self.__class__.__name__}: Processing inputs...")
            
            # Process text and image together
            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to model device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            logger.info(f"{self.__class__.__name__}: Starting generation on {device}...")
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                )
            
            logger.info(f"{self.__class__.__name__}: Generation complete, decoding...")
            
            # Extract only the generated tokens (exclude input tokens)
            generated_ids = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            
            raw_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            if not raw_text or not raw_text.strip():
                logger.warning(f"{self.__class__.__name__} returned empty response")
                return AgentResponse(
                    raw_text="",
                    parsed={},
                    success=False,
                    error="Model returned empty response",
                )
            
            logger.info(f"{self.__class__.__name__} analysis complete")
            logger.info(f"Raw response preview: {raw_text[:200]}...")
            
            parsed = self._parse_response(raw_text)
            
            return AgentResponse(
                raw_text=raw_text,
                parsed=parsed,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed: {e}", exc_info=True)
            return AgentResponse(
                raw_text="",
                parsed={},
                success=False,
                error=str(e),
            )
    
    @abstractmethod
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse agent-specific response format."""
        pass
