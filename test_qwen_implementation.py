"""
Test script for Qwen2-VL implementation.
Run: python test_qwen_implementation.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test 1: Verify all imports work."""
    print("\n" + "=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print(f"  ❌ torch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✅ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ❌ transformers: {e}")
        return False
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"  ✅ Qwen2VLForConditionalGeneration")
        print(f"  ✅ AutoProcessor")
    except ImportError as e:
        print(f"  ❌ Qwen2VL classes: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"  ✅ PIL.Image")
    except ImportError as e:
        print(f"  ❌ PIL: {e}")
        return False
    
    try:
        from src.agents.specialists.base_agent_local import BaseAgentLocal, LocalModelManager, AgentResponse
        print(f"  ✅ BaseAgentLocal")
        print(f"  ✅ LocalModelManager")
    except ImportError as e:
        print(f"  ❌ Local agents: {e}")
        return False
    
    try:
        from src.agents.coordinator_local import CoordinatorAgentLocal
        print(f"  ✅ CoordinatorAgentLocal")
    except ImportError as e:
        print(f"  ❌ Coordinator: {e}")
        return False
    
    print("\n  ✅ All imports successful!")
    return True


def test_model_loading():
    """Test 2: Verify model can be loaded."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Loading")
    print("=" * 60)
    
    try:
        from src.agents.specialists.base_agent_local import LocalModelManager
        
        print("  Loading model (this may take a while on first run)...")
        manager = LocalModelManager.get_instance()
        model, processor = manager.load_model()
        
        print(f"  ✅ Model loaded successfully")
        print(f"     Model type: {type(model).__name__}")
        print(f"     Processor type: {type(processor).__name__}")
        
        # Check model device
        device = next(model.parameters()).device
        print(f"     Model device: {device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processor_format():
    """Test 3: Verify processor accepts our message format."""
    print("\n" + "=" * 60)
    print("TEST 3: Processor Message Format")
    print("=" * 60)
    
    try:
        from src.agents.specialists.base_agent_local import LocalModelManager
        from PIL import Image
        import torch
        
        manager = LocalModelManager.get_instance()
        model, processor = manager.load_model()
        
        # Create a simple test image (red square)
        print("  Creating test image...")
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test the message format we use
        print("  Testing message format...")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What color is this image?"},
                ],
            }
        ]
        
        # Apply chat template
        print("  Applying chat template...")
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        print(f"  ✅ Chat template applied")
        print(f"     Template preview: {text_prompt[:100]}...")
        
        # Process inputs
        print("  Processing inputs...")
        inputs = processor(
            text=[text_prompt],
            images=[test_image],
            padding=True,
            return_tensors="pt",
        )
        print(f"  ✅ Inputs processed")
        print(f"     Input keys: {list(inputs.keys())}")
        print(f"     input_ids shape: {inputs.input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """Test 4: Verify model can generate output."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Generation")
    print("=" * 60)
    
    try:
        from src.agents.specialists.base_agent_local import LocalModelManager
        from PIL import Image
        import torch
        
        manager = LocalModelManager.get_instance()
        model, processor = manager.load_model()
        
        # Create test image
        print("  Creating test image (blue square)...")
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the main color in this image? Answer in one word."},
                ],
            }
        ]
        
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[test_image],
            padding=True,
            return_tensors="pt",
        )
        
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate
        print("  Generating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
            )
        
        # Decode
        generated_ids = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        print(f"  ✅ Generation successful")
        print(f"     Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_chart_image():
    """Test 5: Test with an actual chart image if available."""
    print("\n" + "=" * 60)
    print("TEST 5: Chart Image Analysis")
    print("=" * 60)
    
    # Look for existing chart images
    chart_dirs = [
        Path("data/charts"),
        Path("data"),
    ]
    
    chart_image = None
    for chart_dir in chart_dirs:
        if chart_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images = list(chart_dir.glob(ext))
                if images:
                    chart_image = images[0]
                    break
        if chart_image:
            break
    
    if not chart_image:
        print("  ⚠️  No chart images found in data/charts/")
        print("     Skipping chart analysis test")
        print("     Run a full analysis first to generate chart images")
        return True  # Not a failure, just skip
    
    print(f"  Found chart: {chart_image}")
    
    try:
        from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
        
        print("  Running PatternDetectorAgentLocal...")
        detector = PatternDetectorAgentLocal()
        result = detector.analyze(str(chart_image))
        
        print(f"  ✅ Analysis complete")
        print(f"     Success: {result.success}")
        print(f"     Pattern: {result.parsed.get('pattern', 'N/A')}")
        print(f"     Confidence: {result.parsed.get('confidence', 0):.0%}")
        print(f"     Response preview: {result.raw_text[:200] if result.raw_text else 'Empty'}...")
        
        if result.error:
            print(f"     Error: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"  ❌ Chart analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  QWEN2-VL IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n❌ Import test failed. Fix imports before continuing.")
        return
    
    # Test 2: Model Loading
    results['model_loading'] = test_model_loading()
    if not results['model_loading']:
        print("\n❌ Model loading failed. Check GPU/memory.")
        return
    
    # Test 3: Processor Format
    results['processor'] = test_processor_format()
    
    # Test 4: Generation
    results['generation'] = test_generation()
    
    # Test 5: Chart Analysis
    results['chart'] = test_with_chart_image()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("  ✅ ALL TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
