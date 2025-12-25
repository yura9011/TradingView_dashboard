
import unittest
from unittest.mock import MagicMock, patch
import torch

# Mock modules before importing components that use them
sys_modules_patch = patch.dict('sys.modules', {
    'qwen_vl_utils': MagicMock(),
    'transformers': MagicMock(),
    'torch': MagicMock()
})

class TestQwenIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_qwen_utils = MagicMock()
        self.mock_transformers = MagicMock()
        self.mock_torch = MagicMock()

        # Apply patch
        self.patcher = patch.dict('sys.modules', {
            'qwen_vl_utils': self.mock_qwen_utils,
            'transformers': self.mock_transformers,
            'torch': self.mock_torch
        })
        self.patcher.start()

        # Setup mocks
        self.mock_processor = MagicMock()
        self.mock_model = MagicMock()
        self.mock_transformers.AutoProcessor.from_pretrained.return_value = self.mock_processor
        self.mock_transformers.Qwen2VLForConditionalGeneration.from_pretrained.return_value = self.mock_model
        self.mock_torch.cuda.is_available.return_value = False

        # Now import the module under test
        from src.agents.specialists.base_agent_local import LocalModelManager, BaseAgentLocal
        self.LocalModelManager = LocalModelManager
        self.BaseAgentLocal = BaseAgentLocal

        # Reset singleton
        LocalModelManager._instance = None

    def tearDown(self):
        self.patcher.stop()

    def test_model_loading_qwen(self):
        """Test that LocalModelManager loads Qwen2-VL correctly."""
        manager = self.LocalModelManager.get_instance()
        model, processor = manager.load_model("Qwen/Qwen2-VL-7B-Instruct")

        # Verify transformers calls
        self.mock_transformers.AutoProcessor.from_pretrained.assert_called_with(
            "Qwen/Qwen2-VL-7B-Instruct",
            min_pixels=200704,  # 256*28*28
            max_pixels=1003520   # 1280*28*28
        )
        self.mock_transformers.Qwen2VLForConditionalGeneration.from_pretrained.assert_called_once()
        self.assertEqual(model, self.mock_model)

    def test_agent_generates_prompt_qwen(self):
        """Test that BaseAgentLocal constructs Qwen-compatible prompts."""
        # Setup specific mock behavior
        self.mock_processor.apply_chat_template.return_value = "mock_text_prompt"
        self.mock_qwen_utils.process_vision_info.return_value = ("mock_images", "mock_videos")

        # Mock inputs
        # The code iterates over inputs.input_ids, so we need to mock that structure
        # zip(inputs.input_ids, output_ids) expects both to be iterable
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.input_ids = [[1, 2, 3]] # List of lists (batch size 1)
        self.mock_processor.return_value = mock_inputs

        # Mock generation output
        # zip iterates over this too, so it should be a list of lists (batch size 1)
        # and each element should be longer than input_ids[i]
        self.mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        self.mock_processor.batch_decode.return_value = ["Parsed Response"]

        # Create concrete implementation of abstract class
        class TestAgent(self.BaseAgentLocal):
            def _parse_response(self, text):
                return {"result": text}

        # Create a dummy chart file
        with patch("pathlib.Path.exists", return_value=True):
            agent = TestAgent("pattern_detector.yaml")
            response = agent.analyze("dummy_chart.png")

            # Verify process_vision_info was called
            self.assertTrue(self.mock_qwen_utils.process_vision_info.called)
            call_args = self.mock_qwen_utils.process_vision_info.call_args
            messages = call_args[0][0]

            # Verify message structure
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            content = messages[0]["content"]
            self.assertEqual(len(content), 2)
            self.assertEqual(content[0]["type"], "image")
            self.assertEqual(content[1]["type"], "text")

            # Verify success
            self.assertTrue(response.success)
            self.assertEqual(response.parsed["result"], "Parsed Response")

if __name__ == "__main__":
    unittest.main()
