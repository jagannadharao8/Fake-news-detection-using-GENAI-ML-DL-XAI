"""Vision configuration contracts; provider and image boundaries are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

def load_vision():
    spec = importlib.util.spec_from_file_location("vision_under_test", Path(__file__).resolve().parents[1] / "src" / "vlm" / "infer.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class VisionTests(unittest.TestCase):
    def test_explicit_gemini_key_and_model_are_used(self):
        vision = load_vision()
        provider = types.ModuleType("google.genai")
        provider.types = types.SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
        client = Mock()
        client.models.generate_content.return_value = types.SimpleNamespace(text=json.dumps({
            "label": "uncertain", "probs": {"consistent": .1, "mismatched": .1, "uncertain": .8}}))
        provider.Client = Mock(return_value=client)
        google = types.ModuleType("google")
        google.genai = provider
        with patch.dict(sys.modules, {"google": google, "google.genai": provider}), patch.dict(os.environ, {
            "GEMINI_API_KEY": "unit-test-gemini-key", "GOOGLE_API_KEY": "different-unit-test-key",
            "GEMINI_MODEL": "configured-gemini-model"}), patch.object(vision.Image, "open"):
            result = vision._call_gemini("test.png", "Classify")
        provider.Client.assert_called_once_with(api_key="unit-test-gemini-key")
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"], "configured-gemini-model")
        self.assertEqual(result["label"], "uncertain")

    def test_unavailable_provider_abstains(self):
        vision = load_vision()
        with patch.object(vision, "QWEN_AVAILABLE", True), patch.object(vision, "_call_gemini", return_value=None):
            for result in [vision.predict_consistency("test.png", "test"), vision.predict_standalone("test.png")]:
                self.assertFalse(result["available"])
                self.assertEqual(result["label"], "uncertain")
