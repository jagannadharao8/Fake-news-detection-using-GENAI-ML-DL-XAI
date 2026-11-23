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


    def test_provider_failures_have_safe_actionable_categories(self):
        vision = load_vision()
        class Failure(Exception):
            def __init__(self, code, message):
                super().__init__(message)
                self.code = code
        for code, message, category in [
            (400, "API key expired. private-marker", "invalid_credential"),
            (400, "API key not valid", "invalid_credential"),
            (403, "Access denied", "credential_or_permission"),
            (403, "SERVICE_DISABLED", "service_disabled"),
            (429, "Quota exhausted", "quota_or_rate_limit"),
            (404, "Model unavailable", "model_unavailable"),
            (400, "Invalid request body", "invalid_request")]:
            self.assertEqual(vision._failure_category(Failure(code, message)), category)

    def test_failure_log_excludes_raw_provider_details(self):
        vision = load_vision()
        class Failure(Exception):
            code = 400
        provider = types.ModuleType("google.genai")
        provider.types = types.SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
        provider.Client = Mock(side_effect=Failure("API key expired: private-marker"))
        google = types.ModuleType("google")
        google.genai = provider
        with patch.dict(sys.modules, {"google": google, "google.genai": provider}), self.assertLogs(vision.log, level="WARNING") as captured:
            self.assertIsNone(vision._call_gemini("test.png", "test"))
        message = " ".join(captured.output)
        self.assertIn("invalid_credential", message)
        self.assertIn("status=400", message)
        self.assertNotIn("private-marker", message)
