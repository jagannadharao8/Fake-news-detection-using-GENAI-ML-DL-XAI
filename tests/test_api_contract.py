"""HTTP contract tests with model/provider boundaries replaced, no network or secrets."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

def module(name, **members):
    mod = types.ModuleType(name)
    mod.__dict__.update(members)
    return mod

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stubs = {
            "torch": module("torch"),
            "huggingface_hub": module("huggingface_hub", snapshot_download=lambda **kwargs: None),
            "src.xai.text": module("src.xai.text", tokenizer=None, model=None, DEVICE="cpu", explain_text=lambda *a, **k: {"tokens": [], "attributions": []}),
            "src.vlm.infer": module("src.vlm.infer", predict_consistency=lambda *a: {}, predict_standalone=lambda *a: {}),
            "src.framing.sentiment": module("src.framing.sentiment", analyze_sentiment=lambda t: {"sentiment": .5, "subjectivity": .5}),
            "src.framing.clickbait": module("src.framing.clickbait", clickbait_score=lambda t: {"clickbait": 0}),
            "src.rag.claims": module("src.rag.claims", extract_claims=lambda *a: []),
            "src.rag.retrieve": module("src.rag.retrieve", retrieve=lambda *a: []),
        }
        cls.modules = patch.dict(sys.modules, cls.stubs)
        cls.modules.start()
        cls.server = importlib.import_module("server")
        cls.server.roberta_predict = lambda text: {"real": .9, "fake": .05, "uncertain": .05}
        cls.server.translate_to_english = lambda text: text
        cls.client = TestClient(cls.server.app)

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()

    def test_empty_input_rejected(self):
        self.assertEqual(self.client.post("/api/analyze").status_code, 400)

    def test_url_fetch_disabled(self):
        self.assertEqual(self.client.post("/api/analyze", data={"url": "http://127.0.0.1"}).status_code, 400)

    def test_missing_evidence_abstains(self):
        with patch.object(self.server, "live_rag_check", return_value={}):
            response = self.client.post("/api/analyze", data={"text": "An example claim."})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["verdict"], "Uncertain")
        self.assertEqual(data["verdict"], data["card"]["final_label"])
        self.assertIsNone(data["confidence"])

    def test_support_and_card_agree(self):
        rag = {"available": True, "verdict": "supported", "rag_support": .9}
        with patch.object(self.server, "live_rag_check", return_value=rag):
            data = self.client.post("/api/analyze", data={"text": "An example claim."}).json()
        self.assertEqual(data["verdict"], "Real")
        self.assertEqual(data["verdict"], data["card"]["final_label"])

    def test_history_not_exposed(self):
        self.assertEqual(self.client.get("/api/history").status_code, 410)

    def test_oversized_upload_rejected(self):
        response = self.client.post("/api/analyze", files={"image": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")})
        self.assertEqual(response.status_code, 413)

    def test_invalid_pdf_rejected(self):
        response = self.client.post("/api/analyze", files={"image": ("bad.pdf", b"invalid", "application/pdf")})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
