import unittest
from src.decision import decide, evidence_card, score

class DecisionTests(unittest.TestCase):
    def test_missing_evidence(self):
        self.assertEqual(decide({"fake": 0.99}, {})["label"], "Uncertain")
    def test_mixed_evidence(self):
        self.assertEqual(decide({}, {"available": True, "verdict": "mixed"})["label"], "Uncertain")
    def test_supported(self):
        self.assertEqual(decide({}, {"available": True, "verdict": "supported", "rag_support": .9})["label"], "Real")
    def test_contradicted(self):
        self.assertEqual(decide({}, {"available": True, "verdict": "contradicted", "rag_contradict": .9})["label"], "Fake")
    def test_weak(self):
        self.assertEqual(decide({}, {"available": True, "verdict": "supported", "rag_support": .5})["label"], "Uncertain")
    def test_image_changes_decision(self):
        rag = {"available": True, "verdict": "supported", "rag_support": .9}
        self.assertEqual(decide({}, rag, {"available": True, "consistency_score": .1}, True)["label"], "Uncertain")
    def test_card_agrees(self):
        for rag in [{}, {"available": True, "verdict": "supported", "rag_support": .9}, {"available": True, "verdict": "contradicted", "rag_contradict": .9}]:
            decision = decide({}, rag)
            card = evidence_card(decision, {}, rag, {}, False)
            self.assertEqual(card["final_label"], decision["label"])
            self.assertEqual(card["credibility"], decision["score"])
    def test_nonfinite_scores(self):
        self.assertEqual(score(float("nan"), 0), 0)
        self.assertEqual(score(float("inf"), 0), 0)

if __name__ == "__main__":
    unittest.main()
