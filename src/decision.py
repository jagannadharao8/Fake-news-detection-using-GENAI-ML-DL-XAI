"""One conservative evidence policy for API verdicts and evidence cards."""
import math

def score(value, default=0.5):
    try:
        value = float(value)
        return min(1.0, max(0.0, value)) if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default

def decide(text_probs, rag, vision=None, has_image=False):
    vision = vision or {}
    mismatch = has_image and vision.get("available", False) and (
        vision.get("label") == "mismatched" or score(vision.get("consistency_score")) < 0.4)
    # Linguistic style cannot establish whether a real-world claim is true.
    if not rag.get("available") or rag.get("verdict") not in ("supported", "contradicted"):
        return {"label": "Uncertain", "score": None, "reason": "Insufficient or conflicting retrieved evidence."}
    if mismatch:
        return {"label": "Uncertain", "score": None, "reason": "Image and text conflict; manual verification is required."}
    verdict = rag["verdict"]
    strength = score(rag.get("rag_support" if verdict == "supported" else "rag_contradict"), 0)
    if strength < 0.65:
        return {"label": "Uncertain", "score": strength, "reason": "Retrieved evidence is too weak for a definitive label."}
    return {"label": "Real" if verdict == "supported" else "Fake", "score": strength,
            "reason": "Retrieved snippets " + ("support" if verdict == "supported" else "contradict") + " the claim; this is a heuristic, not proof."}

def evidence_card(decision, text_probs, rag, vision, has_image):
    scenarios = {}
    without_web = decide(text_probs, {}, vision, has_image)
    if without_web["label"] != decision["label"]:
        scenarios["remove_contradiction"] = without_web
    without_image = decide(text_probs, rag, {}, False)
    if without_image["label"] != decision["label"]:
        scenarios["remove_image_signal"] = without_image
    return {"final_label": decision["label"], "credibility": decision["score"],
            "causes": [{"text": decision["reason"]}], "counterfactuals": scenarios,
            "notes": "Same evidence policy as the top-level verdict. Scores are uncalibrated."}
