"""
Gemini API inference wrapper for the Streamlit app.
Replaces the local Qwen2-VL model to allow CPU-only execution.
"""
from __future__ import annotations

import os
import json
import logging
import math
from typing import Any, Dict

from PIL import Image

log = logging.getLogger(__name__)

# Keep the original flag name so app.py doesn't break
QWEN_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY"))

_FALLBACK_CONSISTENCY: Dict[str, Any] = {
    "label":             "uncertain",
    "consistency_score":  0.5,
    "probs":             {"consistent": 0.33, "mismatched": 0.33, "uncertain": 0.34},
    "entropy":            math.log(3),
    "available":          False,
}

_FALLBACK_STANDALONE: Dict[str, Any] = {
    "label":      "uncertain",
    "fake_score":  0.5,
    "probs":      {"consistent": 0.33, "mismatched": 0.33, "uncertain": 0.34},
    "entropy":     math.log(3),
    "available":   False,
}

def _entropy(probs_dict: Dict[str, float]) -> float:
    ent = 0.0
    for v in probs_dict.values():
        p = max(v, 1e-12)
        ent -= p * math.log(p)
    return ent

def _call_gemini(image_path: str, prompt: str) -> Dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
        
        # Pulls GEMINI_API_KEY from environment
        client = genai.Client()
        
        img = Image.open(image_path).convert("RGB")
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "label": {
                            "type": "STRING", 
                            "enum": ["consistent", "mismatched", "uncertain"]
                        },
                        "probs": {
                            "type": "OBJECT",
                            "properties": {
                                "consistent": {"type": "NUMBER"},
                                "mismatched": {"type": "NUMBER"},
                                "uncertain": {"type": "NUMBER"}
                            },
                            "required": ["consistent", "mismatched", "uncertain"]
                        }
                    },
                    "required": ["label", "probs"]
                },
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        log.warning("Gemini VLM call failed: %s", e)
        return None

def predict_consistency(
    image_path: str,
    post_text: str,
    ocr_text: str = "",
) -> Dict[str, Any]:
    if not QWEN_AVAILABLE:
        return dict(_FALLBACK_CONSISTENCY)
        
    prompt = (
        f"POST: {post_text}\n"
        f"OCR: {ocr_text}\n"
        "Analyze the provided image and the text. Is the image consistent with the events, locations, and narrative described in the text?\n"
        "Return a JSON object with 'label' (one of: consistent, mismatched, uncertain) and 'probs' (a dictionary with confidence scores summing to 1.0 for each of the three labels)."
    )
    
    res = _call_gemini(image_path, prompt)
    if not res:
        return dict(_FALLBACK_CONSISTENCY)
        
    probs_dict = res["probs"]
    ent = _entropy(probs_dict)
    
    return {
        "label":             res["label"],
        "consistency_score": round(probs_dict.get("consistent", 0.0), 4),
        "probs":             probs_dict,
        "entropy":           round(ent, 4),
        "available":         True,
    }

def predict_standalone(image_path: str) -> Dict[str, Any]:
    if not QWEN_AVAILABLE:
        return dict(_FALLBACK_STANDALONE)
        
    prompt = (
        "Analyze the provided image. Does the image appear to be manipulated, AI-generated, highly sensationalized, or mismatched from reality?\n"
        "Treat 'mismatched' as suspicious/fake, and 'consistent' as realistic/benign.\n"
        "Return a JSON object with 'label' (one of: consistent, mismatched, uncertain) and 'probs' (a dictionary with confidence scores summing to 1.0 for each of the three labels)."
    )
    
    res = _call_gemini(image_path, prompt)
    if not res:
        return dict(_FALLBACK_STANDALONE)
        
    probs_dict = res["probs"]
    ent = _entropy(probs_dict)
    
    fake_score = probs_dict.get("mismatched", 0.0) * 0.7 + probs_dict.get("uncertain", 0.0) * 0.3
    
    return {
        "label":      res["label"],
        "fake_score": round(max(0.0, min(1.0, fake_score)), 4),
        "probs":      probs_dict,
        "entropy":    round(ent, 4),
        "available":  True,
    }
