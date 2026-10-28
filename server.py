import os
import re
import json
import torch
import tempfile
import sqlite3
import datetime
import io
import uvicorn
import sys
from huggingface_hub import snapshot_download
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Optional

# ==========================================
# BOOTSTRAP: DOWNLOAD MISSING LFS MODELS
# ==========================================
safetensor_path = "artifacts/text_bas/hf_model/model.safetensors"
if not os.path.exists(safetensor_path) or os.path.getsize(safetensor_path) < 1000000:
    print("===== Downloading Missing ML Models (LFS Pointers detected) =====", flush=True)
    snapshot_download(
        repo_id='jagannadharao8/fake-news-detection',
        repo_type='space',
        local_dir='.',
        allow_patterns=['artifacts/**'],
        local_dir_use_symlinks=False
    )
    print("===== Models Downloaded! =====", flush=True)

import PyPDF2
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Models
from src.xai.text import tokenizer as _tok, model as _mdl, DEVICE as _DEVICE
from src.vlm.infer import predict_consistency, predict_standalone
from src.framing.sentiment import analyze_sentiment
from src.framing.clickbait import clickbait_score
from src.rag.claims import extract_claims
from src.rag.retrieve import retrieve

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROBERTA_LABELS = ["real", "fake", "uncertain"]

def roberta_predict(text: str) -> dict:
    if not text.strip():
        return {"real": 0.33, "fake": 0.33, "uncertain": 0.33}
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=256).to(_DEVICE)
    with torch.no_grad():
        logits = _mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return {label: round(probs[i], 4) for i, label in enumerate(_ROBERTA_LABELS)}

_RAG_CFG = {
    "claims": {"backend": "rule_based", "max_claims": 2, "min_chars": 20, "max_chars": 280, "dedupe": True},
    "retrieve": {"provider": "serper", "top_k": 3, "min_chars": 60, "serper": {"api_key_env": "SERPER_API_KEY", "country": "in", "num": 5, "allow_domains": [], "block_domains": []}}
}

def live_rag_check(text: str) -> dict:
    _FALLBACK = {
        "rag_support": 0.33, "rag_contradict": 0.33, "verdict": "unverifiable",
        "evidence_summary": "No web evidence found.", "key_sources": [], "available": False
    }
    if not text.strip(): return _FALLBACK
    try:
        claims = extract_claims(text, _RAG_CFG)
        if not claims: return _FALLBACK
        all_snippets = []
        for claim in claims[:2]:
            snippets = retrieve(claim, _RAG_CFG)
            for s in snippets[:3]: all_snippets.append({"claim": claim, "snippet": s})
        if not all_snippets: return _FALLBACK
        
        evidence_lines = []
        for i, item in enumerate(all_snippets[:5], 1):
            s = item["snippet"]
            evidence_lines.append(f"[{i}] {s.get('title','')} ({s.get('url','')})\n    {s.get('snippet','')}")
        
        evidence_block = "\n\n".join(evidence_lines)
        prompt = (f'News claim: "{text}"\n\nWeb search results:\n{evidence_block}\n\n'
                  "Analyse whether these results support or contradict the claim.\n"
                  "Respond ONLY with JSON:\n{"
                  '"support_score": <float>,"contradict_score": <float>,"verdict": "<supported|contradicted|mixed|unverifiable>","evidence_summary": "<summary>","key_sources": ["<domain>"]}')
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return _FALLBACK
        from groq import Groq
        client = Groq(api_key=api_key, timeout=45, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": "You are a precise fact-checking assistant. Return ONLY valid JSON. Start your response with { and end with }."},
                {"role": "user", "content": prompt}
            ], max_tokens=350, temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match: return _FALLBACK
        result = json.loads(json_match.group())
        result["rag_support"] = float(max(0.0, min(1.0, result.get("support_score", 0.33))))
        result["rag_contradict"] = float(max(0.0, min(1.0, result.get("contradict_score", 0.33))))
        result["available"] = True
        return result
    except Exception as e:
        print("RAG Error:", e)
        return _FALLBACK

from src.decision import decide, evidence_card

from src.xai.text import explain_text

def translate_to_english(input_text: str) -> str:
    if not input_text.strip(): return input_text
    try:
        from google import genai
        client = genai.Client()
        prompt = "Translate the following text to English. If it is already in English, return the original text without any translation or introductory remarks. ONLY output the English text:\n\n" + input_text[:3000]
        response = client.models.generate_content(
            model=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash'),
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print("Gemini translation error, trying Groq...", e)
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key, timeout=45, max_retries=1)
                response = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Translate the given text to English. If it's already in English, return the original text. Output ONLY the English text without quotes or explanations."},
                        {"role": "user", "content": input_text[:3000]}
                    ], max_tokens=1000, temperature=0.1
                )
                return response.choices[0].message.content.strip()
        except Exception as e2:
            print("Groq translation error:", e2)
    return input_text

@app.post("/api/analyze")
def analyze_endpoint(text: str = Form(""), url: str = Form(""), image: UploadFile = File(None)):
    import time
    if url.strip():
        raise HTTPException(status_code=400, detail="URL fetching is disabled; paste article text instead.")
    if not text.strip() and not (image and image.filename):
        raise HTTPException(status_code=400, detail="Provide text or an image/document.")
    image_path = None
    
    # Check if a PDF was uploaded
    if image and image.filename:
        filename = image.filename.lower()
        if filename.endswith(".pdf"):
            # Extract PDF text
            content = image.file.read(10 * 1024 * 1024 + 1)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Maximum upload size is 10 MB.")
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            except Exception:
                raise HTTPException(status_code=400, detail="The uploaded PDF could not be read.")
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += (page.extract_text() or "") + "\n"
            text = text + "\n" + extracted_text if text else extracted_text
        else:
            # It's an image
            content = image.file.read(10 * 1024 * 1024 + 1)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Maximum upload size is 10 MB.")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(content)
            tmp.close()
            image_path = tmp.name

    try:
        # Translate text to English if needed
        if text.strip():
            text = translate_to_english(text)
    
        # 1. RoBERTa (Text base)
        rob_probs = roberta_predict(text)
        
        # 2. RAG (Web check)
        rag = live_rag_check(text)
        
        # 3. Vision (Gemini)
        vlm_result = {}
        standalone_result = {}
        if image_path:
            if text.strip():
                vlm_result = predict_consistency(image_path, text)
            else:
                standalone_result = predict_standalone(image_path)
                
        # 4. Sentiment & Clickbait
        sent = analyze_sentiment(text) if text.strip() else {"sentiment": 0.5, "subjectivity": 0.5}
        click = clickbait_score(text) if text.strip() else {"clickbait": 0.0}
        
        # 5. Final Verdict blending
        mode = "image_only" if (image_path and not text.strip()) else "text_image"
        decision = decide(rob_probs, rag, vlm_result, bool(image_path))
        final_label, conf = decision["label"], decision["score"]
        
        # 6. XAI Features
        features = {
            "text_cls_score": rob_probs.get("fake", 0.5),
            "rag_support": rag.get("rag_support", 0.33),
            "rag_contradict": rag.get("rag_contradict", 0.33),
            "vlm_consistency": vlm_result.get("consistency_score", 0.5),
            "vlm_entropy": 1.0 - abs(vlm_result.get("consistency_score", 0.5) - 0.5) * 2.0,
            "ocr_match": 0.0,
            "sentiment": sent.get("sentiment", 0.5),
            "subjectivity": sent.get("subjectivity", 0.5),
            "clickbait": click.get("clickbait", 0.0),
            "text_length_norm": min(len(text) / 300.0, 1.0) if text else 0.0,
            "has_image": 1 if image_path else 0,
        }
        
        xai_data = None
        if text.strip():
            try:
                xai_data = explain_text(text, target_label="fake")
            except Exception:
                pass
                
        card = evidence_card(decision, rob_probs, rag, vlm_result, bool(image_path))
        
            
        # Do not persist submitted text in shared public history.
        return JSONResponse({
            "verdict": final_label,
            "confidence": round(conf * 100, 1) if conf is not None else None,
            "score_type": "uncalibrated evidence heuristic",
            "roberta": rob_probs,
            "rag": rag,
            "vision": vlm_result if text.strip() else standalone_result,
            "sentiment": sent,
            "clickbait": click,
            "xai": xai_data,
            "card": card
        })
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

@app.get("/api/history")
def get_history():
    raise HTTPException(status_code=410, detail="Shared analysis history is disabled for privacy.")

from fastapi.responses import FileResponse

@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok"})

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Mount the static frontend directory to the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
