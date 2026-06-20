import os
import re
import json
import torch
import tempfile
import sqlite3
import datetime
import io
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import PyPDF2

# Load environment variables
load_dotenv()

# Models
from src.xai.text import tokenizer as _tok, model as _mdl, DEVICE as _DEVICE
from src.vlm.infer import predict_consistency, predict_standalone
from src.framing.sentiment import analyze_sentiment
from src.framing.clickbait import clickbait_score
from src.rag.claims import extract_claims
from src.rag.retrieve import retrieve

# Initialize Database
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text TEXT,
            verdict TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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

def combined_verdict(r_probs, rag_result, standalone_result, image_mode):
    fake_prob = r_probs.get("fake", 0.5)
    if image_mode == "image_only" and standalone_result.get("available"):
        fake_prob = standalone_result.get("fake_score", 0.5) * 0.65 + fake_prob * 0.35
        
    if rag_result.get("available"):
        web_verdict = rag_result.get("verdict", "unverifiable")
        if web_verdict == "supported":
            fake_prob *= (0.25 + 0.20 * (1.0 - rag_result.get("rag_support", 0.33)))
        elif web_verdict == "contradicted":
            fake_prob += (1.0 - fake_prob) * (0.35 * max(rag_result.get("rag_contradict", 0.33), 0.5))
        elif web_verdict == "mixed":
            fake_prob *= 0.85
            
    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = 1.0 - fake_prob
    if fake_prob > 0.60: return "Fake", fake_prob
    if real_prob > 0.60: return "Real", real_prob
    if fake_prob > 0.40 and fake_prob >= real_prob: return "Fake", fake_prob
    if real_prob > 0.40: return "Real", real_prob
    return "Uncertain", max(fake_prob, real_prob)

from src.xai.text import explain_text
from src.xai.card_builder import build_evidence_card
from src.xai.counterfactuals import run_counterfactuals

@app.post("/api/analyze")
async def analyze_endpoint(text: str = Form(""), image: UploadFile = File(None)):
    import time
    image_path = None
    
    # Check if a PDF was uploaded
    if image and image.filename:
        filename = image.filename.lower()
        if filename.endswith(".pdf"):
            # Extract PDF text
            content = await image.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
            text = text + "\n" + extracted_text if text else extracted_text
        else:
            # It's an image
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            content = await image.read()
            tmp.write(content)
            tmp.close()
            image_path = tmp.name

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
    final_label, conf = combined_verdict(rob_probs, rag, standalone_result, mode)
    
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
            
    card = {}
    try:
        card = build_evidence_card({
            "id": f"live_{int(time.time())}",
            "image_path": image_path,
            "post_text" : text,
            "features"  : features,
        })
    except Exception as e:
        print("Card error:", e)
    
    if image_path:
        os.remove(image_path)
        
    # Save to history database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_text = text[:500] + "..." if len(text) > 500 else text
        if not save_text and image: save_text = f"[Image/Document: {image.filename}]"
        
        cursor.execute("INSERT INTO history (timestamp, text, verdict, confidence) VALUES (?, ?, ?, ?)",
                       (dt_str, save_text, final_label, conf * 100))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Error:", e)

    return JSONResponse({
        "verdict": final_label,
        "confidence": round(conf * 100, 1),
        "roberta": rob_probs,
        "rag": rag,
        "vision": vlm_result if text.strip() else standalone_result,
        "sentiment": sent,
        "clickbait": click,
        "xai": xai_data,
        "card": card
    })

@app.get("/api/history")
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Mount the static frontend directory to the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
