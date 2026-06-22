---
title: Fake News Detection System
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# 🛡️ Fake News Detection System (Cloud-Hybrid AI Architecture)

A state-of-the-art multimodal fake news detection system that combines **local ML models** for fast linguistic text analysis with **cutting-edge Cloud AI APIs** for image consistency and live internet fact-checking. 

The system accepts a news post (text + optional image) and classifies it as **Fake**, **Real**, or **Uncertain**, providing a human-readable explanation of exactly how it arrived at its verdict.

---

## 🌟 The Cloud-Hybrid AI Stack

This project uses a hybrid architecture to balance speed, cost, and intelligence:

1. **Text Analysis (Local ML):** A 500MB **RoBERTa-base** model fine-tuned on the LIAR dataset runs locally to detect linguistic patterns of deception.
2. **Vision-Language Analysis (Cloud):** Uses **Google Gemini 3.5 Flash** to cross-reference uploaded images with the news text, detecting out-of-context or manipulated images.
3. **Live Web Fact-Checking / RAG (Cloud):** Uses **Serper** to scour the live internet for breaking news evidence, and uses **Groq (Llama-3.1)** to instantly analyze whether the internet evidence supports or contradicts the user's claim.

---

## 🚀 Live Demo

This application is fully deployed and accessible to the public via Hugging Face Spaces:
👉 **[Live Application Link](https://jagannadharao8-fake-news-detection.hf.space/)**

---

## ⚙️ How It Works (The Pipeline)

When a user submits an article, the system executes a 4-stage pipeline:

### Stage 1: Linguistic Profiling (RoBERTa & NLP)
- The local **RoBERTa** model analyzes the semantic structure of the text.
- **TextBlob** computes the sentiment polarity and subjectivity.
- A custom heuristic engine scores the text for clickbait patterns (e.g., ALL CAPS, sensational punctuation, superlatives).

### Stage 2: Multimodal Verification (Gemini Vision)
- If an image is provided, **Gemini 3.5 Flash** acts as an expert photo-journalist.
- It examines the image and the headline together, returning a strict assessment of whether the image legitimately represents the text, or if it is mismatched/manipulated.

### Stage 3: Retrieval-Augmented Generation (Groq + Serper)
- The system extracts the core factual claims from the text.
- It pings the **Google Search API (Serper)** to pull the top 5 live articles about those claims.
- It sends the search results to **Groq**, which reads the evidence and provides a boolean `Supports` or `Refutes` verdict.

### Stage 4: Fusion & Explainability (XGBoost)
- All signals (RoBERTa score, Gemini score, Groq score, Sentiment, Clickbait) are fed into an **XGBoost Meta-Learner**.
- XGBoost makes the final prediction: **Fake**, **Real**, or **Uncertain**.
- The system generates an Explainability Report so the user can understand exactly *why* the prediction was made.

---

## 🛠️ Installation & Local Setup

Want to run this system on your own machine?

### 1. Clone the Repository
```bash
git clone https://github.com/jagannadharao8/Fake-news-detection-using-GENAI-ML-DL-XAI.git
cd Fake-news-detection-using-GENAI-ML-DL-XAI
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory and add your free API keys:
```env
# For Image Verification (https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_key_here

# For Web Searching (https://serper.dev/)
SERPER_API_KEY=your_serper_key_here

# For RAG Fact-Checking (https://console.groq.com/)
GROQ_API_KEY=your_groq_key_here
```

### 5. Start the Server
```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 📁 Core Repository Structure

```
.
├── server.py                 # FastAPI backend & web server
├── static/                   # Frontend UI (HTML, CSS, JS)
├── src/                      # Core pipeline logic
│   ├── vlm/infer.py          # Gemini Vision integration
│   ├── rag/                  # Groq & Serper Fact-Checking
│   ├── text/                 # RoBERTa local NLP models
│   └── framing/              # Sentiment & Clickbait analysis
├── artifacts/                # Local ML weights (RoBERTa & XGBoost)
├── Dockerfile                # Hugging Face deployment config
└── requirements.txt          # Python dependencies
```

---

## 📝 Known Limitations
- The RAG system relies on Google Search. If a fake news story is highly viral and dominates search results, the system may occasionally struggle to find the truth.
- Free-tier API keys (Groq, Gemini) have rate limits. If you process too many requests in a minute, you may hit a cooldown period.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
