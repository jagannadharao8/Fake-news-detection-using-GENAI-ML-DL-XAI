import re

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Replace Streamlit with FastAPI
content = content.replace("The **Streamlit app** (`app.py`)", "The **FastAPI Web App** (`server.py`)")
content = content.replace("USER INPUT (app.py)", "USER INPUT (server.py)")
content = content.replace("Streamlit web app", "FastAPI & HTML/JS web app")
content = content.replace("app.py", "server.py")
content = content.replace("streamlit run app.py", "uvicorn server:app --reload")

# Add new features to Project Overview
new_features = """

### 🔥 Mega Upgrades Included in this Version:
- **FastAPI Backend:** Replaced heavy Streamlit framework with a lightning-fast asynchronous Python API.
- **Chrome Extension:** Highlight and fact-check text anywhere on the internet using the custom `chrome_extension/`.
- **History Database:** Automatic persistent storage of all analyses via an embedded SQLite database (`history.db`).
- **PDF Document Intelligence:** Upload full `.pdf` files to extract text and analyze entire documents.
"""

content = content.replace("## 2. System Architecture", new_features + "\n## 2. System Architecture")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("README.md patched successfully!")
