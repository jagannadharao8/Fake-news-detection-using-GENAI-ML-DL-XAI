import requests
import json

URL = "http://127.0.0.1:8000/api/analyze"

tests = [
    {
        "name": "Fake News Claim",
        "text": "BREAKING: The moon has exploded and debris is falling on New York City right now! Government is hiding the truth!"
    },
    {
        "name": "Real News Claim",
        "text": "The United States Federal Reserve announced today that it will leave interest rates unchanged for the third consecutive month."
    },
    {
        "name": "Empty Input",
        "text": ""
    }
]

print("Starting API tests...\n" + "-"*40)
for t in tests:
    print(f"Testing: {t['name']}")
    try:
        response = requests.post(URL, data={"text": t['text']})
        if response.status_code == 200:
            data = response.json()
            print(f"Verdict: {data.get('verdict')}")
            print(f"Confidence: {data.get('confidence')}%")
            print(f"RAG Verdict: {data.get('rag', {}).get('verdict')}")
            print(f"Has XAI Data: {'Yes' if data.get('xai') else 'No'}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    print("-" * 40)
