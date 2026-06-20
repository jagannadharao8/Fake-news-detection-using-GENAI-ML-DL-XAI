document.addEventListener('DOMContentLoaded', () => {
  // Try to get selected text from the active tab
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.scripting.executeScript({
      target: {tabId: tabs[0].id},
      function: () => window.getSelection().toString()
    }, (results) => {
      if (results && results[0] && results[0].result) {
        document.getElementById('claim-text').value = results[0].result;
      }
    });
  });

  document.getElementById('analyze-btn').addEventListener('click', async () => {
    const text = document.getElementById('claim-text').value;
    if (!text) return;

    document.getElementById('analyze-btn').classList.add('hidden');
    document.getElementById('claim-text').classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');

    const formData = new FormData();
    formData.append('text', text);

    try {
      // Must be full URL since extension runs on chrome-extension://
      const res = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      document.getElementById('loader').classList.add('hidden');
      document.getElementById('results').classList.remove('hidden');

      const vBanner = document.getElementById('verdict-banner');
      const vStr = data.verdict.toLowerCase();
      if (vStr === 'fake') {
        vBanner.innerText = '❌ FAKE';
        vBanner.className = 'verdict-banner v-fake';
      } else if (vStr === 'real') {
        vBanner.innerText = '✅ REAL';
        vBanner.className = 'verdict-banner v-real';
      } else {
        vBanner.innerText = '⚠️ UNCERTAIN';
        vBanner.className = 'verdict-banner v-unc';
      }

      document.getElementById('conf-val').innerText = `${data.confidence}%`;
      document.getElementById('rag-val').innerText = (data.rag?.verdict || "Unverifiable").toUpperCase();

    } catch (e) {
      alert("Error contacting API. Is the local server running?");
      document.getElementById('loader').classList.add('hidden');
      document.getElementById('analyze-btn').classList.remove('hidden');
      document.getElementById('claim-text').classList.remove('hidden');
    }
  });

  document.getElementById('reset-btn').addEventListener('click', () => {
    document.getElementById('results').classList.add('hidden');
    document.getElementById('analyze-btn').classList.remove('hidden');
    document.getElementById('claim-text').classList.remove('hidden');
  });
});
