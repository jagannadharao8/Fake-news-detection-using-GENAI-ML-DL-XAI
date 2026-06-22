let startTime = 0;

document.getElementById('image').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (file) {
    document.getElementById('file-name').innerText = file.name;
    const reader = new FileReader();
    reader.onload = function(e) {
      document.getElementById('preview-img').src = e.target.result;
      document.getElementById('preview-img').style.display = 'block';
      document.getElementById('preview-placeholder').style.display = 'none';
    }
    reader.readAsDataURL(file);
  } else {
    document.getElementById('file-name').innerText = "Drag and drop file here";
    document.getElementById('preview-img').style.display = 'none';
    document.getElementById('preview-placeholder').style.display = 'block';
  }
});

document.getElementById('analyze-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const textVal = document.getElementById('text').value;
  const urlVal = document.getElementById('news-url').value;
  const imageFile = document.getElementById('image').files[0];
  
  if (!textVal.trim() && !urlVal.trim() && !imageFile) {
    alert("Please provide text, a URL, or an image to analyze.");
    return;
  }

  document.getElementById('input-section').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('loader').style.display = 'block';
  
  startTime = Date.now();
  const formData = new FormData();
  formData.append('text', textVal);
  formData.append('url', urlVal);
  if (imageFile) formData.append('image', imageFile);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    if (!res.ok) throw new Error("Server error");
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("An error occurred during analysis: " + err.message);
    resetApp();
  }
});

function renderResults(data) {
  const timeTaken = ((Date.now() - startTime) / 1000).toFixed(2);
  document.getElementById('time-taken').innerText = timeTaken;

  document.getElementById('loader').style.display = 'none';
  document.getElementById('results').style.display = 'block';
  
  // 1. Verdict
  const vMain = document.getElementById('verdict-main');
  vMain.className = 'verdict-main';
  const vStr = data.verdict.toLowerCase();
  
  if (vStr === 'fake') {
    vMain.innerText = '❌ Fake News';
    vMain.classList.add('v-fake');
  } else if (vStr === 'real') {
    vMain.innerText = '✅ Real News';
    vMain.classList.add('v-real');
  } else {
    vMain.innerText = '⚠️ Uncertain';
    vMain.classList.add('v-uncertain');
  }
  
  document.getElementById('conf-val').innerText = `${data.confidence}%`;
  
  let sentStr = "Neutral";
  if (data.sentiment) {
    let s = data.sentiment.sentiment;
    if (s > 0.6) sentStr = "Positive";
    else if (s < 0.4) sentStr = "Negative";
  }
  document.getElementById('sent-val').innerText = `${(data.sentiment?.sentiment*100 || 50).toFixed(0)}% ${sentStr}`;
  
  // Flag Reasons
  const flags = document.getElementById('flag-reasons');
  flags.innerHTML = '';
  if (data.clickbait?.clickbait > 0.5) flags.innerHTML += '<li>The headline uses emotionally charged language designed to create fear or urgency.</li>';
  if (data.vision?.verdict === 'mismatch') flags.innerHTML += '<li>Despite a matching image, the text contains misleading claims.</li>';
  if (data.rag?.verdict === 'contradicted') flags.innerHTML += '<li>Web sources contradict the claims in this content.</li>';
  if (data.rag?.verdict === 'mixed') flags.innerHTML += '<li>Web sources show mixed signals about the claims in this content.</li>';
  if (flags.innerHTML === '') flags.innerHTML = '<li style="color:#16a34a;">No major red flags detected.</li>';

  // 2. RAG
  let ragAlert = "🌐 " + (data.rag?.verdict || "Unverifiable").toUpperCase();
  document.getElementById('rag-alert').innerText = ragAlert;
  document.getElementById('rag-summary').innerText = data.rag?.evidence_summary || "No search results found.";

  // 3. XAI Word Influence
  const wp = document.getElementById('word-pills');
  wp.innerHTML = '';
  if (data.xai && data.xai.tokens) {
    for (let i=0; i<data.xai.tokens.length; i++) {
      let t = data.xai.tokens[i];
      let a = data.xai.attributions[i];
      let bg = '#ffffff';
      let border = '#e2e8f0';
      
      // Fake pushes red, real pushes green
      if (a > 0.1) { bg = '#fef2f2'; border = '#fca5a5'; }
      else if (a < -0.1) { bg = '#f0fdf4'; border = '#86efac'; }
      
      let sign = a > 0 ? '↑' : '';
      wp.innerHTML += `
        <div class="word-pill" style="background:${bg}; border-color:${border};">
          <span>${t}</span>
          <span class="word-score">${sign}${a.toFixed(3)}</span>
        </div>
      `;
    }
  } else {
    wp.innerHTML = '<span style="color:#94a3b8;">No text provided or model unavailable.</span>';
  }

  // 4. Root Causes
  const rc = document.getElementById('root-causes-list');
  rc.innerHTML = '';
  if (data.card && data.card.causes) {
    for (let c of data.card.causes) {
      rc.innerHTML += `<div style="margin-bottom:4px;">• ${c.text}</div>`;
    }
    if (data.card.causes.length === 0) rc.innerHTML = '🤔 No single strong red flag — verdict based on subtle patterns';
  } else {
    rc.innerHTML = '🤔 No single strong red flag — verdict based on subtle patterns';
  }

  // 5. Counterfactuals
  const cf = document.getElementById('cf-list');
  cf.innerHTML = '';
  if (data.card && data.card.counterfactuals) {
    const labels = {
        "remove_sensational"  : "If the writing style were calm and neutral...",
        "remove_contradiction": "If web evidence did not contradict the claims...",
        "remove_image_signal" : "If we ignored the image..."
    };
    for (const [key, val] of Object.entries(data.card.counterfactuals)) {
      let text = labels[key] || key;
      let newV = String(val.label || val.verdict || "Uncertain").toUpperCase();
      cf.innerHTML += `
        <div class="cf-card">
          <div class="cf-icon">⚡</div>
          <div class="cf-text">
            ${text}
            <strong>→ Verdict would change to ${newV}</strong>
          </div>
        </div>
      `;
    }
  }
  if (cf.innerHTML === '') {
    cf.innerHTML = '<span style="color:#94a3b8;">No counterfactual scenarios altered the verdict.</span>';
  }
}

function resetApp() {
  document.getElementById('results').style.display = 'none';
  document.getElementById('loader').style.display = 'none';
  document.getElementById('input-section').style.display = 'block';
}
