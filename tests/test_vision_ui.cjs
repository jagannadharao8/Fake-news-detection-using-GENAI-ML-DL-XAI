const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM}=require('jsdom');
const dom=new JSDOM(fs.readFileSync('static/index.html','utf8'),{runScripts:'outside-only',url:'https://example.test/'});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
dom.window.scrollTo=function(){};
dom.window.eval(fs.readFileSync('static/script.js','utf8'));
const base={verdict:'Uncertain',confidence:null,sentiment:{sentiment:.5},clickbait:{clickbait:0},
  rag:{verdict:'unverifiable',evidence_summary:'Insufficient evidence'},xai:null,card:{causes:[],counterfactuals:{}}};
function render(vision){dom.window.renderResults({...base,vision});return dom.window.document;}
let doc=render({});
assert.equal(doc.getElementById('vision-card').style.display,'none');
doc=render({available:false,label:'uncertain'});
assert.match(doc.getElementById('vision-summary').textContent,/unavailable/);
assert.equal(doc.getElementById('vision-provider').textContent,'');
doc=render({available:true,label:'mismatched',provider:'groq',model:'qwen/qwen3.8-27b'});
assert.match(doc.getElementById('vision-summary').textContent,/completed: mismatched/);
assert.match(doc.getElementById('vision-provider').textContent,/groq/);
assert.match(doc.getElementById('flag-reasons').textContent,/may not match/);
doc=render({available:true,label:'uncertain',provider:'<script>bad()</script>',model:'test'});
assert.equal(doc.getElementById('vision-provider').querySelector('script'),null);
assert.match(doc.getElementById('vision-provider').textContent,/<script>/);
console.log('Vision UI: no-image, unavailable, assessed mismatch and safe metadata rendering passed.');
dom.window.close();
