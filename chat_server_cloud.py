#!/usr/bin/env python3
"""
OW-vsmart — Chat server (Cloud versie, FAISS editie)
- FastAPI + FAISS + DeepSeek API
- <512MB RAM — past op Render free tier
"""
import json, os, sys, pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
os.chdir(str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    print("❌ DEEPSEEK_API_KEY niet ingesteld")
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import numpy as np
from openai import OpenAI

app = FastAPI(title="OW-vsmart — bblthk Catalogus Chat")

# === Globals (lazy load voor geheugen) ===
model = None
index = None
metadata = []
documents = []

# Model wordt direct bij import geladen (voor snelle eerste request)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_model():
    global model
    pass  # model is al geladen bij opstart

def build_index():
    global index, metadata, documents
    cache_file = Path("faiss_index.pkl")
    
    if cache_file.exists():
        print("📦 FAISS index laden uit cache...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        index = data['index']
        metadata = data['metadata']
        documents = data['documents']
        print(f"✅ {len(metadata)} records geladen")
        return
    
    load_model()
    print("📚 Index bouwen uit catalogus_klimaat.json...")
    with open("catalogus_klimaat.json") as f:
        dataset = json.load(f)
    
    docs, metas = [], []
    for r in dataset["results"]:
        parts = [r['title']]
        if r['author']: parts.append(f"door {r['author']}")
        if r['description']: parts.append(r['description'])
        docs.append(" | ".join(parts))
        metas.append({
            "title": r['title'], "author": r['author'],
            "language": r['language'], "onLoan": r['onLoan'], "ppn": r['ppn']
        })
    
    print(f"   Embedden van {len(docs)} documenten...")
    embeddings = model.encode(docs, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    # Normaliseer voor cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    index = embeddings / norms
    metadata = metas
    documents = docs
    
    with open(cache_file, 'wb') as f:
        pickle.dump({'index': index, 'metadata': metadata, 'documents': documents}, f)
    
    print(f"✅ {len(metadata)} records in FAISS")

def search(query: str, n: int = 5):
    global model
    if index is None:
        build_index()
    
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    
    scores = np.dot(index, q_emb.T).flatten()
    top_idx = np.argsort(scores)[-n:][::-1]
    
    return [{
        "meta": metadata[i],
        "doc": documents[i],
        "score": float(scores[i])
    } for i in top_idx]

# === Startup: bouw index (async niet nodig, gebeurt 1x) ===
print("🚀 OW-vsmart cloud starting...")
build_index()

# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=CHAT_HTML)

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Geen vraag"}, status_code=400)
    
    results = search(question, n=5)
    
    books = []
    for r in results:
        m = r['meta']
        status = "beschikbaar" if not m['onLoan'] else "uitgeleend"
        desc = r['doc'].split(" | ")[-1][:200]
        books.append(f"- \"{m['title']}\" door {m['author'] or 'onbekend'} ({m['language']}, {status})\n  {desc}")
    context = "\n".join(books)
    
    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    
    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Je bent een bibliotheekassistent van bblthk Wageningen. Beantwoord de vraag op basis van de catalogus. Noem titels, auteurs en beschikbaarheid. Wees kort, vriendelijk, in het Nederlands."},
                {"role": "user", "content": f"Vraag: {question}\n\nCatalogus:\n{context}\n\nBeantwoord met titels, auteurs en beschikbaarheid."}
            ],
            temperature=0.7, max_tokens=500
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"(LLM niet bereikbaar: {e})\n\nGevonden boeken:\n{context}"
    
    return JSONResponse({
        "answer": answer,
        "sources": [{"title": r['meta']['title'], "author": r['meta']['author'], "available": not r['meta']['onLoan']} for r in results]
    })

CHAT_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>OW-vsmart — bblthk Catalogus Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f0eb;height:100vh;display:flex;flex-direction:column}
header{background:#2d5a27;color:white;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.2em;font-weight:600}header span{font-size:.8em;opacity:.8}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.message{max-width:75%;padding:12px 16px;border-radius:16px;line-height:1.5;font-size:.95em}
.message.user{align-self:flex-end;background:#2d5a27;color:white;border-bottom-right-radius:4px}
.message.assistant{align-self:flex-start;background:white;color:#333;border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.sources{margin-top:12px;padding-top:8px;border-top:1px solid #e0d8cf;font-size:.85em}
.source-item{display:flex;align-items:center;gap:6px;padding:4px 0}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot.available{background:#2d5a27}.dot.loaned{background:#c0392b}
.typing{opacity:.6;font-style:italic}
#input-area{padding:16px 20px;background:white;border-top:1px solid #e0d8cf;display:flex;gap:10px}
#question{flex:1;padding:12px 16px;border:2px solid #d5cdc0;border-radius:24px;font-size:.95em;outline:none;font-family:inherit}
#question:focus{border-color:#2d5a27}
#send{background:#2d5a27;color:white;border:none;padding:12px 24px;border-radius:24px;cursor:pointer;font-size:.95em;font-weight:600}
#send:hover{background:#1e3d1a}#send:disabled{opacity:.5;cursor:default}
</style>
</head>
<body>
<header><div style="font-size:1.4em">📚</div><div><h1>OW-vsmart</h1><span>bblthk Catalogus Chat</span></div></header>
<div id="chat"><div class="message assistant">Hallo! Ik ben de bblthk catalogus-assistent. Stel me een vraag over onze collectie.<br><br>Bijvoorbeeld: <em>"Welke boeken over klimaatverandering zijn geschikt voor kinderen?"</em></div></div>
<div id="input-area"><input id="question" placeholder="Stel je vraag..." onkeydown="if(event.key==='Enter')sendMessage()"><button id="send" onclick="sendMessage()">Verstuur</button></div>
<script>
async function sendMessage(){const i=document.getElementById('question'),q=i.value.trim();if(!q)return;const c=document.getElementById('chat'),b=document.getElementById('send');c.innerHTML+=`<div class="message user">${e(q)}</div>`;i.value='';b.disabled=true;const t='t_'+Date.now();c.innerHTML+=`<div class="message assistant typing" id="${t}">Zoeken...</div>`;c.scrollTop=c.scrollHeight;try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();document.getElementById(t)?.remove();let s='';if(d.sources?.length){s='<div class="sources">📚 <strong>Catalogus:</strong>';d.sources.forEach(x=>{const cl=x.available?'available':'loaned',st=x.available?'✅':'❌ uitgeleend';s+=`<div class="source-item"><span class="dot ${cl}"></span><strong>${e(x.title)}</strong> — ${e(x.author||'onbekend')} (${st})</div>`});s+='</div>'}c.innerHTML+=`<div class="message assistant">${e(d.answer).replace(/\\n/g,'<br>')}${s}</div>`}catch(err){document.getElementById(t)?.remove();c.innerHTML+=`<div class="message assistant">Fout: ${e(err.message)}</div>`}c.scrollTop=c.scrollHeight;b.disabled=false;i.focus()}function e(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8765))
    print(f"\n🚀 http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
