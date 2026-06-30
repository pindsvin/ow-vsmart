#!/usr/bin/env python3
"""
OW-vsmart — Chat server (Cloud versie)
- FastAPI + ChromaDB + DeepSeek API
- Deploybaar op Render / Railway
- Leest DEEPSEEK_API_KEY uit .env
"""
import json, os, sys
from pathlib import Path

# Werk in de projectmap
PROJECT_DIR = Path(__file__).parent
os.chdir(str(PROJECT_DIR))

# Laad .env
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY or DEEPSEEK_KEY == "vul_hier_je_key_in":
    print("❌ DEEPSEEK_API_KEY niet ingesteld in .env")
    print("   1. Maak een key op https://platform.deepseek.com")
    print("   2. Zet 'm in Projecten/OW-vsmart/.env")
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

app = FastAPI(title="OW-vsmart — bblthk Catalogus Chat")

# === ChromaDB ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bblthk_catalog")
print(f"📚 ChromaDB: {collection.count()} records")

# === DeepSeek ===
llm = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com"
)
LLM_MODEL = "deepseek-chat"
print(f"🧠 LLM: DeepSeek/{LLM_MODEL}")

# === Static HTML ===
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=CHAT_HTML)

# === API ===
@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Geen vraag"}, status_code=400)
    
    # 1. Vector search
    results = collection.query(query_texts=[question], n_results=5)
    
    # 2. Context bouwen
    books = []
    for meta, doc in zip(results['metadatas'][0], results['documents'][0]):
        status = "beschikbaar" if not meta['onLoan'] else "uitgeleend"
        desc = doc.split(" | ")[-1][:200]
        books.append(
            f"- \"{meta['title']}\" door {meta['author'] or 'onbekend'} "
            f"({meta['language']}, {status})\n  {desc}"
        )
    context = "\n".join(books)
    
    # 3. LLM
    system = (
        "Je bent een bibliotheekassistent van de bibliotheek Wageningen (bblthk). "
        "Beantwoord de vraag op basis van de catalogusresultaten. "
        "Noem titels, auteurs en of een boek beschikbaar is. "
        "Wees kort, vriendelijk en in het Nederlands. "
        "Als resultaten niet perfect passen, zeg dat eerlijk."
    )
    
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Vraag: {question}\n\nCatalogus:\n{context}\n\nBeantwoord de vraag met titels, auteurs en beschikbaarheid."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"(LLM niet bereikbaar: {e})\n\nRuwe resultaten:\n{context}"
    
    return JSONResponse({
        "answer": answer,
        "sources": [
            {"title": m['title'], "author": m['author'], "available": not m['onLoan'], "ppn": m['ppn']}
            for m in results['metadatas'][0]
        ]
    })

# === HTML (zelfde als lokaal) ===
CHAT_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OW-vsmart — bblthk Catalogus Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f0eb;height:100vh;display:flex;flex-direction:column}
header{background:#2d5a27;color:white;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.2em;font-weight:600}
header span{font-size:.8em;opacity:.8}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.message{max-width:75%;padding:12px 16px;border-radius:16px;line-height:1.5;font-size:.95em}
.message.user{align-self:flex-end;background:#2d5a27;color:white;border-bottom-right-radius:4px}
.message.assistant{align-self:flex-start;background:white;color:#333;border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.sources{margin-top:12px;padding-top:8px;border-top:1px solid #e0d8cf;font-size:.85em}
.source-item{display:flex;align-items:center;gap:6px;padding:4px 0}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot.available{background:#2d5a27}
.dot.loaned{background:#c0392b}
.typing{opacity:.6;font-style:italic}
#input-area{padding:16px 20px;background:white;border-top:1px solid #e0d8cf;display:flex;gap:10px}
#question{flex:1;padding:12px 16px;border:2px solid #d5cdc0;border-radius:24px;font-size:.95em;outline:none;font-family:inherit}
#question:focus{border-color:#2d5a27}
#send{background:#2d5a27;color:white;border:none;padding:12px 24px;border-radius:24px;cursor:pointer;font-size:.95em;font-weight:600}
#send:hover{background:#1e3d1a}
#send:disabled{opacity:.5;cursor:default}
</style>
</head>
<body>
<header><div style="font-size:1.4em">📚</div><div><h1>OW-vsmart</h1><span>bblthk Catalogus Chat</span></div></header>
<div id="chat">
<div class="message assistant">
Hallo! Ik ben de bblthk catalogus-assistent. Stel me een vraag over onze collectie.<br><br>
Bijvoorbeeld: <em>"Welke boeken hebben jullie over klimaatverandering voor kinderen?"</em>
</div></div>
<div id="input-area">
<input id="question" placeholder="Stel je vraag over de bblthk-collectie..." onkeydown="if(event.key==='Enter')sendMessage()">
<button id="send" onclick="sendMessage()">Verstuur</button>
</div>
<script>
async function sendMessage(){
const i=document.getElementById('question'),q=i.value.trim();
if(!q)return;
const c=document.getElementById('chat'),b=document.getElementById('send');
c.innerHTML+=`<div class="message user">${escapeHtml(q)}</div>`;
i.value='';b.disabled=true;
const tid='t_'+Date.now();
c.innerHTML+=`<div class="message assistant typing" id="${tid}">Zoeken...</div>`;
c.scrollTop=c.scrollHeight;
try{
const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();
document.getElementById(tid)?.remove();
let sh='';
if(d.sources?.length){sh='<div class="sources">📚 <strong>Catalogus:</strong>';
d.sources.forEach(s=>{const cl=s.available?'available':'loaned',st=s.available?'✅':'❌ uitgeleend';
sh+=`<div class="source-item"><span class="dot ${cl}"></span><strong>${escapeHtml(s.title)}</strong> — ${escapeHtml(s.author||'onbekend')} (${st})</div>`});
sh+='</div>'}
c.innerHTML+=`<div class="message assistant">${escapeHtml(d.answer).replace(/\\n/g,'<br>')}${sh}</div>`;
}catch(e){document.getElementById(tid)?.remove();c.innerHTML+=`<div class="message assistant">Fout: ${escapeHtml(e.message)}</div>`}
c.scrollTop=c.scrollHeight;b.disabled=false;i.focus()}
function escapeHtml(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8765))
    print(f"\n🚀 Starten op poort {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

# === Startup: rebuild ChromaDB if empty ===
if collection.count() == 0:
    print("📦 ChromaDB leeg — herbouwen uit catalogus_klimaat.json...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open("catalogus_klimaat.json") as f:
        dataset = json.load(f)
    
    docs, metas, ids = [], [], []
    for r in dataset["results"]:
        parts = [r['title']]
        if r['author']: parts.append(f"door {r['author']}")
        if r['description']: parts.append(r['description'])
        docs.append(" | ".join(parts))
        metas.append({"title": r['title'], "author": r['author'], "type": r['type'], "language": r['language'], "onLoan": r['onLoan'], "ppn": r['ppn']})
        ids.append(f"ppn_{r['ppn']}")
    
    for i in range(0, len(docs), 32):
        collection.add(documents=docs[i:i+32], metadatas=metas[i:i+32], ids=ids[i:i+32])
    
    print(f"✅ {collection.count()} records in ChromaDB")
