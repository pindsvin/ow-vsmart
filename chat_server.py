#!/usr/bin/env python3
"""
OW-vsmart — Chat server
- FastAPI backend op poort 8765
- ChromaDB vector search + Ollama LLM
- Web chat interface
"""
import json, os, sys
os.chdir("/Users/ron/Documents/ronOS/Projecten/OW-vsmart")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import chromadb
from chromadb.config import Settings as ChromaSettings
import httpx

app = FastAPI(title="OW-vsmart — bblthk Catalogus Chat")

# === ChromaDB setup ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bblthk_catalog")

# === LLM config ===
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"

print(f"📚 ChromaDB: {collection.count()} records")
print(f"🧠 LLM: Ollama/{OLLAMA_MODEL}")

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
    
    # 1. Vector search in catalogus
    chroma_results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    # 2. Formatteer context voor de LLM
    books_context = []
    for i, (meta, doc, dist) in enumerate(zip(
        chroma_results['metadatas'][0],
        chroma_results['documents'][0],
        chroma_results['distances'][0]
    )):
        status = "beschikbaar" if not meta['onLoan'] else "uitgeleend"
        books_context.append(
            f"{i+1}. \"{meta['title']}\" door {meta['author'] or 'onbekend'}"
            f" — {meta['language']} — {status}"
            f"\n   {doc.split(' | ')[-1][:200]}"
        )
    
    context = "\n\n".join(books_context)
    
    # 3. Prompt naar Ollama
    system_prompt = (
        "Je bent een behulpzame bibliotheekassistent voor de bibliotheek Wageningen (bblthk). "
        "Beantwoord de vraag van de gebruiker op basis van de catalogusresultaten hieronder. "
        "Vermeld titels, auteurs, en of het boek beschikbaar of uitgeleend is. "
        "Geef een kort, vriendelijk antwoord in het Nederlands. "
        "Als de resultaten niet perfect passen, zeg dat dan eerlijk en geef het beste wat je hebt."
    )
    
    user_message = (
        f"Vraag van de gebruiker: {question}\n\n"
        f"Gevonden boeken in de catalogus:\n{context}\n\n"
        f"Beantwoord de vraag. Noem de titels, auteurs, en beschikbaarheid."
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            ollama_resp = await client.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "stream": False
            })
            
            if ollama_resp.status_code == 200:
                ollama_data = ollama_resp.json()
                answer = ollama_data.get("message", {}).get("content", "Geen antwoord")
            else:
                answer = f"(LLM fout: {ollama_resp.status_code})"
    except Exception as e:
        answer = f"(Kan Ollama niet bereiken: {e}). Hier zijn de ruwe resultaten:\n\n{context}"
    
    # 4. Return antwoord + bronnen
    return JSONResponse({
        "answer": answer,
        "sources": [
            {
                "title": meta['title'],
                "author": meta['author'],
                "language": meta['language'],
                "available": not meta['onLoan'],
                "ppn": meta['ppn']
            }
            for meta in chroma_results['metadatas'][0]
        ]
    })

# === Chat HTML ===
CHAT_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OW-vsmart — bblthk Catalogus Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f0eb; height: 100vh; display: flex; flex-direction: column; }
header { background: #2d5a27; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
header h1 { font-size: 1.2em; font-weight: 600; }
header span { font-size: 0.8em; opacity: 0.8; }
#chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.message { max-width: 75%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; font-size: 0.95em; }
.message.user { align-self: flex-end; background: #2d5a27; color: white; border-bottom-right-radius: 4px; }
.message.assistant { align-self: flex-start; background: white; color: #333; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.message.assistant .sources { margin-top: 12px; padding-top: 8px; border-top: 1px solid #e0d8cf; font-size: 0.85em; }
.source-item { display: flex; align-items: center; gap: 6px; padding: 4px 0; }
.source-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.source-item .dot.available { background: #2d5a27; }
.source-item .dot.loaned { background: #c0392b; }
.typing { opacity: 0.6; font-style: italic; }
#input-area { padding: 16px 20px; background: white; border-top: 1px solid #e0d8cf; display: flex; gap: 10px; }
#question { flex: 1; padding: 12px 16px; border: 2px solid #d5cdc0; border-radius: 24px; font-size: 0.95em; outline: none; font-family: inherit; }
#question:focus { border-color: #2d5a27; }
#send { background: #2d5a27; color: white; border: none; padding: 12px 24px; border-radius: 24px; cursor: pointer; font-size: 0.95em; font-weight: 600; }
#send:hover { background: #1e3d1a; }
#send:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<header>
  <div style="font-size:1.4em">📚</div>
  <div><h1>OW-vsmart</h1><span>bblthk Catalogus Chat</span></div>
</header>
<div id="chat">
  <div class="message assistant">
    Hallo! Ik ben de bblthk catalogus-assistent. Stel me een vraag over onze collectie.<br><br>
    Bijvoorbeeld: <em>"Welke boeken hebben jullie over klimaatverandering voor kinderen van 10 jaar?"</em>
  </div>
</div>
<div id="input-area">
  <input id="question" placeholder="Stel je vraag over de bblthk-collectie..." onkeydown="if(event.key==='Enter')sendMessage()">
  <button id="send" onclick="sendMessage()">Verstuur</button>
</div>
<script>
async function sendMessage() {
  const input = document.getElementById('question');
  const question = input.value.trim();
  if (!question) return;
  
  const chat = document.getElementById('chat');
  const sendBtn = document.getElementById('send');
  
  // Toon gebruikerbericht
  chat.innerHTML += `<div class="message user">${escapeHtml(question)}</div>`;
  input.value = '';
  sendBtn.disabled = true;
  
  // Typing indicator
  const typingId = 'typing_' + Date.now();
  chat.innerHTML += `<div class="message assistant typing" id="${typingId}">Zoeken in catalogus...</div>`;
  chat.scrollTop = chat.scrollHeight;
  
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await resp.json();
    
    // Verwijder typing indicator
    document.getElementById(typingId)?.remove();
    
    // Bouw bronnen HTML
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = '<div class="sources">📚 <strong>Gevonden in catalogus:</strong>';
      data.sources.forEach(s => {
        const cls = s.available ? 'available' : 'loaned';
        const status = s.available ? 'beschikbaar' : 'uitgeleend';
        sourcesHtml += `<div class="source-item"><span class="dot ${cls}"></span> <strong>${escapeHtml(s.title)}</strong> — ${escapeHtml(s.author || 'onbekend')} (${status})</div>`;
      });
      sourcesHtml += '</div>';
    }
    
    chat.innerHTML += `<div class="message assistant">${escapeHtml(data.answer).replace(/\\n/g,'<br>')}${sourcesHtml}</div>`;
  } catch(e) {
    document.getElementById(typingId)?.remove();
    chat.innerHTML += `<div class="message assistant">Sorry, er ging iets mis: ${escapeHtml(e.message)}</div>`;
  }
  
  chat.scrollTop = chat.scrollHeight;
  sendBtn.disabled = false;
  input.focus();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n🚀 Starten op http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
