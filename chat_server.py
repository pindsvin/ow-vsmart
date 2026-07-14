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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import chromadb
from chromadb.config import Settings as ChromaSettings
import httpx

app = FastAPI(title="OW-vsmart — bblthk Catalogus Chat")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === ChromaDB setup ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bblthk_catalog")

# === LLM config ===
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"

print(f"📚 ChromaDB: {collection.count()} records")
print(f"🧠 LLM: Ollama/{OLLAMA_MODEL}")

# === Routes ===
@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/stats")
async def stats():
    from collections import Counter
    with open("catalogus_klimaat.json") as f:
        dataset = json.load(f)
    theme_counts = Counter(r.get('theme', 'overig') for r in dataset["results"])
    return JSONResponse({
        "total": len(dataset["results"]),
        "themes": [{"theme": t, "count": c} for t, c in theme_counts.most_common()],
        "timestamp": dataset.get("timestamp", "")
    })

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



if __name__ == "__main__":
    print("\n🚀 Starten op http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
