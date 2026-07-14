#!/usr/bin/env python3
"""
OW-vsmart — Chat server (lokaal, semantisch)
- FastAPI backend op poort 8765
- ChromaDB vector search (meertalige embeddings) + DeepSeek LLM
- Zelfde chat-UI en gedrag als de cloud-versie, maar met écht
  semantisch zoeken: "hulp bij lastige opgroeiende kinderen" vindt
  ook opvoedboeken zonder woordoverlap.
"""
import json, os, re, sys
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).parent
os.chdir(str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    print("❌ DEEPSEEK_API_KEY ontbreekt (.env)")
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

app = FastAPI(title="OW-vsmart — bblthk Catalogus Chat (lokaal)")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === ChromaDB setup (zelfde embedding-model als embed_and_query.py) ===
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bblthk_catalog", embedding_function=embed_fn)

print(f"📚 ChromaDB: {collection.count()} records")
print(f"🧠 Embeddings: {EMBED_MODEL} · LLM: DeepSeek")

# === Routes ===
@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/stats")
async def stats():
    with open("catalogus_klimaat.json") as f:
        dataset = json.load(f)
    theme_counts = Counter(r.get('theme', 'overig') for r in dataset["results"])
    return JSONResponse({
        "total": len(dataset["results"]),
        "themes": [{"theme": t, "count": c} for t, c in theme_counts.most_common()],
        "timestamp": dataset.get("timestamp", "")
    })

def clean_title(title):
    """Jaartal-suffix als '([2024])' of '(juni 2025)' van een catalogustitel strippen."""
    return re.sub(r'\s*\(\[?\s*\w*\s*\d{4}\]?\)\s*$', '', title).strip()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def fetch_book_info(title, author):
    """Zoek online aanvullende info over een boek via Gemini met Google Search-grounding.
    Geeft None terug bij ontbrekende key, quota of fouten — de UI laat het blok dan weg."""
    if not GEMINI_KEY:
        return None
    import requests as _rq
    t = clean_title(title)
    prompt = (
        f"Zoek informatie over het boek \"{t}\"{f' van {author}' if author else ''}. "
        "Geef in het Nederlands een korte samenvatting (3-5 zinnen) en, indien gevonden, "
        "uitgever en verschijningsjaar. Alleen platte tekst, geen opsommingstekens of markdown."
    )
    try:
        resp = _rq.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent",
            params={"key": GEMINI_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "tools": [{"google_search": {}}]},
            timeout=45,
        )
        resp.raise_for_status()
        cand = resp.json().get("candidates", [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
        if not text:
            return None
        chunks = cand.get("groundingMetadata", {}).get("groundingChunks", [])
        web = next((c.get("web") for c in chunks if c.get("web", {}).get("uri")), None)
        return {
            "title": t,
            "authors": author,
            "description": text[:900],
            "publisher": "", "publishedDate": "", "pageCount": None, "thumbnail": "",
            "link": web["uri"] if web else "",
            "source": web.get("title", "Google Zoeken") if web else "Google Zoeken",
        }
    except Exception as e:
        print(f"⚠️ Boekinfo ophalen faalde: {e}")
        return None

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Geen vraag"}, status_code=400)

    chroma_results = collection.query(query_texts=[question], n_results=8)
    results = list(zip(chroma_results['metadatas'][0], chroma_results['documents'][0]))

    books = []
    for i, (meta, doc) in enumerate(results, 1):
        status = "beschikbaar" if not meta['onLoan'] else "uitgeleend"
        desc = doc.split(' | ')[-1][:200]
        books.append(f"{i}. \"{meta['title']}\" door {meta['author'] or 'onbekend'} ({meta['language']}, {status})\n   {desc}")
    context = "\n".join(books) if books else "(geen relevante titels gevonden in de catalogus)"

    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    relevant = list(range(len(results)))  # fallback: alles tonen
    info_request = None
    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "Je bent een bibliotheekassistent van bblthk Wageningen. Beantwoord de vraag op basis "
                    "van de genummerde catalogusresultaten. Noem titels, auteurs en beschikbaarheid. Wees kort, "
                    "vriendelijk, in het Nederlands. De resultaten komen uit semantisch zoeken en kunnen missers "
                    "bevatten; negeer titels die niet bij de vraag passen. Sluit je antwoord af met een aparte "
                    "laatste regel in exact dit formaat: BRONNEN: gevolgd door de nummers van de resultaten "
                    "die echt bij de vraag passen, kommagescheiden (bijv. BRONNEN: 1,3). Passen er geen, "
                    "schrijf dan BRONNEN: geen. Vraagt de gebruiker om meer informatie, een samenvatting of "
                    "details over één specifieke titel, zet dan direct vóór de BRONNEN-regel een aparte regel "
                    "INFO: gevolgd door het nummer van die titel (bijv. INFO: 2)."
                )},
                {"role": "user", "content": f"Vraag: {question}\n\nCatalogus:\n{context}\n\nBeantwoord met titels, auteurs en beschikbaarheid, en eindig met de BRONNEN-regel."}
            ],
            temperature=0.7, max_tokens=450
        )
        answer = resp.choices[0].message.content
        mt = re.search(r'\n?\s*BRONNEN:\s*(.*?)\s*$', answer)
        if mt:
            answer = answer[:mt.start()].rstrip()
            nums = re.findall(r'\d+', mt.group(1))
            relevant = [int(n) - 1 for n in nums if 0 < int(n) <= len(results)] if nums else []
        mi = re.search(r'\n?\s*INFO:\s*(\d+)\s*$', answer)
        if mi:
            answer = answer[:mi.start()].rstrip()
            idx = int(mi.group(1)) - 1
            if 0 <= idx < len(results):
                info_request = idx
    except Exception as e:
        answer = f"(Fout: {e})\n\n{context}"

    bookinfo = None
    if info_request is not None:
        m0 = results[info_request][0]
        bookinfo = fetch_book_info(m0['title'], m0['author'])

    return JSONResponse({
        "answer": answer,
        "bookinfo": bookinfo,
        "sources": [
            {
                "title": meta['title'],
                "author": meta['author'],
                "language": meta['language'],
                "available": not meta['onLoan'],
                "ppn": meta['ppn']
            }
            for i, (meta, _) in enumerate(results) if i in relevant
        ]
    })


if __name__ == "__main__":
    print("\n🚀 Starten op http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
