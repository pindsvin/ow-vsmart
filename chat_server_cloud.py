#!/usr/bin/env python3
"""
OW-vsmart — Chat server (Render editie, pure Python)
- Handgeschreven TF-IDF — zero dependencies behalve stdlib
- DeepSeek API voor antwoorden
"""
import json, os, re, sys, math
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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from openai import OpenAI
from collections import Counter

app = FastAPI(title="OW-vsmart")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Pure Python TF-IDF ===
print("📚 Laden catalogus...")
with open("catalogus_klimaat.json") as f:
    dataset = json.load(f)

docs, metas = [], []
for r in dataset["results"]:
    parts = [r['title']]
    if r['author']: parts.append(f"door {r['author']}")
    if r['description']: parts.append(r['description'])
    docs.append(" ".join(parts).lower())
    metas.append({
        "title": r['title'], "author": r['author'],
        "language": r['language'], "onLoan": r['onLoan'], "ppn": r['ppn']
    })

STOPWORDS = {
    "de", "het", "een", "en", "van", "in", "op", "te", "met", "voor", "is", "dat", "die",
    "zijn", "aan", "bij", "ook", "naar", "uit", "om", "als", "dan", "maar", "wat", "wie",
    "hoe", "waar", "er", "hier", "daar", "dit", "deze", "niet", "wel", "geen", "iets",
    "veel", "meer", "je", "jij", "jouw", "ik", "mijn", "we", "wij", "ons", "onze", "ze",
    "zij", "hun", "hij", "haar", "hem", "u", "uw", "over", "onder", "tussen", "door",
    "tegen", "zonder", "heb", "hebben", "heeft", "had", "kan", "kunnen", "wil", "willen",
    "moet", "moeten", "mag", "mogen", "zal", "zullen", "zou", "wordt", "worden", "werd",
    "was", "waren", "ben", "bent", "goed", "goede", "mooi", "mooie", "leuk", "leuke",
    "boek", "boeken",
}

def tokenize(text):
    words = [w.strip(".,!?()[]\"':;") for w in text.lower().split()]
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]

# Bouw vocabulary
print("📊 TF-IDF index bouwen...")
all_tokens = [tokenize(d) for d in docs]
doc_freq = Counter()
for t in all_tokens:
    doc_freq.update(set(t))
vocab = sorted(w for w, _ in doc_freq.most_common(3000))  # max 3000 features, meest voorkomende eerst

N = len(docs)
df = Counter()
for t in all_tokens:
    df.update(set(t))

# TF-IDF vectors
def compute_tfidf(tokens):
    tf = Counter(tokens)
    vec = []
    for w in vocab:
        if w in tf:
            vec.append((tf[w] / max(len(tokens), 1)) * math.log(N / (df[w] + 1)))
        else:
            vec.append(0.0)
    return vec

doc_vectors = [compute_tfidf(t) for t in all_tokens]
print(f"✅ {len(metas)} records, {len(vocab)} features")

def cosine(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    return dot / (n1 * n2) if n1 * n2 > 0 else 0

def search(query: str, n: int = 5):
    q_vec = compute_tfidf(tokenize(query))
    scores = [(i, cosine(q_vec, doc_vectors[i])) for i in range(N)]
    scores.sort(key=lambda x: -x[1])
    top = scores[:n]
    return [{"meta": metas[i], "doc": docs[i], "score": s} for i, s in top]

# === Routes ===
@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")

SOURCE_SCORE_THRESHOLD = 0.05  # bronnen onder deze relevantie tonen we niet

@app.get("/api/stats")
async def stats():
    theme_counts = Counter(r.get('theme', 'overig') for r in dataset["results"])
    return JSONResponse({
        "total": len(dataset["results"]),
        "themes": [{"theme": t, "count": c} for t, c in theme_counts.most_common()],
        "timestamp": dataset.get("timestamp", "")
    })

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Geen vraag"}, status_code=400)

    results = [r for r in search(question, n=8) if r['score'] >= SOURCE_SCORE_THRESHOLD]
    books = []
    for i, r in enumerate(results, 1):
        m = r['meta']
        status = "beschikbaar" if not m['onLoan'] else "uitgeleend"
        books.append(f"{i}. \"{m['title']}\" door {m['author'] or 'onbekend'} ({m['language']}, {status})")
    context = "\n".join(books) if books else "(geen relevante titels gevonden in de catalogus)"

    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    relevant = list(range(len(results)))  # fallback: alles tonen
    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "Je bent een bibliotheekassistent van bblthk Wageningen. Beantwoord de vraag op basis "
                    "van de genummerde catalogusresultaten. Noem titels, auteurs en beschikbaarheid. Wees kort, "
                    "vriendelijk, in het Nederlands. De resultaten komen uit woordmatching en kunnen missers "
                    "bevatten; negeer titels die niet bij de vraag passen. Sluit je antwoord af met een aparte "
                    "laatste regel in exact dit formaat: BRONNEN: gevolgd door de nummers van de resultaten "
                    "die echt bij de vraag passen, kommagescheiden (bijv. BRONNEN: 1,3). Passen er geen, "
                    "schrijf dan BRONNEN: geen."
                )},
                {"role": "user", "content": f"Vraag: {question}\n\nCatalogus:\n{context}\n\nBeantwoord met titels, auteurs en beschikbaarheid, en eindig met de BRONNEN-regel."}
            ],
            temperature=0.7, max_tokens=450
        )
        answer = resp.choices[0].message.content
        # BRONNEN-regel eruit vissen en van het antwoord strippen
        m = re.search(r'\n?\s*BRONNEN:\s*(.*?)\s*$', answer)
        if m:
            answer = answer[:m.start()].rstrip()
            nums = re.findall(r'\d+', m.group(1))
            relevant = [int(n) - 1 for n in nums if 0 < int(n) <= len(results)] if nums else []
    except Exception as e:
        answer = f"(Fout: {e})\n\n{context}"

    sources = [results[i] for i in relevant]
    return JSONResponse({
        "answer": answer,
        "sources": [{"title": r['meta']['title'], "author": r['meta']['author'], "available": not r['meta']['onLoan']} for r in sources]
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
    print(f"🚀 http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
