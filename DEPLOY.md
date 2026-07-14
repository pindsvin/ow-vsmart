# OW-vsmart — Online deployen

Live demo: **https://ow-vsmart.onrender.com** (Render, gratis tier, auto-deploy vanaf GitHub `master`)

## Architectuur (cloud)

```
vraag → Gemini embedding-call (RETRIEVAL_QUERY, 768 dims)
      → numpy dot-product tegen voorberekende vectoren (embeddings_gemini.npz)
      → top-8 naar DeepSeek → antwoord + BRONNEN-regel
      → UI toont alleen de bronnen die DeepSeek relevant acht
```

- **Semantisch zoeken** zonder zwaar model op de server: de catalogus is
  vooraf geëmbed (op je eigen Mac, `embed_gemini.py`) en de vectoren gaan
  als klein bestand (~2MB) mee in de repo. Past ruim in de gratis 512MB.
- **Fallback**: ontbreekt `GEMINI_API_KEY` of het vectorbestand, dan draait
  de server automatisch op pure-Python TF-IDF (met Nederlandse stopwoorden).

## Render-instellingen

1. https://render.com → **New +** → **Web Service** → koppel de GitHub-repo
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `python chat_server_cloud.py`
4. **Environment Variables**:
   - `DEEPSEEK_API_KEY` — voor de chat-antwoorden (platform.deepseek.com)
   - `GEMINI_API_KEY` — voor query-embeddings én de meer-info-zoekfunctie
     (Google Search-grounding, ~1-3 ct per opzoeking; key aanmaken op
     aistudio.google.com, budget gekoppeld sinds 14-07-2026)
5. Elke push naar `master` deployt automatisch.

## Workflow bij datasetwijzigingen

```bash
# 1. catalogus_klimaat.json bijwerken (nieuwe entries, met 'theme'-veld)
python3 embed_gemini.py     # 2. vectoren opnieuw genereren (leest .env)
python3 embed_and_query.py  # 3. optioneel: lokale ChromaDB-index herbouwen
git add catalogus_klimaat.json embeddings_gemini.npz
git commit -m "..." && git push origin master   # 4. Render deployt vanzelf
```

Let op: het aantal vectoren in `embeddings_gemini.npz` moet gelijk zijn aan
het aantal entries in de dataset — de server controleert dit en valt anders
terug op TF-IDF (met een waarschuwing in de logs).

## Wat gaat er mee

| Bestand | Functie |
|---------|---------|
| `chat_server_cloud.py` | FastAPI server (semantisch + TF-IDF fallback) |
| `catalogus_klimaat.json` | Dataset (721 records, 20 thema's) |
| `embeddings_gemini.npz` | Voorberekende Gemini-embeddings (768 dims) |
| `static/` | Chat-UI (bblthk-huisstijl, stats, autocomplete) |
| `requirements.txt` | Dependencies (bewust licht: geen torch/chromadb) |

## Let op

- `.env` met API-keys **nooit** committen (staat in `.gitignore`) — Render
  gebruikt zijn eigen environment variables.
- `chroma_db/` is alleen voor de lokale server en is gitignored.
- De gratis Render-instance slaapt na inactiviteit; de eerste request
  daarna duurt ~1 minuut (cold start).
- Beschikbaarheid ("beschikbaar/uitgeleend") is een momentopname van de
  scrape-datum, geen live status.
