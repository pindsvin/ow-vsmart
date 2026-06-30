# OW-vsmart — Online deployen

## Snel: Render (gratis)

1. Ga naar https://render.com en maak een account (GitHub login)
2. Klik **New +** → **Web Service**
3. Connect je repository (of upload de bestanden handmatig)
4. Instellingen:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python chat_server_cloud.py`
5. Bij **Environment Variables**:
   - Key: `DEEPSEEK_API_KEY`
   - Value: jouw DeepSeek key
6. Klik **Deploy**

Eerste deploy duurt ~5 min (download embeddings model). Daarna op https://ow-vsmart.onrender.com

## Alternatief: Railway

1. https://railway.app → New Project → Deploy from GitHub
2. Zet `DEEPSEEK_API_KEY` in Variables
3. Deploy

## Wat gaat er mee

| Bestand | Functie |
|---------|---------|
| `chat_server_cloud.py` | FastAPI server |
| `catalogus_klimaat.json` | Dataset (90 records) |
| `chroma_db/` | ChromaDB (of herbouwd bij lege start) |
| `requirements.txt` | Dependencies |
| `.gitignore` | Sluit .env uit |

## Let op
- `.env` met je API key **nooit** committen! Render/Railway gebruiken hun eigen env vars.
- ChromaDB wordt bij lege start herbouwd uit `catalogus_klimaat.json`
- Eerste start duurt langer (embedding model downloaden), daarna snel
